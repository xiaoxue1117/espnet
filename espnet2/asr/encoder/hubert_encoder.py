# Copyright 2021 Tianzi Wang
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0

# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert


"""Encoder definition."""
import contextlib
import copy
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import math
import torch
from torch import nn
import yaml
from filelock import FileLock
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class FairseqHubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module, used for loading pretrained weight and finetuning

    Args:
        input_size: input dim
        hubert_url: url to Hubert pretrained model
        hubert_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        output_size: dimension of attention
        normalize_before: whether to use layer_norm before the first block
        freeze_finetune_updates: steps that freeze all layers except output layer
            before tuning the whole model (nessasary to prevent overfit).
        dropout_rate: dropout rate
        activation_dropout: dropout rate in activation function
        attention_dropout: dropout rate in attention
    Hubert specific Args:
        Please refer to:
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py
    """

    def __init__(
        self,
        input_size: int,
        hubert_url: str = "./",
        hubert_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str = "static",
        layerdrop: float = 0.0,
        feature_grad_mult: float = 0.0,
        heads: bool = False, 
        attention_in_heads: bool = False,
        heads_ff_dim: int = 512,
        heads_att_dim: int = 512, 
        heads_layer_list: str = "5_10_15_20"
    ):
        assert check_argument_types()
        super().__init__()
        self.apply_mask = apply_mask
        try:
            import fairseq
            from fairseq.models.hubert.hubert import HubertModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": hubert_dir_path,
        }

        if hubert_url == "espnet":
            self.hubert_model_path = hubert_dir_path
            s = torch.load(
                self.hubert_model_path,
                map_location=torch.device("cpu"),
            )

            if all("encoder.encoder" in k for k in s):
                try:
                    state = {
                        k.replace("encoder.encoder.", ""): v
                        for k, v in s.items()
                        if "label_embs_concat" not in k
                    }
                except Exception as e:
                    raise e

            config_file = os.path.join(
                "/".join(self.hubert_model_path.split("/")[:-1]),
                "config.yaml",
            )
            config_file = Path(config_file)

            with config_file.open("r", encoding="utf-8") as f:
                self.pretrained_cfg = yaml.safe_load(f)

            model = FairseqHubertPretrainEncoder(
                input_size=self.pretrained_cfg["input_size"],
                hubert_dict=self.pretrained_cfg["hubert_dict"],
                **self.pretrained_cfg["encoder_conf"],
            )
            model = model.encoder

            d = self.pretrained_cfg["encoder_conf"]["output_size"]
            self.pretrained_params = copy.deepcopy(state)

        else:

            self.hubert_model_path = download_hubert(hubert_url, hubert_dir_path)
            
            (
                models,
                self.pretrained_cfg,
                task,
            ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path],
                arg_overrides=arg_overrides,
                strict=False,
            )
            model = models[0]
            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())

        self._output_size = output_size

        if not isinstance(model, HubertModel):
            try:
                model = model.hubert_encoder.hubert_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'HubertModel, Hubertctc' classes, etc."
                )
                raise e

        self.encoders = model

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(d, output_size),
            )
        else:
            self.output_layer = None
        
        # Early Exit code 
        self.heads = heads 
        self.heads_layer_list = [int(a) for a in heads_layer_list.split("_")]
        if self.heads:
            self.heads_ff_dim = heads_ff_dim
            self.n_ee_heads = len(self.heads_layer_list)
        self.attention_in_heads = heads and attention_in_heads
        if self.attention_in_heads :
            self.heads_att_dim = heads_att_dim

        if (self.heads and self.attention_in_heads) :
            self.ee_heads = torch.nn.ModuleList(
                torch.nn.Sequential(
                    nn.Linear(d, d), 
                    MultiHead_Attn(n_heads=1, emb_dim=d, qk_dim=self.heads_att_dim, v_dim=self.heads_att_dim, dropout=0.3), 
                    FeedForward(d, self.heads_ff_dim, dropout=0.3),  # tune dropout 
                )
                for i in range(self.n_ee_heads)
            )
        elif (self.heads) :
            self.ee_heads = torch.nn.ModuleList(
                torch.nn.Sequential(
                    nn.Linear(d, d),
                    FeedForward(d, self.heads_ff_dim, dropout=0.3),  # tune dropout  
                )
                for i in range(self.n_ee_heads)
            )

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))
    
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert ASR Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1    
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                padding_mask=masks,
                mask=self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)
        layer_results = enc_outputs["layer_results"] # à voir!

        if self.heads:
            intermediate_outs = [(i, layer_results[i][0]) for i in self.heads_layer_list]  # here modify to have only certain heads
        else : # write it better but I need this also if no heads but still intermediate CTC
            intermediate_outs = [(i, layer_results[i][0]) for i in self.heads_layer_list] 

        # save gpu memory
        del enc_outputs

        olens = (~masks).sum(dim=1)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        if self.heads : # add a param later
            return (xs_pad, intermediate_outs), olens, None

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        #for name, para in self.encoders.named_parameters():
        #    print('{}: {}'.format(name, para.shape))
        #assert 5==0
        self.encoders.load_state_dict(self.pretrained_params, strict=False)
        #assert 9==0, (len(self.pretrained_params.keys() & self.encoders.state_dict().keys()), len(self.pretrained_params.keys()), len(self.encoders.state_dict().keys()) 
        logging.info("Pretrained Hubert model parameters reloaded!")

    #def reload_pretrained_parameters_ctc(self):



class FairseqHubertPretrainEncoder(AbsEncoder):
    """FairSeq Hubert pretrain encoder module, only used for pretraining stage

    Args:
        input_size: input dim
        output_size: dimension of attention
        linear_units: dimension of feedforward layers
        attention_heads: the number of heads of multi head attention
        num_blocks: the number of encoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        hubert_dict: target dictionary for Hubert pretraining
        label_rate: label frame rate. -1 for sequence label
        sample_rate: target sample rate.
        use_amp: whether to use automatic mixed precision
        normalize_before: whether to use layer_norm before the first block
    """

    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1024,
        linear_units: int = 1024,
        attention_heads: int = 12,
        num_blocks: int = 12,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        activation_dropout_rate: float = 0.0,
        hubert_dict: str = "./dict.txt",
        label_rate: int = 100,
        checkpoint_activations: bool = False,
        sample_rate: int = 16000,
        use_amp: bool = False,
        **kwargs,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.use_amp = use_amp
        try:
            from fairseq.data.dictionary import Dictionary
            from fairseq.models.hubert.hubert import HubertConfig  # noqa: H301
            from fairseq.models.hubert.hubert import HubertModel  # noqa: H301
            from fairseq.models.hubert.hubert import (  # noqa: H301
                HubertPretrainingConfig,
            )
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print("Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done")
            raise e

        cfg_overides = {
            "encoder_embed_dim": output_size,
            "encoder_ffn_embed_dim": linear_units,
            "encoder_attention_heads": attention_heads,
            "encoder_layers": num_blocks,
            "final_dim": output_size,
            "dropout": dropout_rate,
            "attention_dropout": attention_dropout_rate,
            "label_rate": label_rate,
            "checkpoint_activations": checkpoint_activations,
        }
        cfg_overides = {**cfg_overides, **kwargs}
        self.cfg = HubertConfig()

        for key, value in cfg_overides.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        hubert_task_cfg = HubertPretrainingConfig()
        hubert_task_cfg_overides = {
            "label_rate": label_rate,
            "sample_rate": sample_rate,
        }
        for key, value in hubert_task_cfg_overides.items():
            if hasattr(hubert_task_cfg, key):
                setattr(hubert_task_cfg, key, value)

        d = Dictionary()
        self._build_dictionary(d, hubert_dict)
        self.encoder = HubertModel(self.cfg, hubert_task_cfg, self.dictionaries)

    def _build_dictionary(self, dictionary, hubert_dict_path):
        if os.path.exists(f"{hubert_dict_path}"):
            setattr(dictionary, "symbols", [])
            setattr(dictionary, "count", [])
            setattr(dictionary, "indices", {})
            dictionary.add_from_file(f"{hubert_dict_path}")
        else:
            dictionary.add_symbol("0")

        self.dictionaries = [dictionary]

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward Hubert Pretrain Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        self.cast_mask_emb()
        masks = make_pad_mask(ilens).to(xs_pad.device)
        ys_pad = ys_pad[:, : min(ys_pad_length)]
        enc_outputs = self.encoder(
            xs_pad,
            padding_mask=masks,
            mask=True,
            target_list=[ys_pad],
            features_only=False,
        )
        return enc_outputs

    def cast_mask_emb(self):
        if self.use_amp and self.encoder.mask_emb.dtype != torch.cuda.HalfTensor:
            self.encoder.mask_emb = torch.nn.Parameter(self.encoder.mask_emb.half())

    def reload_pretrained_parameters(self):
        self.encoder.mask_emb = torch.nn.Parameter(
            torch.HalfTensor(self.cfg.encoder_embed_dim).uniform_()
        )
        logging.info(
            f"Hubert mask embedding re-initiallized!, \
            {self.encoder.mask_emb.dtype}, \
            {self.use_amp}"
        )


def download_hubert(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)
    if False : # often problems from FAIRSEQ, download manually and then skip this phase !!
        with FileLock(model_path + ".lock"):
            if not os.path.exists(model_path):
                torch.hub.download_url_to_file(model_url, model_path)
                logging.info(f"Hubert model downloaded {model_path}")
            else:
                logging.info(f"Hubert model {model_path} already exists.")
    logging.info(f"Hubert model {model_path} already exists.")

    return model_path




class FeedForward(torch.nn.Module):
  def __init__(self, emb_dim, ff_dim, dropout):
    super(FeedForward, self).__init__()
    self.FF_in = torch.nn.Linear(emb_dim, ff_dim)
    self.FF_out = torch.nn.Linear(ff_dim, emb_dim)
    self.dropout = torch.nn.Dropout(dropout) #this regularization is not used in the original model

  def forward(self, x):
    tmp = self.FF_in(x)
    tmp = torch.nn.functional.relu(tmp)
    tmp = self.dropout(tmp)
    tmp = self.FF_out(tmp)
    tmp = self.dropout(tmp)
    return tmp



##############################################################################################################
### MultiHead_Attn ###########################################################################################
##############################################################################################################
class MultiHead_Attn(torch.nn.Module):
  def __init__(self, n_heads, emb_dim, qk_dim, v_dim, dropout):
    super(MultiHead_Attn, self).__init__()
    self.nh = n_heads
    self.ed = emb_dim
    self.qd = qk_dim
    self.kd = qk_dim
    self.vd = v_dim
    self.WQ = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WK = torch.nn.Linear(emb_dim, qk_dim*n_heads)
    self.WV = torch.nn.Linear(emb_dim, v_dim*n_heads)
    self.WO = torch.nn.Linear(v_dim*n_heads, emb_dim)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, q, k=None, v=None, msk=None):
    #q is [bs, lq, ed]
    #k is [bs, lk, ed]
    #v is [bs, lv, ed]
    #msk is [bs, 1, ls] or [bs, lt, lt]
    #logging.info('q = {}'.format(q.shape))
    #logging.info('k = {}'.format(k.shape))
    #logging.info('v = {}'.format(v.shape))

    if k is None : 
        k = q 
        v=q 

    if msk is not None:
      msk = msk.unsqueeze(1) #[bs, 1, 1, ls] or [bs, 1, lt, lt]

    #logging.info('msk = {}'.format(msk.shape))

    bs = q.shape[0]
    lq = q.shape[1] ### sequence length of q vectors (length of target sentences)
    lk = k.shape[1] ### sequence length of k vectors (may be length of source/target sentences)
    lv = v.shape[1] ### sequence length of v vectors (may be length of source/target sentences)
    ed = q.shape[2]
    assert self.ed == q.shape[2] == k.shape[2] == v.shape[2]
    assert lk == lv #when applied in decoder both refer the source-side (lq refers the target-side)
    Q = self.WQ(q).contiguous().view([bs,lq,self.nh,self.qd]).permute(0,2,1,3) #=> [bs,lq,nh*qd] => [bs,lq,nh,qd] => [bs,nh,lq,qd]
    K = self.WK(k).contiguous().view([bs,lk,self.nh,self.kd]).permute(0,2,1,3) #=> [bs,lk,nh*kd] => [bs,lk,nh,kd] => [bs,nh,lk,kd]
    V = self.WV(v).contiguous().view([bs,lv,self.nh,self.vd]).permute(0,2,1,3) #=> [bs,lv,nh*vd] => [bs,lv,nh,vd] => [bs,nh,lv,vd]
    #Scaled dot-product Attn from multiple Q, K, V vectors (bs*nh*l vectors)
    Q = Q / math.sqrt(self.kd)
    s = torch.matmul(Q, K.transpose(2, 3)) #[bs,nh,lq,qd] x [bs,nh,kd,lk] = [bs,nh,lq,lk] # thanks to qd==kd #in decoder lq are target words and lk are source words

    #logging.info('s = {}'.format(s.shape))

    if msk is not None:
      s = s.masked_fill(msk == 0, float('-inf')) #score=-Inf to masked tokens
    w = torch.nn.functional.softmax(s, dim=-1) #[bs,nh,lq,lk] (these are the attention weights)
    #### Minh uses relu instead of softmax: w = torch.nn.functional.relu(s)
    w = self.dropout(w) #[bs,nh,lq,lk] 

    z = torch.matmul(w,V) #[bs,nh,lq,lk] x [bs,nh,lv,vd] = [bs,nh,lq,vd] #thanks to lk==lv
    z = z.transpose(1,2).contiguous().view([bs,lq,self.nh*self.vd]) #=> [bs,lq,nh,vd] => [bs,lq,nh*vd]
    z = self.WO(z) #[bs,lq,ed]
    return self.dropout(z)