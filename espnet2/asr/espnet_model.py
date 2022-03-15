import contextlib
from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        freeze_finetune_updates: int = 1000000000,
        apply_moe_on: str = "hubert",
        layer_selection_hubert: str = "12",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        if (hasattr(self.frontend, "align_method") and self.frontend.align_method == "elevator"):
            self.project_hubert = torch.nn.Linear(in_features=self.frontend.output_size_s3prl(),
                                                  out_features=self.encoder._output_size)
            self.MOE_n_experts = 2
            self.MOE_proj=torch.nn.Linear(in_features=self.encoder._output_size, out_features=self.MOE_n_experts)
            self.layer_selection_hubert = [int(x) for x in layer_selection_hubert.split()]

        if (hasattr(self.frontend, "align_method") and self.frontend.align_method == "encoder_linear_fusion"):
            self.project_hubert = torch.nn.Linear(in_features=self.frontend.output_size_s3prl(),
                                                  out_features=self.encoder._output_size)
            self.project_final = torch.nn.Linear(in_features=2*self.encoder._output_size, out_features=self.encoder._output_size)

        # autre idée aussi : drop some frames --> mettre 3 experts
        self.use_transducer_decoder = joint_network is not None
        self.num_updates=0
        self.freeze_finetune_updates = freeze_finetune_updates
        self.apply_moe_on=apply_moe_on
        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        store: bool=False, path: str="",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]
       
        stop_ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif stop_ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Stop tuning MOE parameters")
        else:
            self.num_updates += 1


        # 1. Encoder
        if (hasattr(self.frontend, "align_method") and self.frontend.align_method == "elevator" and store):
            encoder_out, encoder_out_lens, mat_moe = self.encode(speech, speech_lengths, stop_ft,store)
            self.mat_moe=mat_moe
        else:
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

            # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att

        # Collect total loss stats
        stats["loss"] = loss.detach()


        if False and hasattr(self.frontend, "align_method") and self.frontend.align_method == "elevator" :
            loss_ctc_hub, _ = self._calc_ctc_loss(
                self.feats_hubert, self.feats_lengths_hubert, text, text_lengths
            )
            logging.info("ctc hubert : ",loss_ctc_hub)
            loss = loss + 0.1*loss_ctc_hub

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            if (
                hasattr(self.frontend, "align_method")
                and (self.frontend.align_method == "elevator" or self.frontend.align_method=="encoder_linear_fusion")
            ):
                feats, feats_lengths, feats_hubert, feats_lengths_hubert  = self._extract_feats(speech, speech_lengths)
            else :
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor, stop_ft: bool=False, store: bool=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            if (hasattr(self.frontend, "align_method") and (self.frontend.align_method == "elevator" or self.frontend.align_method=="encoder_linear_fusion")):
                # 1. Extract feats
                feats, feats_lengths, feats_hubert, feats_lengths_hubert = self._extract_feats(speech, speech_lengths)

                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)
                    feats_hubert, feats_lengths_hubert = self.specaug(feats_hubert, feats_lengths_hubert)
                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)
                    feats_hubert, feats_lengths_hubert = self.normalize(feats_hubert, feats_lengths_hubert)
                # Pre-encoder, e.g. used for raw input data but here used as a linear layer only ! 

                #a,b,c = feats_hubert.shape
                #self.feats_hubert=feats_hubert.view(13,a,b,-1)
                #self.feats_lengths_hubert=feats_lengths_hubert.view(13,a,b,-1)

                #self.feats_hubert=self.feats_hubert[x for x in self.layer_selection_hubert]


                self.feats_hubert = self.project_hubert(feats_hubert)


            else:
                # 1. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)

                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        #assert 6==0
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )


        if (hasattr(self.frontend, "align_method") and self.frontend.align_method == "elevator"):

            # mettre la gate ICI sur MFCC only :
            with torch.no_grad() if (stop_ft and False) else contextlib.nullcontext():
                if self.apply_moe_on == "hubert":
                    MOE_weights = self.MOE_proj(self.feats_hubert)
                else:
                    assert 8==0
                    MOE_weights = self.MOE_proj(encoder_out)
                # MOE_weights=torch.nn.functional.softmax(MOE_weights, dim=-1)
                # assert 6==0, MOE_weights.shape
                MOE_weights = torch.nn.functional.log_softmax(MOE_weights, dim=-1)
            # assert 6==0, MOE_weights.shape


            # fusion part, just weighted sum with an alpha here !! 
            # drop few frames : 
            #logging.info("{} {}".format(encoder_out.shape,feats_hubert.shape))
            m = min(feats_hubert.shape[1],encoder_out.shape[1])
            diff = max(feats_hubert.shape[1]-m, encoder_out.shape[1]-m)
            assert diff<8, "we had to drop {} frames, this seems to be too much".format(diff)
            encoder_out, feats_hubert, MOE_weights = encoder_out[:,:m,:], self.feats_hubert[:,:m,:], MOE_weights[:,:m,:]

            a, b, c = encoder_out.shape
            #w_fbank = MOE_weights[:,:,0].expand(c,a,b).permute(1,2,0)
            #w_hub = [MOE_weights[:,:,1+i].expand(c,a,b).permute(1,2,0) for i in range(len(self.layer_selection_hubert))]

            #w1, w2 = MOE_weights[:,:,:,0], MOE_weights[:,:,:,1]
            #alpha=self.frontend.alpha
            #encoder_out = alpha*feats_hubert + (1-alpha)*encoder_out
           # logging.info("hubert/mfcc weights : {}".format(MOE_weights))
            #assert 9==0, (w_fbank.shape, w_hub[0].shape)

            w1, w2 = MOE_weights[:,:,0].expand(c,a,b), MOE_weights[:,:,1].expand(c, a, b)
            w1, w2 = w1.permute(1,2,0), w2.permute(1,2,0)
            #logging.info(feats_hubert.shape)
            #logging.info(encoder_out.shape)
            encoder_out = w1*feats_hubert + w2*encoder_out

            if store:
                return encoder_out, encoder_out_lens, MOE_weights


        if (hasattr(self.frontend, "align_method") and self.frontend.align_method == "encoder_linear_fusion"):

            # fusion part, just weighted sum with an alpha here !!
            # drop few frames :
            #logging.info("{} {}".format(encoder_out.shape,feats_hubert.shape))
            m = min(feats_hubert.shape[1],encoder_out.shape[1])
            diff = max(feats_hubert.shape[1]-m, encoder_out.shape[1]-m)
            assert diff<8, "we had to drop {} frames, this seems to be too much".format(diff)
            encoder_out, feats_hubert = encoder_out[:,:m,:], self.feats_hubert[:,:m,:]


            encoder_out = self.project_final(torch.cat((feats_hubert, encoder_out), dim=-1))

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)

            if (
                hasattr(self.frontend, "align_method")
                and (self.frontend.align_method == "elevator" or self.frontend.align_method == "encoder_linear_fusion")
            ):
                (
                    feats,
                    feats_lengths,
                    feats_hubert,
                    feats_lengths_hubert,
                ) = self.frontend(speech, speech_lengths)
                return (
                    feats,
                    feats_lengths,
                    feats_hubert,
                    feats_lengths_hubert,
                )
            else:
                feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer
