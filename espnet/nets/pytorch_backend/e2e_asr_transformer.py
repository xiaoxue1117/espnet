# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
import logging
import math

import numpy
import torch
import gtn

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

from espnet.nets.pytorch_backend.ctc_beam_search import ctcBeamSearch

from espnet.nets.pytorch_backend.allo import AlloLayer
from espnet.nets.pytorch_backend.allo_biphone_ctc import AlloBiCTCLayer
from espnet.nets.pytorch_backend.conv_allo import ConvAlloLayer

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, langdict, allo, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param dict langdict: {'phone' : phone_dim, 'langid' : phoneme_dim, ... }
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        if not hasattr(args, 'pn_dropout_rate') or args.pn_dropout_rate is None:
            args.pn_dropout_rate = args.dropout_rate
        if not hasattr(args, 'pn_transformer_attn_dropout_rate') or args.pn_transformer_attn_dropout_rate is None:
            args.pn_transformer_attn_dropout_rate = args.pn_dropout_rate

        alloWdict, alloGdict = allo

        # speech encoder
        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.decoder = None
        self.criterion = None

        self.langdict = langdict

        self.sos = {}
        self.eos = {}
        self.odim = odim
        for lid in alloWdict.keys():
            self.sos[lid] = odim[lid] - 1
            self.eos[lid] = odim[lid] - 1

        # phone output
        self.phone_out = torch.nn.Sequential(
                                torch.nn.Linear(args.adim, args.adim),
                                torch.nn.Dropout(args.dropout_rate),
                                torch.nn.Linear(args.adim, langdict['phone']))


        # allophone layer
        if hasattr(args, 'am_type'):
            self.am_type = args.am_type
        else:
            self.am_type = "graph"

        if hasattr(args, 'sm_allo'):
            sm_allo = args.sm_allo
        else:
            sm_allo = False

        if hasattr(args, 'sm_phonemes'):
            sm_phonemes = args.sm_phonemes
        else:
            sm_phonemes = False

        if hasattr(args, 'phoneme_bias'):
            phoneme_bias = args.phoneme_bias
        else:
            phoneme_bias = False

        if hasattr(args, 'full_constrained'):
            full_constrained = args.full_constrained
        else:
            full_constrained = False

        if hasattr(args, 'sm_after'):
            sm_after = args.sm_after
        else:
            sm_after = False

        if hasattr(args, 'use_gtn_ctc'):
            self.use_gtn_ctc = args.use_gtn_ctc
        else:
            self.use_gtn_ctc = False

        if hasattr(args, 'use_conv_allo'):
            self.use_conv_allo = args.use_conv_allo
        else:
            self.use_conv_allo = False

        self.alloW = torch.nn.ParameterDict()
        self.allodict = torch.nn.ModuleDict()
        for lid in alloWdict.keys():
            if self.am_type != "allomatbaseline":
                mask = torch.Tensor(alloWdict[lid]).sum(0).bool().unsqueeze(0).unsqueeze(0)
                if self.use_conv_allo:
                    logging.warning("Use conv allo: " + str(self.use_conv_allo))
                    self.allodict[lid] = ConvAlloLayer(allo_gtn=alloGdict[lid], idim=args.adim, n_phones=langdict['phone'], odim=langdict[lid], mask=mask, kernel=5)
                elif self.use_gtn_ctc:
                    self.allodict[lid] = AlloBiCTCLayer(alloGdict[lid], langdict[lid], langdict['phone'], args.trainable, args.redis, mask, lid, sm_allo, phoneme_bias)
                else:
                    self.allodict[lid] = AlloLayer(alloGdict[lid], langdict[lid], args.trainable, args.redis, mask, lid, sm_allo, phoneme_bias, sm_phonemes, full_constrained, sm_after)

                logging.warning("Setting allograph weights as trainable:" + str(args.trainable))
            else:
                self.alloW[lid] = torch.nn.Parameter(torch.Tensor(alloWdict[lid]))
                if not hasattr(args, 'alloW_grad'):
                    self.alloW[lid].requires_grad = False
                else:
                    self.alloW[lid].requires_grad = args.alloW_grad

        self.allotype = 'avg'
        if hasattr(args, 'allotype'):
            self.allotype = args.allotype #'max'

        # baseline w/o allophone
        if self.allotype == 'none':
            self.phoneme_out = torch.nn.ModuleDict()
            for lid in alloWdict.keys():
                self.phoneme_out[lid] = torch.nn.Sequential(
                                torch.nn.Linear(args.adim, args.adim),
                                torch.nn.Dropout(args.dropout_rate),
                                torch.nn.Linear(args.adim, langdict[lid]))
        self.blank = 0
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        self.reset_parameters(args)
        self.adim = args.adim  # used for CTC (equal to d_model)
        self.mtlalpha = args.mtlalpha

        self.ctc = torch.nn.ModuleDict()
        for lid in alloWdict.keys():
            self.ctc[lid] = CTC(
                langdict[lid], langdict[lid], args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, ys_ph_pad, cats):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # setup batch
        if len(set(cats)) > 1:
            logging.warning("Batch is mixed")
            logging.warning(cats)
        lid = cats[0]
        batch_size = xs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)

        if self.use_conv_allo == True:
            out = self.phone_out(hs_pad)
            hs_pad = self.allodict[lid](out, hs_pad)    #out = phoneme_emissions
            loss_am, _ = self.ctc[lid](hs_pad, hs_len, ys_ph_pad)

        elif self.use_gtn_ctc == True:
            hs_pad = self.phone_out(hs_pad)
            loss_am = self.allodict[lid](hs_pad, ys_ph_pad)
        else:
            if self.allotype == 'none':
                hs_pad = self.phoneme_out[lid](hs_pad)
                hs_pad = hs_pad.log_softmax(dim=-1)
            else:
                # phone logits
                hs_pad = self.phone_out(hs_pad)

                if self.am_type != 'allomatbaseline':
                    # am CTC
                    hs_pad = self.allodict[lid](hs_pad)    #hs_pad = phoneme_emissions
                else:
                    hs_pad = hs_pad.unsqueeze(2) * self.alloW[lid].unsqueeze(0).unsqueeze(0)
                    #hs_pad = hs_pad.sum(dim=-1)
                    hs_pad = hs_pad.max(dim=-1)[0]
                    hs_pad = hs_pad.log_softmax(dim=-1)

            # input is hs_pad which is already log_sm
            loss_am, _ = self.ctc[lid](hs_pad, hs_len, ys_ph_pad)

        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_ph_hat = self.ctc[lid].argmax(hs_pad.view(batch_size, -1, self.langdict[lid])).data
            cer_ctc = self.error_calculator(ys_ph_hat.cpu(), ys_ph_pad.cpu(), is_ctc=True)
        # for visualization
        if not self.training:
            self.ctc[lid].softmax(hs_pad)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_pn
            loss_pn_data = float(loss_pn)
            loss_am_data = None
        elif alpha == 1:
            self.loss = loss_am
            loss_pn_data = None
            loss_am_data = float(loss_am)
            self.acc = 0
        else:
            self.loss = alpha * loss_am + (1 - alpha) * loss_pn
            loss_pn_data = float(loss_pn)
            loss_am_data = float(loss_am)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_am_data, loss_pn_data, self.acc, None, None, None, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def recognize(self, x, cat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        #import numpy as np
        #for lid in self.allodict.keys():
        #    #dst = "/project/ocean/byan/espnet-ml/egs/babel/asr1/exp/train_swbd_pytorch_gtn-traintrue-redisfalse-fixedSM_specaug_ngpu3/alloWDense50_numpy_"+lid
        #    #np_alloWDense = self.allodict[lid].alloWDense.detach().cpu().numpy()
        #    #np.save(dst, np_alloWDense)
        #    #dst = "/project/ocean/byan/espnet-ml/egs/babel/asr1/exp/train_swbd_pytorch_gtn-traintrue-redisfalse-fixedSM_specaug_ngpu3/alloW_numpy_"+lid
        #    #np_alloW = self.allodict[lid].alloWDense.log_softmax(dim=-1)[self.allodict[lid].alloWMask==True].detach().cpu().numpy()
        #    #np.save(dst, np_alloW)
        #    dst = "/project/ocean/byan/espnet-ml/egs/babel/asr1/exp/train_swbd_pytorch_gtn-traintrue-redisfalse-SMconstraint-phonemebias_specaug_ngpu3/alloW_numpy_"+lid
        #    #np_alloW = self.allodict[lid].alloW.log_softmax(dim=-1).detach().cpu().numpy()
        #    np_alloW = self.allodict[lid].get_alloW_SM().detach().cpu().numpy()
        #    np.save(dst, np_alloW)
        #import pdb; pdb.set_trace()
        if cat not in self.alloW.keys():
            if cat == 'en':
                cat = '000'

        #alignments
        align = [[1], [1], [1]]

        logging.info("langid: %s", cat)
        hs_pad = self.encode(x).unsqueeze(0)

        if self.allotype == 'none':
            hs_pad = self.phoneme_out[cat](hs_pad)
            hs_pad = hs_pad.log_softmax(dim=-1)

            m_out = hs_pad.max(dim=-1)[1].squeeze()
            align[1] = m_out.tolist()   # phonemes
        else:
            phone_hs = self.phone_out(hs_pad)
            if self.use_conv_allo == True:
                hs_pad = self.allodict[lid](phone_hs, hs_pad)    #out = phoneme_emissions
            elif self.use_gtn_ctc == True:
                hs_pad = self.allodict[cat].predict(phone_hs)
            elif self.am_type != 'allomatbaseline':
                # am CTC
                hs_pad = self.allodict[cat](phone_hs, training=False)    #hs_pad = phoneme_emissions
            else:
                hs_pad = phone_hs.unsqueeze(2) * self.alloW[cat].unsqueeze(0).unsqueeze(0)
                #hs_pad = hs_pad.sum(dim=-1)
                hs_pad = hs_pad.max(dim=-1)[0]
                hs_pad = hs_pad.log_softmax(dim=-1)

            n_out = phone_hs.max(dim=-1)[1].squeeze()
            align[0] = n_out.tolist()   # phones
            m_out = hs_pad.max(dim=-1)[1].squeeze()
            align[1] = m_out.tolist()   # phonemes

        from itertools import groupby
        collapsed_indices = [x[0] for x in groupby(m_out)]
        ph_hyps = [x.item() for x in filter(lambda x: x != self.blank, collapsed_indices)]
        return [{"score" : 0.0, "yseq" : [self.sos[cat]]}], ph_hyps, align

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_ph_pad, cats):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        return 0.0
        #self.eval()
        #with torch.no_grad():
        #    self.forward(xs_pad, ilens, ys_pad, ys_ph_pad, cats)
        #ret = dict()
        #for name, m in self.named_modules():
        #    if (
        #        isinstance(m, MultiHeadedAttention)
        #        or isinstance(m, DynamicConvolution)
        #        or isinstance(m, RelPositionMultiHeadedAttention)
        #    ):
        #        ret[name] = m.attn.cpu().numpy()
        #    if isinstance(m, DynamicConvolution2D):
        #        ret[name + "_time"] = m.attn_t.cpu().numpy()
        #        ret[name + "_freq"] = m.attn_f.cpu().numpy()
        #self.train()
        #return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, ys_ph_pad, cats):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        return 0.0
        #ret = None
        #if self.mtlalpha == 0:
        #    return ret

        #self.eval()
        #with torch.no_grad():
        #    self.forward(xs_pad, ilens, ys_pad)
        #for name, m in self.named_modules():
        #    if isinstance(m, CTC) and m.probs is not None:
        #        ret = m.probs.cpu().numpy()
        #self.train()
        #return ret
