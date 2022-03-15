import copy
from typing import Optional
from typing import Tuple
from typing import Union
from argparse import Namespace
import logging
import os
import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.nets_utils import pad_list


def base_s3prl_setup(args):
    args.upstream_feature_selection = getattr(args, "upstream_feature_selection", None)
    args.upstream_model_config = getattr(args, "upstream_model_config", None)
    args.upstream_refresh = getattr(args, "upstream_refresh", False)
    args.upstream_ckpt = getattr(args, "upstream_ckpt", None)
    args.init_ckpt = getattr(args, "init_ckpt", None)
    args.verbose = getattr(args, "verbose", False)
    args.tile_factor = getattr(args, "tile_factor", 1)
    return args


class Default_S3prl_Frontend(AbsFrontend):
    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf_default: Optional[dict] = get_default_kwargs(Frontend),
        frontend_conf_s3prl: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        download_dir: str = None,
        multilayer_feature: bool = False,
        align_method: str = "linear_projection",
        store_moe_path: str = "MOE_weights",
        alpha: float = 0.5,
        apply_frontend_moe_on: str = "hubert",
    ):

        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        ####### DEFAULT RELATED
        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf_default = copy.deepcopy(frontend_conf_default)

        self.fbanks_hop_length = hop_length

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf_default is not None:
            self.frontend_default = Frontend(
                idim=n_fft // 2 + 1, **frontend_conf_default
            )
        else:
            self.frontend_default = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.align_method = align_method
        self.alpha=alpha
        self.store_moe_path=store_moe_path

        ######### S3PRL RELATED
        if download_dir is not None:
            torch.hub.set_dir(download_dir)

        self.multilayer_feature = multilayer_feature
        self.upstream, self.featurizer = self._get_upstream(frontend_conf_s3prl)
        if getattr(
            self.upstream, "model", None
        ) is not None and self.upstream.model.__class__.__name__ in [
            "Wav2Vec2Model",
            "HuberModel",
        ]:
            self.upstream.model.encoder.layerdrop = 0.0
        self.pretrained_params = copy.deepcopy(self.upstream.state_dict())
        self.output_dim_s3prl = self.featurizer.output_dim
        # assert 9 == 0 , self.output_dim_s3prl

        ################### combine s3prl and default related
        self.gcd = np.gcd(self.fbanks_hop_length, self.s3prl_hop_length)
        self.factor_for_s3prl, self.factor_for_fbanks = (
            self.fbanks_hop_length // self.gcd,
            self.s3prl_hop_length // self.gcd,
        )

        if self.align_method == "linear_projection":
            self.projection_layer_s3prl = torch.nn.Linear(
                in_features=self.output_dim_s3prl,
                out_features=self.factor_for_s3prl * self.n_mels,
            )  # this way we have 80 and not 1024 for s3prl --> more balcances with fbanks
            self.projection_layer_fbanks = torch.nn.Linear(
                in_features=self.n_mels,
                out_features=self.factor_for_fbanks * self.n_mels,
            )

        if self.align_method == "linear_projection_plus":
            self.projection_layer_s3prl = torch.nn.Linear(
                in_features=self.output_dim_s3prl,
                out_features=self.factor_for_s3prl * self.n_mels,
            )  # this way we have 80 and not 1024 for s3prl --> more balcances with fbanks
            self.projection_layer_fbanks = torch.nn.Linear(
                in_features=self.n_mels,
                out_features=self.factor_for_fbanks * self.n_mels,
            )

        elif self.align_method == "coattention":
            self.projection_layer_s3prl = torch.nn.Linear(
                in_features=self.output_dim_s3prl,
                out_features=self.factor_for_s3prl * self.n_mels,
            )  # this way we have 80 and not 1024 for s3prl --> more balcances with fbanks
            self.projection_layer_fbanks = torch.nn.Linear(
                in_features=self.n_mels,
                out_features=self.factor_for_fbanks * self.n_mels,
            )
            self.sqrt_dim = np.sqrt(self.n_mels)
            self.dropout = torch.nn.Dropout(0.2)

        elif self.align_method == "conv":
            self.conv_align_s3prl = torch.nn.Conv1d(
                self.output_dim_s3prl,
                self.factor_for_s3prl * self.n_mels,
                kernel_size=self.factor_for_s3prl * 2 + 1,
                padding=self.factor_for_s3prl,
            )
            self.conv_align_default = torch.nn.Conv1d(
                self.n_mels,
                self.factor_for_fbanks * self.n_mels,
                kernel_size=self.factor_for_fbanks * 2 + 1,
                padding=self.factor_for_fbanks,
            )

        elif self.align_method == "rnn":
            self.lstm_align_s3prl = torch.nn.LSTM(
                self.output_dim_s3prl,
                self.factor_for_s3prl * self.n_mels,
                batch_first=True,
            )
            self.lstm_align_default = torch.nn.LSTM(
                self.n_mels, self.factor_for_fbanks * self.n_mels, batch_first=True
            )

        # frontend_moe
        elif self.align_method == "frontend_moe":
            self.projection_layer_s3prl = torch.nn.Linear(
                in_features=self.output_dim_s3prl,
                out_features=self.factor_for_s3prl * self.n_mels,
            )  # this way we have 80 and not 1024 for s3prl --> more balcances with fbanks
            self.projection_layer_fbanks = torch.nn.Linear(
                in_features=self.n_mels,
                out_features=self.factor_for_fbanks * self.n_mels,
            )
            self.moe_layer = torch.nn.linear(
                in_features=self.n_mels, out_features=2,
            )
            self.apply_frontend_moe_on=apply_frontend_moe_on


    #######################################
    ##   default frontend functions
    #######################################

    def output_size_default(self) -> int:
        return self.n_mels

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens

    #######################################
    ##   s3prl frontend functions
    #######################################

    def output_size_s3prl(self) -> int:
        return self.output_dim_s3prl

    def _get_upstream(self, frontend_conf):
        """Get S3PRL upstream model."""
        s3prl_args = base_s3prl_setup(
            Namespace(**frontend_conf, device="cpu"),
        )
        self.args = s3prl_args

        s3prl_path = None
        python_path_list = os.environ.get("PYTHONPATH", "(None)").split(":")
        for p in python_path_list:
            if p.endswith("s3prl"):
                s3prl_path = p
                break
        assert s3prl_path is not None

        s3prl_upstream = torch.hub.load(
            s3prl_path,
            s3prl_args.upstream,
            ckpt=s3prl_args.upstream_ckpt,
            model_config=s3prl_args.upstream_model_config,
            refresh=s3prl_args.upstream_refresh,
            source="local",
        ).to("cpu")

        from s3prl.upstream.interfaces import Featurizer

        self.s3prl_hop_length = s3prl_upstream.get_downsample_rates("key")

        if self.multilayer_feature is None:
            feature_selection = "last_hidden_state"
        else:
            feature_selection = "hidden_states"
        s3prl_featurizer = Featurizer(
            upstream=s3prl_upstream,
            feature_selection=feature_selection,
            upstream_device="cpu",
        )

        return s3prl_upstream, s3prl_featurizer

    def _tile_representations(self, feature):
        def _tile_representations(self, feature):
            """Tile up the representations by `tile_factor`.

            Input - sequence of representations
                    shape: (batch_size, seq_len, feature_dim)
            Output - sequence of tiled representations
                     shape: (batch_size, seq_len * factor, feature_dim)
            """
            assert (
                len(feature.shape) == 3
            ), "Input argument `feature` has invalid shape: {}".format(feature.shape)
            tiled_feature = feature.repeat(1, 1, self.args.tile_factor)
            tiled_feature = tiled_feature.reshape(
                feature.size(0),
                feature.size(1) * self.args.tile_factor,
                feature.size(2),
            )
            return tiled_feature

    def reload_pretrained_parameters(self):
        self.upstream.load_state_dict(self.pretrained_params)
        logging.info("Pretrained S3PRL frontend model parameters reloaded!")

    #######################################
    ##   mixed forward
    #######################################

    def output_size(self) -> int:
        if self.align_method == "elevator" or self.align_method=="frontend_moe" or self.align_method=="encoder_linear_fusion" :
            return self.output_size_default()
        return self.output_size_default() + self.output_size_s3prl()  # a verifier

    def forward_default(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens_default = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens_default = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend_default is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend_default(input_stft, feats_lens_default)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats_default: (Batch, Length, Dim)
        input_feats_default, _ = self.logmel(input_power, feats_lens_default)

        return input_feats_default, feats_lens_default

    def forward_s3prl(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wavs = [wav[: input_lengths[i]] for i, wav in enumerate(input)]
        self.upstream.eval()
        with torch.no_grad():
            feats = self.upstream(wavs)
        feats = self.featurizer(wavs, feats)
        if self.args.tile_factor != 1:
            feats = self._tile_representations(feats)
        input_feats = pad_list(feats, 0.0)
        feats_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

        # Saving CUDA Memory
        del feats
        return input_feats, feats_lens

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            input_feats_default, feats_lens_default = self.forward_default(
                input, input_lengths
            )
            input_feats_s3prl, feats_lens_s3prl = self.forward_s3prl(
                input, input_lengths
            )
        if self.align_method == "elevator" or self.align_method == "encoder_linear_fusion":  # a modif celle la avec les gcd
            return (
                input_feats_default,
                feats_lens_default,
                input_feats_s3prl,
                feats_lens_s3prl,
            )

        if self.align_method == "frontend_moe":  # a modif celle la avec les gcd
            assert (
                    input_feats_default.shape[-1] == 80
            ), "{}   ,    {}".format(
                input_feats_default.shape[-1], input_feats_s3prl.shape[-1]
            )

            input_feats_default_proj = self.projection_layer_fbanks(input_feats_default)
            input_feats_s3prl_proj = self.projection_layer_s3prl(input_feats_s3prl)

            # 2nd step : reshape
            bs, nf, dim = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj,
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, nf, dim = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj,
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            if self.apply_frontend_moe_on == "hubert":
                moe_weights = self.moe_layer(input_feats_s3prl_final)
            else:
                moe_weights = self.moe_layer(input_feats_default_final)
            a, b, c = input_feats_s3prl_final.shape
            assert 9==0, moe_weights.shape
            w1, w2 = moe_weights[:, :, 0].expand(c, a, b), moe_weights[:, :, 1].expand(c, a, b)
            w1, w2 = w1.permute(1, 2, 0), w2.permute(1, 2, 0)

            input_feats = w1 * input_feats_s3prl_final + w2 * input_feats_default_final
            feats_lens = torch.ones_like(feats_lens_default) * (m)

        elif self.align_method == "repeat":  # a modif celle la avec les gcd
            a, b = input_feats_default, input_feats_s3prl
            aa = a.repeat((1, 2, 1))
            bb = b.repeat((1, 5, 1))

            m = min(aa.shape[1], bb.shape[1])
            if m % 2 == 1:
                m -= 1
            A = aa[:, : m // 2, :]
            B = bb[:, : m // 2, :]

            C = torch.cat((A, B), dim=-1)

            input_feats = C

            feats_lens = torch.ones_like(feats_lens_default) * (m // 2)

        elif self.align_method == "linear_projection":

            # first step : projections
            # assert 9 == 5 , "{}    , {}   ,   {}   ,    {}".format(input_feats_default.shape, self.n_mels, input_feats_s3prl.shape, self.output_dim_s3prl )
            assert (
                input_feats_default.shape[-1] == 80
            ), "{}   ,    {}".format(
                input_feats_default.shape[-1], input_feats_s3prl.shape[-1]
            )

            input_feats_default_proj = self.projection_layer_fbanks(input_feats_default)
            input_feats_s3prl_proj = self.projection_layer_s3prl(input_feats_s3prl)

            # 2nd step : reshape
            bs, nf, dim = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj,
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, nf, dim = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj,
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            input_feats = torch.cat(
                (input_feats_default_final, input_feats_s3prl_final), dim=-1
            )
            feats_lens = torch.ones_like(feats_lens_default) * (m)

        elif self.align_method == "linear_projection_plus":

            # first step : projections
            # assert 9 == 5 , "{}    , {}   ,   {}   ,    {}".format(input_feats_default.shape, self.n_mels, input_feats_s3prl.shape, self.output_dim_s3prl )
            assert (
                input_feats_default.shape[-1] == 80
            ), "{}   ,    {}".format(
                input_feats_default.shape[-1], input_feats_s3prl.shape[-1]
            )

            input_feats_default_proj = self.projection_layer_fbanks(input_feats_default)
            input_feats_default_proj = torch.nn.functional.relu(
                input_feats_default_proj
            )
            input_feats_s3prl_proj = self.projection_layer_s3prl(input_feats_s3prl)
            input_feats_s3prl_proj = torch.nn.functional.relu(input_feats_s3prl_proj)

            # 2nd step : reshape
            bs, nf, dim = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj,
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, nf, dim = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj,
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            input_feats = torch.cat(
                (input_feats_default_final, input_feats_s3prl_final), dim=-1
            )

            feats_lens = torch.ones_like(feats_lens_default) * (m)

        elif self.align_method == "coattention":

            # first step : projections
            input_feats_default_proj = self.projection_layer_fbanks(input_feats_default)
            input_feats_s3prl_proj = self.projection_layer_s3prl(input_feats_s3prl)

            # 2nd step : reshape
            bs, nf, dim = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj,
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, nf, dim = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj,
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            # step 4 : coattention block
            # step 4.1 : frontend 1 is query, frontend 2 is key,value
            query = input_feats_s3prl_final
            key = input_feats_default_final
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
            attn = torch.nn.functional.softmax(score, -1)
            input_feats_tmp = torch.bmm(attn, key)
            input_feats_s3prl_final_coatt = input_feats_tmp

            # step 4.2 : frontend 1 is query, frontend 2 is key,value
            query = input_feats_default_final
            key = input_feats_s3prl_final
            score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
            attn = torch.nn.functional.softmax(score, -1)
            input_feats_tmp = torch.bmm(attn, key)
            input_feats_default_final_coatt = input_feats_tmp

            input_feats = torch.cat(
                (input_feats_default_final_coatt, input_feats_s3prl_final_coatt), dim=-1
            )
            feats_lens = torch.ones_like(feats_lens_default) * (m)

        elif self.align_method == "conv":
            input_feats_default_proj = self.conv_align_default(
                input_feats_default.permute(0, 2, 1)
            )
            input_feats_s3prl_proj = self.conv_align_s3prl(
                input_feats_s3prl.permute(0, 2, 1)
            )

            # 2nd step : reshape
            bs, dim, nf = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj.permute(0, 2, 1),
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, dim, nf = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj.permute(0, 2, 1),
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            input_feats = torch.cat(
                (input_feats_default_final, input_feats_s3prl_final), dim=-1
            )
            feats_lens = torch.ones_like(feats_lens_default) * (m)

        elif self.align_method == "rnn":
            input_feats_default_proj, _ = self.lstm_align_default(input_feats_default)
            input_feats_s3prl_proj, _ = self.lstm_align_s3prl(input_feats_s3prl)

            # 2nd step : reshape
            bs, nf, dim = input_feats_default_proj.shape
            input_feats_default_reshaped = torch.reshape(
                input_feats_default_proj.permute(0, 2, 1),
                (bs, nf * self.factor_for_fbanks, dim // self.factor_for_fbanks),
            )
            bs, nf, dim = input_feats_s3prl_proj.shape
            input_feats_s3prl_reshaped = torch.reshape(
                input_feats_s3prl_proj.permute(0, 2, 1),
                (bs, nf * self.factor_for_s3prl, dim // self.factor_for_s3prl),
            )

            # 3rd step : drop the few last frames

            m = min(
                input_feats_s3prl_reshaped.shape[1],
                input_feats_default_reshaped.shape[1],
            )
            input_feats_s3prl_final = input_feats_s3prl_reshaped[:, :m, :]
            input_feats_default_final = input_feats_default_reshaped[:, :m, :]

            input_feats = torch.cat(
                (input_feats_default_final, input_feats_s3prl_final), dim=-1
            )
            feats_lens = torch.ones_like(feats_lens_default) * (m)

        else:
            raise NotImplementedError

        return input_feats, feats_lens
