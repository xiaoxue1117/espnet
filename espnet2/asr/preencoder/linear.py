#!/usr/bin/env python3
#  2020, Technische Universität München;  Ludwig Kürzinger
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Sinc convolutions for raw audio input."""

from collections import OrderedDict
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
import humanfriendly
from typeguard import check_argument_types
from typing import Optional
from typing import Tuple
from typing import Union

import torch


class LinearProjector(AbsPreEncoder):
    """Linear Projector Preencoder.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_output_relu: bool = False,
    ):
        """Initialize the module.
        """
        assert check_argument_types()
        super().__init__()

        # TODO(xkc09): compare w/ and w/o non-linear activation
        self.output_dim = output_size
        if not use_output_relu:
            self.linear_out = torch.nn.Linear(input_size, output_size)
        else:
            self.linear_out = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size), torch.nn.ReLU()
            )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.
        """
        output = self.linear_out(input)
        return output, input_lengths  # no state in this layer
    
    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim