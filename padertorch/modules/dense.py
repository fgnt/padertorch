import padertorch as pt
from typing import List
from torch import nn
import torch.nn.functional as F
from padertorch.ops.mappings import ACTIVATION_FN_MAP

class DenseStack(pt.Module):
    def __init__(
            self,
            input_size: int = 513,
            num_units: List[int] = 3*[1024],
            activation_fn: str = 'relu',
            dropout: int = 0.5
    ):
        super().__init__()
        self.num_units = num_units
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.input_size = input_size

    def build(self):
        l_n_units = [self.opts.input_size] + self.opts.num_units
        for l_idx, n_units in enumerate(self.opts.num_units):
            self.__setattr__(f'linear_{l_idx}',
                     nn.Linear(l_n_units[l_idx], n_units))

    def forward(self, x):
        for l_idx in range(len(self.opts.num_units)):
            x = ACTIVATION_FN_MAP[self.opts.activation_fn](
                self.__getattr__(f'linear_{l_idx}')(
                    F.dropout(x, self.opts.dropout, self.training))
            )
        return x