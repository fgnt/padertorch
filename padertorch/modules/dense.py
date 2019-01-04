import collections
from typing import List

from torch import nn
import torch.nn.functional as F

import padertorch as pt
from padertorch.ops.mappings import ACTIVATION_FN_MAP, ACTIVATION_NN_MAP


# Is a class or a function better? DenseStack vs dense_stack() -> nn.Sequential
class DenseStack(pt.Module, nn.Sequential):
    def __init__(
            self,
            input_size: int = 513,
            num_units: List[int] = 3 * [1024],
            activation_fn: str = 'relu',
            dropout: int = 0.5
    ):
        """

        dropout describes the forget-probability.
        More information to dropout: https://arxiv.org/pdf/1207.0580.pdf
        TODO: Please discuss, if "dense" is the correct name in Torch.

        CB: Why distinguish input_size and num_units? From the signature I
            would expect num_units[0] == input_size
        CB: Why activation_fn? activation would be better, especially since it
            is not a function.
        CB: Is dropout == 0.5 a proper default? Isn't 0 a better value?
        CB: num_units shouldn't have a default. Add a doc text explain what is
            expected.

        Args:
            input_size:
            num_units:
            activation_fn:
            dropout: Dropout forget ratio (opposite to TensorFlow)

        >>> DenseStack()
        DenseStack(
          (dropout_0): Dropout(p=0.5)
          (linear_0): Linear(in_features=513, out_features=1024, bias=True)
          (relu_0): ReLU()
          (dropout_1): Dropout(p=0.5)
          (linear_1): Linear(in_features=1024, out_features=1024, bias=True)
          (relu_1): ReLU()
          (dropout_2): Dropout(p=0.5)
          (linear_2): Linear(in_features=1024, out_features=1024, bias=True)
          (relu_2): ReLU()
        )
        """
        super().__init__()
        l_n_units = [input_size] + num_units

        layers = collections.OrderedDict()

        for l_idx, n_units in enumerate(num_units):
            layers[f'dropout_{l_idx}'] = nn.Dropout(dropout)
            layers[f'linear_{l_idx}'] = nn.Linear(l_n_units[l_idx], n_units)
            layers[f'{activation_fn}_{l_idx}'] = \
                ACTIVATION_NN_MAP[activation_fn]()

        super().__init__(layers)
