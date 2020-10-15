import collections
from typing import List

from torch import nn

from padertorch.ops.mappings import ACTIVATION_FN_MAP


def fully_connected_stack(
        input_size: int,
        hidden_size: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout: float = 0.5,
        output_activation: str = None
):
    """

        dropout describes the forget-probability.
        More information to dropout: https://arxiv.org/pdf/1207.0580.pdf

        Args:
            input_size: has to be defined
            hidden_size: size of the hidden layers
                either None, int, list or tuple
            output_size: has to be defined
            activation: used in all layers except the last
            dropout: Dropout forget ratio (opposite to TensorFlow)
                default take from:
                    https://www.reddit.com/r/MachineLearning/comments/3oztvk/why_50_when_using_dropout/
            output_activation: applied after the last layer

        >>> fully_connected_stack(513, [1024, 1024], 1024)
        Sequential(
          (dropout_0): Dropout(p=0.5, inplace=False)
          (linear_0): Linear(in_features=513, out_features=1024, bias=True)
          (relu_0): ReLU()
          (dropout_1): Dropout(p=0.5, inplace=False)
          (linear_1): Linear(in_features=1024, out_features=1024, bias=True)
          (relu_1): ReLU()
          (dropout_2): Dropout(p=0.5, inplace=False)
          (linear_2): Linear(in_features=1024, out_features=1024, bias=True)
        )
        """
    assert input_size is not None, input_size
    assert output_size is not None, output_size

    layers = collections.OrderedDict()
    if hidden_size is None:
        l_n_units = [input_size, output_size]
    elif isinstance(hidden_size, (list, tuple)):
        l_n_units = [input_size] + list(hidden_size) + [output_size]
    elif isinstance(hidden_size, int):
        l_n_units = [input_size, hidden_size, output_size]
    else:
        raise TypeError(hidden_size)

    activation = [activation] * (len(l_n_units) - 2) + [output_activation]

    for l_idx, n_units in enumerate(l_n_units[:-1]):
        layers[f'dropout_{l_idx}'] = nn.Dropout(dropout)
        layers[f'linear_{l_idx}'] = nn.Linear(n_units, l_n_units[l_idx + 1])
        if activation[l_idx] is not None and activation[l_idx] != 'identity':
            layers[f'{activation[l_idx]}_{l_idx}'] = \
                ACTIVATION_FN_MAP[activation[l_idx]]()
    return nn.Sequential(layers)
