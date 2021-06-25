from collections import namedtuple
from typing import Tuple
import numpy as np
from torch import nn


def num_parameters(module: nn.Module) -> Tuple[int, int]:
    """Counts the number of parameters for `module`.

    Args:
        module: The module to count the number of parameters for

    Returns: The total number of parameters and the number of trainable
        parameters.

    Examples:
        >>> num_parameters(nn.Linear(10, 10))
        model_parameter_size(total=110, trainable=110)
        >>> net = nn.Sequential(nn.Linear(10, 10).requires_grad_(False), nn.Linear(10, 10))
        >>> num_parameters(net)
        model_parameter_size(total=220, trainable=110)
    """
    total_number = 0
    trainable_number = 0
    for parameter in module.parameters():
        size = np.prod(parameter.shape)
        if parameter.requires_grad:
            trainable_number += size
        total_number += size

    return namedtuple(
        'model_parameter_size', ('total', 'trainable')
    )(total_number, trainable_number)


def human_num_parameters(module: nn.Module) -> str:
    import humanize
    total, trainable = num_parameters(module)
    if total == trainable:
        return humanize.intword(total)
    else:
        return (
            f'{humanize.intword(total)} '
            f'(trainable: {humanize.intword(trainable)})'
        )
