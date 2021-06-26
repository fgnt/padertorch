from dataclasses import dataclass
import torch
from torch import nn


@dataclass(repr=False)
class ModelParameterSize:
    total_count: int = 0
    trainable_count: int = 0
    total_bytes: int = 0
    trainable_bytes: int = 0

    def __repr__(self):
        try:
            import humanize
            return (
                f'{self.__class__.__name__}('
                f'total_count={humanize.intword(self.total_count)}, '
                f'trainable_count={humanize.intword(self.trainable_count)}, '
                f'total_bytes={humanize.naturalsize(self.total_bytes)}, '
                f'trainable_bytes={humanize.naturalsize(self.trainable_bytes)})'
            )
        except ImportError:
            return (
                f'{self.__class__.__name__}('
                f'total_count={self.total_count}, '
                f'trainable_count={self.trainable_count}, '
                f'total_bytes={self.total_bytes}, '
                f'trainable_bytes={self.trainable_bytes})'
            )


def num_parameters(module: nn.Module) -> ModelParameterSize:
    """Counts the number of parameters for `module`.

    Args:
        module: The module to count the number of parameters for

    Returns: The total number of parameters and the number of trainable
        parameters.

    Examples:
        >>> num_parameters(nn.Linear(10, 10))
        ModelParameterSize(total_count=110, trainable_count=110, total_bytes=440 Bytes, trainable_bytes=440 Bytes)
        >>> net = nn.Sequential(nn.Linear(10, 10).requires_grad_(False), nn.Linear(10, 10))
        >>> num_parameters(net)
        ModelParameterSize(total_count=220, trainable_count=110, total_bytes=880 Bytes, trainable_bytes=440 Bytes)
    """
    result = ModelParameterSize()

    for parameter in module.parameters():
        size = parameter.numel()
        bytes = parameter.element_size()

        if parameter.requires_grad:
            result.trainable_count += size
            result.trainable_bytes += size * bytes
        result.total_count += size
        result.total_bytes += size * bytes

    return result
