import operator
from dataclasses import dataclass

import numpy as np
import torch
from typing import Union, Iterable
import paderbox as pb

__all__ = [
    'example_to_device',
    'example_to_numpy',
    'Sorter',
]


def example_to_device(example, device=None, memo=None):
    """
    Moves a nested structure to the device.
    Numpy arrays are converted to `torch.Tensor`. Complex numpy arrays are
    converted if supported by the used torch version.

    >>> import torch, numpy as np
    >>> example_to_device(np.ones(5, dtype=np.float32))
    tensor([1., 1., 1., 1., 1.])
    >>> example_to_device({'signal': np.ones(5, dtype=np.float32)})
    {'signal': tensor([1., 1., 1., 1., 1.])}
    >>> example_to_device({'signal': [np.ones(5, dtype=np.float32)], 'a': 'b'})
    {'signal': [tensor([1., 1., 1., 1., 1.])], 'a': 'b'}
    >>> example_to_device({'signal': (np.ones(5, dtype=np.float32),)})
    {'signal': (tensor([1., 1., 1., 1., 1.]),)}
    >>> example_to_device({'signal': (torch.ones(5),)})
    {'signal': (tensor([1., 1., 1., 1., 1.]),)}

    This function tracks already moved objects and like `copy.deepcopy` to
    avoid multiple moves of the same object:
    >>> a = np.ones(2)
    >>> ex = example_to_device({'a': a, 'b': a})
    >>> ex['a'] is ex['b']
    True

    The original doctext from torch for `.to`:
    Tensor.to(device=None, dtype=None, non_blocking=False, copy=False) → Tensor
        Returns a Tensor with the specified device and (optional) dtype. If
        dtype is None it is inferred to be self.dtype. When non_blocking, tries
        to convert asynchronously with respect to the host if possible, e.g.,
        converting a CPU Tensor with pinned memory to a CUDA Tensor. When copy
        is set, a new Tensor is created even when the Tensor already matches
        the desired conversion.

    Args:
        example:
        device: None, 'cpu', 0, 1, ...

    Returns:
        example on device

    """
    if memo is None:
        memo = {}

    def convert(value):
        id_ = id(value)
        if id_ in memo:
            return memo[id_]

        if isinstance(value, np.ndarray):
            try:
                value = torch.from_numpy(value)
            except TypeError:
                # Check if this is caused by an old pytorch version that can't
                # convert complex-valued arrays to tensors. In that case: don't
                # crash
                if value.dtype not in [np.complex64, np.complex128]:
                    raise
        if isinstance(value, torch.Tensor):
            value = value.to(device=device)
        memo[id_] = value
        return value

    return pb.utils.nested.nested_op(convert, example, handle_dataclass=True)


def example_to_numpy(example, detach: bool = False, memo: dict = None):
    """
    Moves a nested structure to numpy. Opposite of `example_to_device`.

    >>> import torch
    >>> example_to_numpy(torch.ones(5))
    array([1., 1., 1., 1., 1.], dtype=float32)
    >>> example_to_numpy({'signal': torch.ones(5)})
    {'signal': array([1., 1., 1., 1., 1.], dtype=float32)}
    >>> example_to_numpy({'signal': [torch.ones(5)]})
    {'signal': [array([1., 1., 1., 1., 1.], dtype=float32)]}
    >>> example_to_numpy({'signal': (torch.ones(5),)})
    {'signal': (array([1., 1., 1., 1., 1.], dtype=float32),)}

    This function tracks already moved objects and like `copy.deepcopy` to
    avoid multiple moves of the same object:
    >>> a = torch.ones(2)
    >>> ex = example_to_numpy({'a': a, 'b': a})
    >>> ex['a'] is ex['b']
    True

    Returns:
        example where each tensor is converted to numpy

    """
    from padertorch.utils import to_numpy

    if memo is None:
        memo = {}

    def convert(value):
        id_ = id(memo)
        if id_ in memo:
            return memo[id_]
        if isinstance(value, torch.Tensor) or 'ComplexTensor' in str(type(value)):
            value = to_numpy(value, detach=detach)
        memo[id_] = value
        return value

    return pb.utils.nested.nested_op(convert, example, handle_dataclass=True)


@dataclass
class Sorter:
    """
    Sorts the example in a batch by `key`. Meant to be mapped to a lazy
    dataset after batching and before collating like
    `dataset.batch(4).map(Sorter('num_samples')).map(collate_fn)`.

    Examples:
        >>> batch = [{'value': x} for x in [5, 1, 3, 2]]
        >>> Sorter('value')(batch)
        ({'value': 5}, {'value': 3}, {'value': 2}, {'value': 1})

    Attributes:
        key: Key to sort by
        reverse: If `True`, sorts in reverse order. The default `True` is
            required if sorting by length for `PackedSequence`s.
    """
    key: Union[str, callable] = 'num_samples'
    reverse: bool = True

    def __post_init__(self):
        if not callable(self.key):
            self.key = operator.itemgetter(self.key)

    def __call__(self, examples: Iterable) -> tuple:
        return tuple(sorted(examples, key=self.key, reverse=self.reverse))
