import collections

import numpy as np
import torch


def normalize_axis(x, axis):
    """Here, `axis` is always understood to reference the unpacked axes.

    TODO: Do we need to assert, that the axes do not collide?

    >>> normalize_axis(torch.zeros(2, 3, 4), -1)
    (2,)

    >>> normalize_axis(torch.zeros(2, 3, 4), (-3, -2, -1))
    (0, 1, 2)
    """
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        ndim = len(x.data.size()) + 1
    else:
        ndim = len(x.size())
    return tuple(a % ndim for a in axis)


def to_list(x, length=None):
    """
    Often a list is required, but for convenience it is desired to enter a
    single object, e.g. string.

    Complicated corner cases are e.g. `range()` and `dict.values()`, which are
    handled here.

    >>> from paderbox.utils.pretty import pprint

    >>> to_list(1)
    [1]
    >>> to_list([1])
    [1]
    >>> to_list((i for i in range(3)))
    [0, 1, 2]
    >>> pprint(to_list(np.arange(3)), nep51=True)  # use pprint to support numpy 1 and 2
    [np.int64(0), np.int64(1), np.int64(2)]
    >>> to_list({'a': 1})
    [{'a': 1}]
    >>> to_list({'a': 1}.keys())
    ['a']
    >>> to_list('ab')
    ['ab']
    >>> from pathlib import Path
    >>> to_list(Path('/foo/bar')) # doctest: +ELLIPSIS
    [...Path('/foo/bar')]
    """
    # Important cases (change type):
    #  - generator -> list
    #  - dict_keys -> list
    #  - dict_values -> list
    #  - np.array -> list (discussable)
    # Important cases (list of original object):
    #  - dict -> list of dict

    def to_list_helper(x_):
        return [x_] * (1 if length is None else length)

    if isinstance(x, collections.abc.Mapping):
        x = to_list_helper(x)
    elif isinstance(x, str):
        x = to_list_helper(x)
    elif isinstance(x, collections.abc.Sequence):
        pass
    elif isinstance(x, collections.abc.Iterable):
        x = list(x)
    else:
        x = to_list_helper(x)

    if length is not None:
        assert len(x) == length, (len(x), length)
    return x


def to_numpy(array, detach: bool = False, copy: bool = False) -> np.ndarray:
    """
    Transforms `array` to a numpy array. `array` can be anything that
    `np.asarray` can handle and torch tensors.

    Args:
        array: The array to transform to numpy
        detach: If `True`, `array` gets detached if it is a `torch.Tensor`.
            This has to be enabled explicitly to prevent unintentional
            truncation of a backward graph.
        copy: If `True`, the array gets copied. Otherwise, it becomes read-only
            to prevent unintened changes on the input array or tensor by
            altering the output.

    Returns:
        `array` as a numpy array.

    >>> t = torch.zeros(2)
    >>> t
    tensor([0., 0.])
    >>> to_numpy(t), np.zeros(2, dtype=np.float32)
    (array([0., 0.], dtype=float32), array([0., 0.], dtype=float32))

    >>> t = torch.zeros(2, requires_grad=True)
    >>> t
    tensor([0., 0.], requires_grad=True)
    >>> to_numpy(t, detach=True), np.zeros(2, dtype=np.float32)
    (array([0., 0.], dtype=float32), array([0., 0.], dtype=float32))


    >>> from torch_complex.tensor import ComplexTensor
    >>> to_numpy(ComplexTensor(t), detach=True)
    array([0.+0.j, 0.+0.j], dtype=complex64)

    >>> from torch_complex.tensor import ComplexTensor
    >>> to_numpy(torch.tensor([1+1j]), detach=True)
    array([1.+1.j], dtype=complex64)
    >>> to_numpy(torch.tensor([1+1j]).conj(), detach=True)
    array([1.-1.j], dtype=complex64)
    """
    # if isinstance(array, torch.Tensor):

    try:
        # Torch 1.10 introduced `resolve_conj`, which can cause the following
        # exception:
        #
        #     RuntimeError: Can't call numpy() on Tensor that has conjugate bit
        #     set. Use tensor.resolve_conj().numpy() instead.
        #
        #     It is likely, that you are evaluating a model in train mode.
        #     You may want to call `model.eval()` first and use a context
        #     manager, which disables gradients: `with torch.no_grad(): ...`.
        #     If you want to detach anyway, use `detach=True` as argument.
        array = array.resolve_conj()
    except AttributeError:
        pass

    try:
        array = array.cpu()
    except AttributeError:
        pass
    else:
        if detach:
            array = array.detach()

    try:
        # torch only supports np.asarray for cpu tensors
        if copy:
            return np.array(array)
        else:
            array = np.asarray(array)
            array.setflags(write=False)
            return array
    except TypeError as e:
        raise TypeError(type(array), array) from e
    except RuntimeError as e:
        import sys
        raise type(e)(str(e) + (
            '\n\n'
            'It is likely, that you are evaluating a model in train mode.\n'
            'You may want to call `model.eval()` first and use a context\n'
            'manager, which disables gradients: `with torch.no_grad(): ...`.\n'
            'If you want to detach anyway, use `detach=True` as argument.'
            )
        ) from e
