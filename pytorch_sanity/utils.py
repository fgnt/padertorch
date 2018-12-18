import numpy as np
import torch

from pytorch_sanity.configurable import Configurable


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
    if not isinstance(x, list):
        x = [x] * (1 if length is None else length)
    if length is not None:
        assert len(x) == length
    return x


def nested_update(orig, update):
    # Todo:
    assert isinstance(update, type(orig))
    if isinstance(orig, list):
        for i, value in enumerate(update):
            if isinstance(value, (dict, list)) \
                    and i < len(orig) and isinstance(orig[i], type(value)):
                nested_update(orig[i], value)
            elif i < len(orig):
                orig[i] = value
            else:
                assert i == len(orig)
                orig.append(value)
    elif isinstance(orig, dict):
        for key, value in update.items():
            if isinstance(value, (dict, list)) \
                    and key in orig and isinstance(orig[key], type(value)):
                nested_update(orig[key], value)
            else:
                orig[key] = value


def nested_op(func, arg1, *args, broadcast=False):
    """
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=[1], b=dict(c=4)), dict(a=[0], b=dict(c=1)))
    {'a': [1], 'b': {'c': 7}}
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3]))
    Traceback (most recent call last):
    ...
    AssertionError: ([1, 3],)
    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3]), broadcast=True)
    {'a': 1, 'b': {'c': [4, 10]}}

    :param func:
    :param arg1:
    :param args:
    :param broadcast:
    :return:
    """
    if isinstance(arg1, dict):
        if not broadcast:
            assert all(
                [isinstance(arg, dict) and arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        else:
            assert all(
                [not isinstance(arg, dict) or arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        keys = arg1.keys()
        return arg1.__class__({
            key: nested_op(
                func,
                arg1[key],
                *[arg[key] if isinstance(arg, dict) else arg
                  for arg in args],
                broadcast=broadcast
            )
            for key in keys
        })
    if isinstance(arg1, (list, tuple)):
        if not broadcast:
            assert all([
                isinstance(arg, (list, tuple)) and len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not isinstance(arg, (list, tuple)) or len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        return arg1.__class__([
            nested_op(
                func,
                arg1[j],
                *[arg[j] if isinstance(arg, list) else arg
                  for arg in args],
                broadcast=broadcast
            )
            for j in range(len(arg1))]
        )
    return func(arg1, *args)


def squeeze_nested(orig):
    if isinstance(orig, (dict, list)):
        keys = list(orig.keys() if isinstance(orig, dict) else range(len(orig)))
        squeezed = True
        for key in keys:
            orig[key] = squeeze_nested(orig[key])
            if isinstance(orig[key], (list, dict)):
                squeezed = False
        if squeezed and all([orig[key] == orig[keys[0]] for key in keys]):
            return orig[keys[0]]
    return orig


def pad_tensor(vec, pad, axis):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        axis - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    pad_size = list(vec.shape)
    pad_size[axis] = pad - vec.shape[axis]
    return np.concatenate([vec, np.zeros(pad_size)], axis=axis)


def collate_fn(batch):
    """Moves list inside of dict recursively.

    Can be used as input to batch iterator.

    Args:
        batch:

    Returns:

    """
    def nested_batching(value, key, nested_batch):
        # recursively nesting the batch
        if isinstance(value, dict):
            if key not in nested_batch:
                nested_batch[key] = dict()
            return {k: nested_batching(v, k, nested_batch[key])
                    for k, v in value.items()}
        else:
            if key not in nested_batch:
                nested_batch[key] = []
            nested_batch[key].append(value)
        return nested_batch[key]

    nested_batch = {}
    for elem in batch:
        assert isinstance(elem, dict)
        nested_batch = {key: nested_batching(value, key, nested_batch)
                        for key, value in elem.items()}
    return nested_batch


class Padder(Configurable):
    def __init__(
            self,
            to_torch: bool = True,
            sort_by_key: str = None,
            padding: bool = True,
            padding_keys: list = None
    ):
        assert not to_torch ^ (padding and to_torch)
        self.to_torch = to_torch
        self.padding = padding
        self.padding_keys = padding_keys
        self.sort_by_key = sort_by_key

    def pad_batch(self, batch):
        if isinstance(batch[0], np.ndarray):
            if len(batch[0].shape) > 0:

                dims = np.array(
                    [[idx for idx in array.shape] for array in batch]).T
                axis = [idx for idx, dim in enumerate(dims) if
                        not all(dim == dim[::-1])]

                assert len(axis) in [0, 1], (
                    f'only one axis is allowed to differ, '
                    f'axis={axis} and dims={dims}'
                )
                if len(axis) == 1:
                    axis = axis[0]
                    pad = max(dims[axis])
                    array = np.stack([pad_tensor(vec, pad, axis)
                                      for vec in batch], axis=0)
                else:
                    array = np.stack(batch, axis=0)
                if self.to_torch:
                    return torch.from_numpy(array)
                else:
                    return array
            else:
                # sort num_samples / num_frames to fit the sorted batches
                return np.array(batch)
        elif isinstance(batch[0], int):
            # sort num_samples / num_frames to fit the sorted batches
            return np.array(batch)
        else:
            return batch

    def sort(self, batch):
        return sorted(batch, key=lambda x: x[self.sort_by_key], reverse=True)

    def __call__(self, unsorted_batch):
        # assumes batch to be a list of dicts
        # ToDo: do we automatically sort by sequence length?

        if self.sort_by_key:
            batch = self.sort(unsorted_batch)
        else:
            batch = unsorted_batch

        nested_batch = collate_fn(batch)


        if self.padding:
            if self.padding_keys is None:
                padding_keys = nested_batch.keys()
            else:
                assert len(self.padding_keys) > 0, \
                    'Empty padding key list was provided default should be None'

            def nested_padding(value, key):
                if isinstance(value, dict):
                    return {k: nested_padding(v, k) for k, v in value.items()}
                else:
                    if key in padding_keys:
                        return self.pad_batch(value)
                    else:
                        return value

            return {key: nested_padding(value, key) for key, value in
                    nested_batch.items()}
        else:
            assert self.padding_keys is None or len(self.padding_keys) == 0, (
                'Padding keys have to be None or empty if padding is set to '
                'False, but they are:', self.padding_keys
            )
            return nested_batch
