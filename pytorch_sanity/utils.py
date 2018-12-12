import numpy as np
import torch

from pytorch_sanity.parameterized import Parameterized

def to_list(x, length=None):
    if not isinstance(x, list):
        x = [x] * (1 if length is None else length)
    if length is not None:
        assert len(x) == length
    return x


def nested_update(orig, update):
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


class Padder(Parameterized):
    def __init__(
            self,
            to_torch: bool = True,
            sort_by_length: bool = True,
            padding: bool = True,
            padding_keys: list = None
    ):
        assert not to_torch ^ (padding and to_torch)
        self.to_torch = to_torch
        self.sort_by_length = sort_by_length
        self.padding = padding
        self.padding_keys = padding_keys

    def pad_batch(self, batch):
        if isinstance(batch[0], np.ndarray):
            if len(batch[0].shape) > 0:

                dims = np.array(
                    [[idx for idx in array.shape] for array in batch]).T
                axis = [idx for idx, dim in enumerate(dims) if
                        not all(dim == dim[::-1])]

                assert len(axis) in [0,
                                     1], f'only one axis is allowed to differ,' \
                                         f' axis={axis} and dims={dims}'
                if len(axis) == 1:
                    axis = axis[0]
                    if self.sort_by_length:
                        batch = sorted(batch, key=lambda x: x.shape[axis],
                                       reverse=True)
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
                if self.sort_by_length:
                    return np.array(sorted(batch, reverse=True))
                else:
                    return np.array(batch)
        elif isinstance(batch[0], int):
            # sort num_samples / num_frames to fit the sorted batches
            if self.sort_by_length:
                return np.array(sorted(batch, reverse=True))
            else:
                return np.array(batch)
        else:
            return batch

    def __call__(self, batch):
        # assumes batch to be a list of dicts
        # ToDo: do we automaticaly sort by sequence length?

        def nested_batching(value, key, nested_batch):
            # recursively nesting the batch
            if isinstance(value, dict):
                if not key in nested_batch:
                    nested_batch[key] = dict()
                return {k: nested_batching(v, k, nested_batch[key])
                        for k, v in value.items()}
            else:
                if not key in nested_batch:
                    nested_batch[key] = []
                nested_batch[key].append(value)
            return nested_batch[key]

        nested_batch = {}
        for elem in batch:
            assert isinstance(elem, dict)
            nested_batch = {key: nested_batching(value, key, nested_batch)
                            for key, value in elem.items()}

        if self.padding:
            if self.padding_keys is None:
                padding_keys = nested_batch.keys()
            else:
                assert len(self.padding_keys) > 0, 'Empty padding key list was ' \
                                                   'provided default should be None'

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
                'Padding keys have to be None or empty if padding is set to False, '
                'but they are:', self.padding_keys
            )
            return nested_batch


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