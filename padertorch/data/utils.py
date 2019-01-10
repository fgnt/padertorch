
import numpy as np
from padertorch.configurable import Configurable
import torch


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
                padding_keys = self.padding_keys

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
