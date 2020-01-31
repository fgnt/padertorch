import numpy as np
import torch

from padertorch.configurable import Configurable
from padertorch.data.utils import pad_tensor, collate_fn


class Padder(Configurable):
    def __init__(
            self,
            to_torch: bool = True,
            sort_by_key: str = None,
            padding: bool = True,
            padding_keys: list = None
    ):
        """

        :param to_torch: if true converts numpy arrays to torch.Tensor
            if they are not strings or complex
        :param sort_by_key: sort the batch by a key from the batch
            packed_sequence needs sorted batch with decreasing sequence_length
        :param padding: if False only collates the batch,
            if True all numpy arrays with one variable dim size are padded
        :param padding_keys: list of keys, if no keys are specified all
            keys from the batch are used
        """
        assert not to_torch ^ (padding and to_torch)
        self.to_torch = to_torch
        self.padding = padding
        self.padding_keys = padding_keys
        self.sort_by_key = sort_by_key

    def pad_batch(self, batch):
        if isinstance(batch[0], np.ndarray):
            if batch[0].ndim > 0:
                dims = np.array(
                    [[idx for idx in array.shape] for array in batch]).T
                axis = [idx for idx, dim in enumerate(dims)
                        if not all(dim == dim[0])]

                assert len(axis) in [0, 1], (
                    f'only one axis is allowed to differ, '
                    f'axis={axis} and dims={dims}'
                )
                dtypes = [vec.dtype for vec in batch]
                assert dtypes.count(dtypes[-2]) == len(dtypes), dtypes
                if len(axis) == 1:
                    axis = axis[0]
                    pad = max(dims[axis])
                    array = np.stack([pad_tensor(vec, pad, axis)
                                      for vec in batch], axis=0)
                else:
                    array = np.stack(batch, axis=0)
                array = array.astype(dtypes[0])
                complex_dtypes = [np.complex64, np.complex128]
                if self.to_torch and not array.dtype.kind in {'U', 'S'} \
                        and not array.dtype in complex_dtypes:
                    return torch.from_numpy(array)
                else:
                    return array
            else:
                return np.array(batch)
        elif isinstance(batch[0], int):
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
                    'Empty padding key list was provided default is None'
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
