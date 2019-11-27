
import numpy as np


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
