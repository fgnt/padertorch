import numpy as np

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


def pad_batch(batch):

    if isinstance(batch[0], np.ndarray):
        if len(batch[0].shape) > 0:
            dims = np.array([[idx for idx in array.shape] for array in batch]).T
            axis = [idx for idx, dim in enumerate(dims) if
                    not all(dim == dim[::-1])]
            assert len(axis) == 1, f'only one axis is allowed to differ,' \
                                   f' axis={axis} and dims={dims}'
            axis = axis[0]
            sorted_batch = sorted(batch, key=lambda x: x.shape[axis],
                                  reverse=True)
            pad = sorted_batch[0].shape[axis]
            return np.stack([pad_tensor(vec, pad, axis)
                         for vec in sorted_batch], axis=0)
        else:
            return np.array(sorted(batch, reverse=True))
    else:
        return batch



def collate_fn(batch):
    # only works with shallow dict structure
    merged_batch = {}
    for elem in batch:
        if isinstance(elem, dict):
            for key, value in elem.items():
                if key not in merged_batch:
                    merged_batch[key] = []
                merged_batch[key].append(value)
    return {key: pad_batch(vec) for key, vec in merged_batch.items()}