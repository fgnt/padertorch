
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
    """Moves list inside of dict/dataclass recursively.

    Can be used as map after batching of an dataset:
        `dataset.batch(...).map(collate_fn)`

    Args:
        batch: list of examples

    Returns:

    >>> batch = [{'a': 1}, {'a': 2}]
    >>> collate_fn(batch)
    {'a': [1, 2]}
    >>> collate_fn(tuple(batch))
    {'a': (1, 2)}

    >>> batch = [{'a': {'b': [1, 2]}}, {'a': {'b': [3, 4]}}]
    >>> collate_fn(batch)
    {'a': {'b': [[1, 2], [3, 4]]}}

    >>> import dataclasses
    >>> Point = dataclasses.make_dataclass('Point', ['x', 'y'])
    >>> batch = [Point(1, 2), Point(3, 4)]
    >>> batch
    [Point(x=1, y=2), Point(x=3, y=4)]
    >>> collate_fn(batch)
    Point(x=[1, 3], y=[2, 4])
    >>> collate_fn(tuple(batch))
    Point(x=(1, 3), y=(2, 4))
    """
    assert isinstance(batch, (tuple, list)), (type(batch), batch)

    if isinstance(batch[0], dict):
        for b in batch[1:]:
            assert batch[0].keys() == b.keys(), batch
        return batch[0].__class__({
            k: (collate_fn(batch.__class__([b[k] for b in batch])))
            for k in batch[0]
        })
    elif hasattr(batch[0], '__dataclass_fields__'):
        for b in batch[1:]:
            assert batch[0].__dataclass_fields__ == b.__dataclass_fields__, batch
        return batch[0].__class__(**{
            k: (collate_fn(batch.__class__([getattr(b, k) for b in batch])))
            for k in batch[0].__dataclass_fields__
        })
    else:
        return batch
