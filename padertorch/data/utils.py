
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

    Can be used as map after batching of a dataset:
        `dataset.batch(...).map(collate_fn)`

    Args:
        batch: list or tuple of examples

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
    >>> Data = dataclasses.make_dataclass('Data', ['x', 'y'])
    >>> batch = [Data(1, 2), Data(3, 4)]
    >>> batch
    [Data(x=1, y=2), Data(x=3, y=4)]
    >>> collate_fn(batch)
    Data(x=[1, 3], y=[2, 4])
    >>> collate_fn(tuple(batch))
    Data(x=(1, 3), y=(2, 4))

    >>> from paderbox.array.sparse import zeros
    >>> batch = [zeros(10), zeros(20)]
    >>> collate_fn(batch)
    [SparseArray(shape=(10,)), SparseArray(shape=(20,))]
    >>> batch = [Data(zeros(1), zeros(1)), Data(zeros(1), zeros(1))]
    >>> collate_fn(batch)
    Data(x=[SparseArray(shape=(1,)), SparseArray(shape=(1,))], y=[SparseArray(shape=(1,)), SparseArray(shape=(1,))])
    """
    assert isinstance(batch, (tuple, list)), (type(batch), batch)

    e = batch[0]

    if isinstance(e, dict):
        for b in batch[1:]:
            assert b.keys() == e.keys(), batch
        return e.__class__({
            k: collate_fn(batch.__class__([b[k] for b in batch]))
            for k in e
        })
    elif (
            hasattr(e, '__dataclass_fields__')
            # Specifically ignore SparseArray, which is a dataclass but should be treated as an array here
            and f'{e.__class__.__module__}.{e.__class__.__qualname__}' != 'paderbox.array.sparse.SparseArray'
    ):
        for b in batch[1:]:
            assert b.__dataclass_fields__ == e.__dataclass_fields__, batch
        return e.__class__(**{
            k: collate_fn(batch.__class__([getattr(b, k) for b in batch]))
            for k in e.__dataclass_fields__
        })
    else:
        return batch
