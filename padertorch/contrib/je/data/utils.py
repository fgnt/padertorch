import numpy as np


def split_dataset(dataset, fold, nfolfds=5, seed=0):
    """

    Args:
        dataset:
        fold:
        nfolfds:
        seed:

    Returns:

    >>> split_dataset(np.array([1,2,3,4,5]), 0, nfolfds=2)
    [array([2, 4, 5]), array([1, 3])]
    >>> split_dataset(np.array([1,2,3,4,5]), 1, nfolfds=2)
    [array([1, 3]), array([2, 4, 5])]
    """
    indices = np.arange(len(dataset))
    np.random.RandomState(seed).shuffle(indices)
    folds = np.split(
        indices,
        np.linspace(0, len(dataset), nfolfds + 1)[1:-1].astype(np.int64)
    )
    validation_indices = folds.pop(fold)
    training_indices = np.concatenate(folds)
    return [
        dataset[sorted(indices.tolist())]
        for indices in (training_indices, validation_indices)
    ]
