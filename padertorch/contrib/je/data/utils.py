import numpy as np
from lazy_dataset.core import DynamicTimeSeriesBucket


class DynamicDisjointExamplesTimeSeriesBucket(DynamicTimeSeriesBucket):
    def __init__(self, init_example, **kwargs):
        """
        Extension of the DynamicTimeSeriesBucket such that examples are
        balanced with respect to the dataset they originate from

        Args:
            init_example: first example in the bucket
            dataset_sizes:
            **kwargs: kwargs of DynamicTimeSeriesBucket
        """
        super().__init__(init_example, **kwargs)
        self.example_ids = set()

    def assess(self, example):
        return (
            example["example_id"] not in self.example_ids and super().assess(example)
        )

    def _append(self, example):
        super()._append(example)
        self.example_ids.add(example["example_id"])


def split_dataset(dataset, fold, nfolds=5, seed=0):
    """

    Args:
        dataset:
        fold:
        nfolfds:
        seed:

    Returns:

    >>> split_dataset(np.array([1,2,3,4,5]), 0, nfolds=2)
    [array([2, 4, 5]), array([1, 3])]
    >>> split_dataset(np.array([1,2,3,4,5]), 1, nfolds=2)
    [array([1, 3]), array([2, 4, 5])]
    """
    indices = np.arange(len(dataset))
    if seed is not None:
        np.random.RandomState(seed).shuffle(indices)
    folds = np.split(
        indices,
        np.linspace(0, len(dataset), nfolds + 1)[1:-1].astype(np.int64)
    )
    validation_indices = folds.pop(fold)
    training_indices = np.concatenate(folds)
    return [
        dataset[sorted(indices.tolist())]
        for indices in (training_indices, validation_indices)
    ]
