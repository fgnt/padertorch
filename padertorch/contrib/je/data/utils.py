import numpy as np
from lazy_dataset.core import DynamicTimeSeriesBucket
from padertorch.utils import to_list


class DynamicExtendedTimeSeriesBucket(DynamicTimeSeriesBucket):
    def __init__(
            self,
            init_example,
            min_label_diversity=0,
            label_key=None,
            min_dataset_examples=None,
            bucket_id=None,
            **kwargs
    ):
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
        self.min_label_diversity = min_label_diversity
        if min_label_diversity > 0:
            assert label_key is not None
            assert min_label_diversity <= self.batch_size
        self.label_key = label_key
        self.label_classes = set()
        if min_dataset_examples is not None:
            self.missing_dataset_examples = {
                key: value for key, value in min_dataset_examples.items()
            }
            assert sum(min_dataset_examples.values()) <= self.batch_size
        else:
            self.missing_dataset_examples = None
        self.bucket_id = bucket_id

    def assess(self, example):
        if example["example_id"] in self.example_ids:
            return False
        if self.bucket_id is not None and example[self.bucket_id] != self.data[0][self.bucket_id]:
            return False
        if not super().assess(example):
            return False
        if self.missing_dataset_examples is not None:
            dataset_names = example['dataset'].split('+')  # '+' indicates mixtures
            assert all([name in self.missing_dataset_examples for name in dataset_names]), (
                dataset_names, sorted(self.missing_dataset_examples.keys())
            )
            if not (
                (self.batch_size - len(self.data)) > sum(self.missing_dataset_examples.values())
                or
                any([self.missing_dataset_examples[name] > 0 for name in dataset_names])
            ):
                return False
        if not (
            (self.batch_size - len(self.data)) > (self.min_label_diversity - len(self.label_classes))
            or
            any([label not in self.label_classes for label in to_list(example.get(self.label_key, []))])
        ):
            return False
        return True

    def _append(self, example):
        super()._append(example)
        self.example_ids.add(example["example_id"])
        if self.missing_dataset_examples is not None:
            for name in example['dataset'].split('+'):
                if self.missing_dataset_examples[name] > 0:
                    self.missing_dataset_examples[name] -= 1
        if self.label_key is not None and self.label_key in example:
            self.label_classes.update(to_list(example[self.label_key]))


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
