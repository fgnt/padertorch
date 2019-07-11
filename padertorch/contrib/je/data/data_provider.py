import numpy as np
from padertorch import Configurable
from padertorch.contrib.je.data.transforms import Collate
from functools import partial


class DataProvider(Configurable):
    def __init__(
            self,
            transform=None,
            prefetch_buffer=None, max_workers=8, backend='t',
            shuffle_buffer=None, batch_size=None, bucketing_key='seq_len',
            max_padding_rate=None, total_size_threshold=None,
            bucket_expiration=None
    ):
        self.transform = transform
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = 0 if prefetch_buffer is None \
            else min(prefetch_buffer, max_workers)
        self.backend = backend
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.bucketing_key = bucketing_key
        self.max_padding_rate = max_padding_rate
        self.total_size_threshold = total_size_threshold
        self.bucket_expiration = bucket_expiration

    def prepare_iterable(
            self,
            dataset,
            transform=True,
            training=False,
            shuffle=True,
            prefetch=True,
            fragment=False,
            reps=1,
            batch=True,
            drop_incomplete=False
    ):
        if transform and self.transform is not None:
            dataset = dataset.map(partial(self.transform, training=training))

        if shuffle:
            dataset = dataset.shuffle(reshuffle=True)

        if prefetch and self.prefetch_buffer and self.num_workers:
            dataset = dataset.prefetch(
                self.num_workers, self.prefetch_buffer,
                catch_filter_exception=True, backend=self.backend
            )

        if fragment:
            dataset = dataset.unbatch()
            if shuffle and self.shuffle_buffer:
                dataset = dataset.shuffle(
                    reshuffle=True, buffer_size=self.shuffle_buffer
                )

        assert reps > 0
        if reps > 1:
            dataset = dataset.tile(reps)
        if batch:
            if self.bucketing_key:
                dataset = dataset.batch_bucket_dynamic(
                    batch_size=self.batch_size,
                    key=self.bucketing_key,
                    max_padding_rate=self.max_padding_rate,
                    total_size_threshold=self.total_size_threshold,
                    expiration=self.bucket_expiration,
                    drop_incomplete=drop_incomplete,
                    sort_by_key=True
                )
            else:
                dataset = dataset.batch(self.batch_size)

            dataset = dataset.map(Collate())

        return dataset


def split_dataset(dataset, fold, nfolfds=5, seed=0):
    """

    Args:
        dataset:
        fold:
        nfolfds:
        seed:

    Returns:

    >>> split_dataset(np.array([1,2,3,4,5]), 0, nfolfds=2)
    [array([1, 3]), array([2, 4, 5])]
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
