import numpy as np
from padertorch import Configurable
from padertorch.contrib.je.data.transforms import Collate
from functools import partial


class DataProvider(Configurable):
    def __init__(
            self,
            transform=None,
            prefetch_buffer=None, max_workers=8,
            shuffle_buffer=None, batch_size=None
    ):
        self.transform = transform
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = 0 if prefetch_buffer is None \
            else min(prefetch_buffer, max_workers)
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size

    def prepare_iterable(
            self,
            dataset,
            training=False,
            shuffle=True,
            prefetch=True,
            fragment=False,
            batch=True
    ):
        if self.transform is not None:
            dataset = dataset.map(partial(self.transform, training=training))

        if shuffle:
            dataset = dataset.shuffle(reshuffle=True)

        if prefetch and self.prefetch_buffer and self.num_workers:
            dataset = dataset.prefetch(self.num_workers, self.prefetch_buffer)

        if fragment:
            dataset = dataset.unbatch()
            if shuffle and self.shuffle_buffer:
                dataset = dataset.shuffle(
                    reshuffle=True, buffer_size=self.shuffle_buffer
                )

        if batch:
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(Collate())

        return dataset


def split_dataset(dataset, splits, seed):
    splits = np.cumsum(splits)
    assert splits[-1] <= 1.
    splits = (splits*len(dataset)).astype(np.int64).tolist()
    indices = np.arange(len(dataset))
    np.random.RandomState(seed).shuffle(indices)
    split_indices = np.split(indices, splits)
    return (
        dataset[sorted(indices.tolist())] for indices in split_indices
    )