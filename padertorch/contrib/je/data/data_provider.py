from collections import OrderedDict
from functools import partial

from cached_property import cached_property
from natsort import natsorted
from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from padertorch import Configurable
from padertorch.contrib.je.data.transforms import Collate
from padertorch import utils
from lazy_dataset import Dataset
import numpy as np
import lazy_dataset


class DataProvider(Configurable):
    def __init__(
            self, database_name, training_sets, validation_sets,
            seed=int(np.random.choice(2**32)),
            pre_transform_filters=None,
            transforms=None, subset_size=None, storage_dir=None,
            post_transform_filters=None,
            max_workers=0, prefetch_buffer=100,
            fragment=False, shuffle_buffer=None,
            batch_size=None
    ):
        self.database_name = database_name

        if isinstance(training_sets, str):
            training_sets = training_sets.split(',')
        if isinstance(validation_sets, str):
            validation_sets = validation_sets.split(',')
        self.training_sets = training_sets
        self.validation_sets = validation_sets
        self.seed = seed

        self.pre_transform_filter = None if pre_transform_filters is None \
            else lambda ex: all([
                fil(ex) for fil in to_list(pre_transform_filters)
                if fil is not None
            ])
        if transforms is None:
            self.transforms = []
        else:
            self.transforms = to_list(transforms)
        self.subset_size = subset_size
        self.storage_dir = storage_dir
        self.post_transform_filter = None if post_transform_filters is None \
            else lambda ex: all([
                fil(ex) for fil in to_list(post_transform_filters)
                if fil is not None
            ])
        self.num_workers = min(prefetch_buffer, max_workers)
        self.prefetch_buffer = prefetch_buffer
        self.fragment = fragment
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size

        self.initialized_transforms = False

    @cached_property
    def db(self):
        return JsonDatabase(
            json_path=database_jsons / f'{self.database_name}.json'
        )

    def prepare_dataset(self, dataset, training=False):

        if not isinstance(dataset, Dataset):
            dataset = self.db.get_dataset(dataset)

        if self.pre_transform_filter is not None:
            dataset = dataset.filter(self.pre_transform_filter, lazy=False)

        for transform in self.transforms:
            if transform is None:
                continue
            if not self.initialized_transforms:
                subset = dataset
                if self.subset_size is not None:
                    subset = subset.shuffle()[:self.subset_size]
                transform.init_params(
                    storage_dir=self.storage_dir, dataset=self.prefetch(subset)
                )
            transform = partial(transform, training=training)
            dataset = dataset.map(transform)
        self.initialized_transforms = True

        if self.post_transform_filter is not None:
            dataset = dataset.filter(self.post_transform_filter, lazy=False)

        return dataset

    def prepare_iterable(
            self, dataset, shuffle=True, prefetch=True, batch=True
    ):
        if shuffle:
            dataset = dataset.shuffle(reshuffle=True)

        if prefetch:
            dataset = self.prefetch(dataset)

        if self.fragment:
            dataset = dataset.unbatch()
            if self.shuffle_buffer is not None and shuffle:
                dataset = dataset.shuffle(
                    reshuffle=True, buffer_size=self.shuffle_buffer
                )

        if batch:
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(Collate())

        return dataset

    def prefetch(self, dataset):
        if self.num_workers > 0:
            assert self.prefetch_buffer is not None
            dataset = dataset.prefetch(
                self.num_workers, self.prefetch_buffer
            )
        return dataset

    def split_dataset(self, dataset, splits):
        splits = np.cumsum(splits).astype(np.int64).tolist()
        assert splits[-1] <= len(dataset)
        indices = np.arange(len(dataset))
        np.random.RandomState(self.seed).shuffle(indices)
        split_indices = np.split(indices, splits)
        return (
            dataset[sorted(indices.tolist())] for indices in split_indices
        )

    def get_iterables(self):
        training_sets = list()
        validation_sets = list()
        for name in {*self.training_sets, *self.validation_sets}:
            dataset = self.prepare_dataset(self.db.get_dataset(name))
            if (
                name in self.training_sets
                and name in self.validation_sets
            ):
                assert (
                    isinstance(self.training_sets, dict)
                    and isinstance(self.validation_sets, dict)
                )
                train_set, validate_set, _ = self.split_dataset(
                    dataset,
                    [self.training_sets[name], self.validation_sets[name]]
                )
                print(len(train_set), len(validate_set))
                training_sets.append(train_set)
                validation_sets.append(validate_set)
            elif name in self.training_sets:
                if isinstance(self.training_sets, dict):
                    dataset, _ = self.split_dataset(
                        dataset, [self.training_sets[name]]
                    )
                training_sets.append(dataset)
            elif name in self.validation_sets:
                if isinstance(self.validation_sets, dict):
                    dataset, _ = self.split_dataset(
                        dataset, [self.training_sets[name]]
                    )
                validation_sets.append(dataset)
        training_iterable = self.prepare_iterable(
            lazy_dataset.concatenate(training_sets)
        )
        validation_iterable = self.prepare_iterable(
            lazy_dataset.concatenate(validation_sets)
        )
        return training_iterable, validation_iterable


def to_list(value):
    if isinstance(value, dict):
        if not isinstance(value, OrderedDict):
            l = list()
            for i, key in enumerate(natsorted(value.keys())):
                try:
                    idx = int(key)
                    assert idx == i
                except (ValueError, AssertionError):
                    raise ValueError(
                        'transforms is an unordered dict '
                        'with keys not being an enumeration.'
                    )
                l.append(value[key])
        else:
            l = [transform for key, transform in value.items()]
    else:
        l = utils.to_list(value)
    return l
