import numbers
from collections import OrderedDict
from functools import partial

from cached_property import cached_property
from natsort import natsorted
from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from padertorch import Configurable
from torch.utils.data.dataloader import default_collate


class DataProvider(Configurable):
    def __init__(
            self, database_name, training_set_names, validation_set_names=None,
            transforms=None, subset_size=None, storage_dir=None,
            max_workers=0, prefetch_buffer=100,
            fragment=False, shuffle_buffer=None,
            batch_size=None, collate_fn=None
    ):
        self.database_name = database_name
        self.training_set_names = training_set_names
        self.validation_set_names = validation_set_names
        if transforms is None:
            self.transforms = []
        elif isinstance(transforms, dict):
            if not isinstance(transforms, OrderedDict):
                self.transforms = list()
                for i, key in enumerate(natsorted(transforms.keys())):
                    try:
                        idx = int(key)
                        assert idx == i
                    except (ValueError, AssertionError):
                        raise ValueError(
                            'transforms is an unordered dict '
                            'with keys not being an enumeration.'
                        )
                    self.transforms.append(transforms[key])
            else:
                self.transforms = [
                    transform for key, transform in transforms.items()
                ]
        elif isinstance(transforms, (list, tuple)):
            self.transforms = transforms
        elif callable(transforms):
            self.transforms = [transforms]
        else:
            raise ValueError(
                f'transforms of type {type(transforms)} not allowed'
            )
        self.subset_size = subset_size
        self.storage_dir = storage_dir
        self.num_workers = min(prefetch_buffer, max_workers)
        self.prefetch_buffer = prefetch_buffer
        self.fragment = fragment
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        self.initialized_transforms = False
        self.get_train_iterator()

    @cached_property
    def db(self):
        return JsonDatabase(
            json_path=database_jsons / f'{self.database_name}.json'
        )

    def prefetch(self, dataset):
        if self.num_workers > 0:
            assert self.prefetch_buffer is not None
            dataset = dataset.prefetch(
                self.num_workers, self.prefetch_buffer
            )
        return dataset

    def get_iterator(
            self, dataset_names, training=False, shuffle=False, fragment=True,
            batch=True
    ):
        if dataset_names is None:
            return None

        dataset = self.db.get_iterator_by_names(dataset_names)

        for transform in self.transforms:
            if not self.initialized_transforms:
                assert dataset_names == self.training_set_names
                subset = dataset
                if self.subset_size is not None:
                    assert isinstance(self.subset_size, numbers.Integral)
                    subset = subset.shuffle()[:self.subset_size]
                transform.init_params(
                    storage_dir=self.storage_dir, dataset=self.prefetch(subset)
                )
            transform = partial(transform, training=training)
            dataset = dataset.map(transform)
        self.initialized_transforms = True

        if shuffle:
            dataset = dataset.shuffle(reshuffle=True)
        dataset = self.prefetch(dataset)

        if self.fragment and fragment:
            dataset = dataset.unbatch()
            if self.shuffle_buffer is not None and shuffle:
                dataset = dataset.shuffle(
                    reshuffle=True, buffer_size=self.shuffle_buffer
                )

        if batch:
            dataset = dataset.batch(self.batch_size)
            if self.collate_fn is None:
                collate_fn = default_collate
            else:
                collate_fn = self.collate_fn
            dataset = dataset.map(collate_fn)
        return dataset

    def get_train_iterator(self):
        return self.get_iterator(
            self.training_set_names, training=True, shuffle=True
        )

    def get_validation_iterator(self):
        return self.get_iterator(self.validation_set_names)
