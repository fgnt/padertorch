from collections import OrderedDict
from functools import partial
from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from padertorch.contrib.je.transforms import Compose, GlobalNormalize
from padertorch import Configurable
from cached_property import cached_property
import numbers
from torch.utils.data.dataloader import default_collate


class DataProvider(Configurable):
    def __init__(
            self, database_name, training_set_names,
            validation_set_names=None, test_set_names=None,
            label_encoders=None, transforms=None,
            normalize_features=None, subset_size=None, storage_dir=None,
            segmenters=None, fragmenter=None,
            max_workers=0, prefetch_buffer=None,
            shuffle_buffer=None,
            batch_size=None, collate_fn=None
    ):
        self.database_name = database_name
        self.training_set_names = training_set_names
        self.validation_set_names = validation_set_names
        self.test_set_names = test_set_names
        self.label_encoders = label_encoders
        self.transforms = transforms
        self.normalize_features = normalize_features
        self.subset_size = subset_size
        self.storage_dir = storage_dir
        self.segmenters = segmenters
        self.fragmenter = fragmenter
        self.num_workers = min(prefetch_buffer, max_workers)
        self.prefetch_buffer = prefetch_buffer
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    @cached_property
    def db(self):
        return JsonDatabase(
            json_path=database_jsons / f'{self.database_name}.json'
        )

    @cached_property
    def labels_encoder(self):
        if self.label_encoders is None:
            return None

        if isinstance(self.label_encoders, (list, tuple)):
            label_encoders = self.label_encoders
        elif isinstance(self.label_encoders, OrderedDict):
            label_encoders = [
                label_encoder
                for key, label_encoder in self.label_encoders.items()
            ]
        elif callable(self.label_encoders):
            label_encoders = [self.label_encoders]
        else:
            raise ValueError
        for label_encoder in label_encoders:
            dataset = self.db.get_iterator_by_names(
                self.training_set_names
            )
            if self.subset_size is not None:
                dataset = dataset.shuffle()[:self.subset_size]
            label_encoder.init_labels(
                storage_dir=self.storage_dir,
                dataset=dataset
            )
        return Compose(label_encoders)

    @cached_property
    def transform(self):
        if self.transforms is None:
            return None
        return Compose(self.transforms)

    @cached_property
    def normalizer(self):
        if not self.normalize_features:
            return
        norm = GlobalNormalize(self.normalize_features)

        dataset = self.db.get_iterator_by_names(self.training_set_names)
        if self.subset_size is not None:
            assert isinstance(self.subset_size, numbers.Integral)
            dataset = dataset.shuffle()[:self.subset_size]
        if self.transform:
            if self.labels_encoder is not None:
                dataset = dataset.map(self.labels_encoder)
            if self.transform is not None:
                dataset = dataset.map(self.transform)
            dataset = self.maybe_prefetch(dataset)

        norm.init_moments(
            dataset=dataset, storage_dir=self.storage_dir
        )
        return norm

    @cached_property
    def segmenter(self):
        if self.segmenters is None:
            return
        return Compose(self.segmenters)

    def maybe_prefetch(self, dataset):
        if self.num_workers > 0:
            assert self.prefetch_buffer is not None
            dataset = dataset.prefetch(
                self.num_workers, self.prefetch_buffer
            )
        return dataset

    def get_iterator(self, dataset_names, training=False):
        if dataset_names is None:
            return None
        dataset = self.db.get_iterator_by_names(dataset_names)

        for func in [
            self.labels_encoder, self.transform, self.normalizer,
            self.segmenter, self.fragmenter
        ]:
            if func is not None:
                func = partial(func, training=training)
                dataset = dataset.map(func)

        if training:
            dataset = dataset.shuffle(reshuffle=True)

        dataset = self.maybe_prefetch(dataset)

        if self.fragmenter is not None:
            dataset = dataset.unbatch()
            if training:
                dataset = dataset.shuffle(
                    reshuffle=True, buffer_size=self.shuffle_buffer
                )

        dataset = dataset.batch(self.batch_size)
        if self.collate_fn is None:
            collate_fn = default_collate
        else:
            collate_fn = self.collate_fn
        dataset = dataset.map(collate_fn)
        return dataset

    def get_train_iterator(self):
        return self.get_iterator(self.training_set_names, training=True)

    def get_validation_iterator(self):
        return self.get_iterator(self.validation_set_names, training=False)

    def get_test_iterator(self):
        return self.get_iterator(self.test_set_names, training=False)
