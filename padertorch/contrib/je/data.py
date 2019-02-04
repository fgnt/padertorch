from collections import OrderedDict
from functools import partial
from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from padertorch.data.transforms import Compose, GlobalNormalize
from padertorch import Configurable
from cached_property import cached_property
import numbers
from torch.utils.data.dataloader import default_collate


class DataProvider(Configurable):
    def __init__(
            self, database_name, training_set_names, validation_set_names=None,
            label_encoders=None, transforms=None, max_workers=0,
            normalize_features=None, subset_size=None, storage_dir=None,
            fragmenters=None, shuffle_buffer_size=1000, batch_size=None,
            collate_fn=None
    ):
        self.database_name = database_name
        self.training_set_names = training_set_names
        self.validation_set_names = validation_set_names
        self.label_encoders = label_encoders
        self.transforms = transforms
        self.num_workers = min(2*batch_size, max_workers)
        self.normalize_features = normalize_features
        self.subset_size = subset_size
        self.storage_dir = storage_dir
        self.fragmenters = fragmenters
        self.shuffle_buffer_size = shuffle_buffer_size
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
            data_pipe = self.db.get_iterator_by_names(
                self.training_set_names
            )
            if self.subset_size is not None:
                data_pipe = data_pipe.shuffle()[:self.subset_size]
            label_encoder.init_labels(
                storage_dir=self.storage_dir,
                iterator=data_pipe
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

        data_pipe = self.db.get_iterator_by_names(self.training_set_names)
        if self.subset_size is not None:
            assert isinstance(self.subset_size, numbers.Integral)
            data_pipe = data_pipe.shuffle()[:self.subset_size]
        if self.transform:
            data_pipe = data_pipe.map(
                self.transform, num_workers=self.num_workers,
                buffer_size=2*self.batch_size
            )

        norm.init_moments(
            iterator=data_pipe, storage_dir=self.storage_dir
        )
        return norm

    def _get_iterator(self, dataset_names, training=False):
        if dataset_names is None:
            return None
        data_pipe = self.db.get_iterator_by_names(dataset_names)

        map_fn = [
            func for func in
            [self.labels_encoder, self.transform, self.normalizer]
            if func is not None
        ]
        if map_fn:
            data_pipe = data_pipe.map(
                Compose(map_fn), num_workers=self.num_workers,
                buffer_size=2*self.batch_size
            )

        if training:
            data_pipe = data_pipe.shuffle(reshuffle=True)

        if self.fragmenters is not None:
            if isinstance(self.fragmenters, (list, tuple)):
                fragmentations = self.fragmenters
            elif isinstance(self.fragmenters, OrderedDict):
                fragmentations = [
                    fragmenter for key, fragmenter in self.fragmenters.items()
                ]
            elif callable(self.fragmenters):
                fragmentations = [self.fragmenters]
            else:
                raise ValueError
            for fragment_fn in fragmentations:
                data_pipe = data_pipe.fragment(
                    partial(fragment_fn, random_onset=training)
                )
            data_pipe = data_pipe.shuffle(
                reshuffle=True, buffer_size=self.shuffle_buffer_size
            )

        data_pipe = data_pipe.batch(self.batch_size)
        if self.collate_fn is None:
            collate_fn = default_collate
        else:
            collate_fn = self.collate_fn
        data_pipe = data_pipe.map(collate_fn)
        return data_pipe

    def get_train_iterator(self):
        return self._get_iterator(self.training_set_names, training=True)

    def get_validation_iterator(self):
        return self._get_iterator(self.validation_set_names, training=False)
