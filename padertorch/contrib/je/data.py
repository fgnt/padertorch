from collections import OrderedDict
from functools import partial
from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from padertorch.data.transforms import Compose, GlobalNormalize


def datasets(
        database_name, training_set_names, validation_set_names=None,
        transforms=None, max_workers=0, normalize_features=None,
        subset_size=1000, storage_dir=None, fragmentations=None,
        shuffle_buffer_size=1000, batch_size=None
):

    db = JsonDatabase(
        json_path=database_jsons / f'{database_name}.json'
    )

    training_iter = db.get_iterator_by_names(training_set_names)
    if validation_set_names is not None:
        validation_iter = db.get_iterator_by_names(validation_set_names)
    else:
        validation_iter = None

    if transforms is not None:
        transform = Compose(transforms)
    else:
        transform = None

    buffer_size = 2*batch_size
    if normalize_features:
        norm = GlobalNormalize(normalize_features)
        subset_iter = training_iter.shuffle()[:subset_size]
        if transform is not None:
            subset_iter = subset_iter.map(
                transform, num_workers=min(buffer_size, max_workers),
                buffer_size=buffer_size
            )
        norm.init_moments(
            iterator=subset_iter, storage_dir=storage_dir
        )
        if transform is None:
            transform = norm
        else:
            transform = Compose(transform, norm)

    if transform is not None:
        training_iter = training_iter.map(
            transform, num_workers=min(buffer_size, max_workers),
            buffer_size=buffer_size
        )
        if validation_iter is not None:
            validation_iter = validation_iter.map(
                transform, num_workers=min(buffer_size, max_workers),
                buffer_size=buffer_size
            )

    training_iter = training_iter.shuffle(reshuffle=True)
    if fragmentations is not None:
        if isinstance(fragmentations, (list, tuple)):
            pass
        elif isinstance(fragmentations, OrderedDict):
            fragmentations = [
                fragmenter for key, fragmenter in fragmentations.items()
            ]
        elif callable(fragmentations):
            fragmentations = [fragmentations]
        for fragmenter in fragmentations:
            training_iter = training_iter.fragment(
                partial(fragmenter, training=True)
            )
            if validation_iter is not None:
                validation_iter = validation_iter.fragment(
                    partial(fragmenter, training=False)
                )
        training_iter = training_iter.shuffle(
            reshuffle=True, buffer_size=shuffle_buffer_size
        )

    training_iter = training_iter.batch(batch_size)
    if validation_iter is not None:
        validation_iter = validation_iter.batch(batch_size)

    return training_iter, validation_iter
