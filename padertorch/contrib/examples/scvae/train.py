"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.scvae.train print_config
python -m padertorch.contrib.examples.scvae.train
"""
import os
from pathlib import Path

from paderbox.database.timit import Timit
from paderbox.database.librispeech import LibriSpeech
from paderbox.utils.nested import deflatten
from paderbox.utils.timer import timeStamped
from padertorch.contrib.examples.scvae.data import Transform
from padertorch.contrib.examples.scvae.model import SCVAE
from padertorch.contrib.je.data.data_provider import DataProvider, \
    split_dataset
from padertorch.contrib.je.modules.conv import CNN1d, MultiScaleCNN1d
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver

nickname = 'scvae-training'
ex = Exp(nickname)
storage_dir = str(
    Path(os.environ['STORAGE_ROOT']) / nickname / timeStamped('')[1:]
)
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


@ex.config
def config():
    debug = False

    # Data configuration
    data_provider = {
        'transform': {
            'factory': Transform,
            'storage_dir': storage_dir
        },
        'max_workers': 8,
        'batch_size': 64
    }
    data_provider['prefetch_buffer'] = 10 * data_provider['batch_size']
    data_provider['shuffle_buffer'] = 10 * data_provider['batch_size']
    DataProvider.get_config(data_provider)

    # Trainer configuration
    trainer = deflatten({
        'model.factory':  SCVAE,
        'model.feature_key': 'log_mel',
        'model.condition_key': 'speaker_id',
        'model.n_conditions': 462,
        'model.condition_dim': 32,
        'model.encoder': {
            'factory': CNN1d,
            'in_channels': data_provider['transform']['n_mels'],
            'hidden_channels': 256,
            'num_layers': 6,
            'kernel_size': 5,
            # 'num_scales': 2,
            'stride': 1,
            'gated': False,
            'dropout': 0.,
            'norm': None,
            'activation': 'relu'
        },
        'model.decoder': {
            'in_channels': 64
        },
        'loss_weights': {'mse': 1., 'kld': 1e-2, 'ce': 1.},
        'optimizer.encoder.factory': Adam,
        'optimizer.encoder.lr': 1e-4,
        # 'optimizer.encoder.weight_decay': 1e-4,
        'optimizer.decoder.factory': Adam,
        'optimizer.decoder.lr': 1e-4,
        # 'optimizer.decoder.weight_decay': 1e-4,
        'optimizer.embed.factory': Adam,
        'optimizer.embed.lr': 1e-3,
        # 'optimizer.embed.weight_decay': 1e-4,
        'optimizer.clf.factory': Adam,
        'optimizer.clf.lr': 1e-3,
        # 'optimizer.clf.weight_decay': 1e-4,
        'storage_dir': storage_dir,
        'summary_trigger': (100, 'iteration'),
        'checkpoint_trigger': (1000, 'iteration'),
        'max_trigger': (20000, 'iteration')
    })
    Trainer.get_config(trainer)


@ex.capture
def get_datasets(data_provider, debug):
    dp = DataProvider.from_config(data_provider)
    db = Timit()
    training_set = db.get_dataset('train')

    # def fix_speaker_id(example):
    #     example['speaker_id'] = example['speaker_id'].split('-')[0]
    #     return example
    #
    # training_set = training_set.map(fix_speaker_id)

    dp.transform.initialize_norm(
        dataset=training_set.shuffle()[:(10 if debug else 5000)],
        max_workers=dp.num_workers
    )
    dp.transform.initialize_labels(dataset=training_set)
    print(len(dp.transform.label_mapping))
    training_set, validation_set = split_dataset(training_set, 0, nfolfds=5)

    return (
        dp.prepare_iterable(training_set, fragment=True, training=True),
        dp.prepare_iterable(validation_set, fragment=True)
    )


@ex.automain
def train(trainer):
    train_iter, validation_iter = get_datasets()

    trainer = Trainer.from_config(trainer)
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
