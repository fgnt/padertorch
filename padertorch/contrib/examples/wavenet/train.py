from pathlib import Path
import os

from paderbox.database import JsonDatabase
from paderbox.io.data_dir import database_jsons
from paderbox.utils.nested import deflatten
from padertorch.configurable import Configurable, config_to_instance
from padertorch.data.fragmenter import TimeFragmenter
from padertorch.data.transforms import Compose, ReadAudio, Spectrogram, \
    GlobalNormalize
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.observers import FileStorageObserver
from torch.utils.data.dataloader import default_collate


ex = Exp('wavenet')
storage_dir = str(Path(os.environ['STORAGE_DIR']))
observer = FileStorageObserver.create(storage_dir)
ex.observers.append(observer)


@ex.config
def config():
    # Data configuration
    data_config = dict(
        database="timit",
        training_sets="train",
        validation_sets="test_core",
        transform_config=dict(
            reader=dict(cls=ReadAudio),
            spectrogram=dict(cls=Spectrogram)
        ),
        batch_size=8,
    )
    Configurable.get_config(
        out_config=data_config['transform_config']['reader']
    )
    Configurable.get_config(
        out_config=data_config['transform_config']['spectrogram']
    )
    data_config['transform_config']['spectrogram']['kwargs']['sample_rate'] = \
        data_config['transform_config']['reader']['kwargs']['sample_rate']

    # Model configuration
    model_config = dict(cls=WaveNet)
    Configurable.get_config(
        out_config=model_config,
        updates=deflatten(
            {
                'wavenet.kwargs': {
                    'n_cond_channels': data_config['transform_config'][
                        'spectrogram']['kwargs']['n_mels'],
                    'upsamp_window': data_config['transform_config'][
                        'spectrogram']['kwargs']['frame_length'],
                    'upsamp_stride': data_config['transform_config'][
                        'spectrogram']['kwargs']['frame_step']
                },
                'sample_rate': data_config['transform_config']['reader'][
                    'kwargs']['sample_rate']
            }
        )
    )

    # Optimizer configuration
    optimizer_config = dict(cls=Adam)
    Configurable.get_config(
        out_config=optimizer_config
    )

    # Trainer configuration
    # TODO: LD beantworten, warum train_config anders als model_config.
    train_config = Trainer.get_signature()
    del train_config['optimizer']
    train_config.update(
        dict(
            summary_trigger=(1000, 'iteration'),
            checkpoint_trigger=(1, 'epoch'),
            max_trigger=(20, 'epoch')
        )
    )


@ex.automain
def train(
        _config, data_config, model_config, optimizer_config, train_config
):

    db = JsonDatabase(
        json_path=database_jsons / f'{data_config["database"]}.json'
    )

    train_iter = db.get_iterator_by_names(data_config['training_sets'])
    validation_iter = db.get_iterator_by_names(data_config['validation_sets'])

    read = Configurable.from_config(data_config['transform_config']['reader'])
    spec = Configurable.from_config(
        data_config['transform_config']['spectrogram']
    )

    def to_tuple(example):
        return example["spectrogram"][0], example["audio_data"]

    train_iter = train_iter.shuffle(reshuffle=True)

    # TODO: LD: Why shuffle validation?
    validation_iter = validation_iter.shuffle()

    buffer_size = 2*data_config['batch_size']
    # TODO: LD: Why Compose and not map().map() und am Ende prefetch()?
    train_iter = train_iter.map(
        Compose(read, spec), num_workers=min(buffer_size, 16),
        buffer_size=buffer_size
    )

    validation_iter = validation_iter.map(
        Compose(read, spec), num_workers=min(buffer_size, 16),
        buffer_size=buffer_size
    )

    norm = GlobalNormalize()
    norm.init_moments(iterator=validation_iter, storage_dir=storage_dir)
    train_iter = train_iter.map(norm)
    validation_iter = validation_iter.map(norm)

    train_iter = train_iter.fragment(
        TimeFragmenter(
            {'spectrogram': 100, 'audio_data': 16000},
            training=True, drop_last=True
        )
    ).shuffle(reshuffle=True, buffer_size=1000).map(to_tuple)
    validation_iter = validation_iter.fragment(
        TimeFragmenter(
            {'spectrogram': 100, 'audio_data': 16000}, drop_last=True
        )
    ).map(to_tuple)

    train_iter = train_iter.batch(
        data_config['batch_size']).map(default_collate)
    validation_iter = validation_iter.batch(
        data_config['batch_size']).map(default_collate)

    model = Configurable.from_config(model_config)

    optimizer = Configurable.from_config(optimizer_config)

    trainer = Trainer(
        model=model,
        storage_dir=storage_dir,
        optimizer=optimizer,
        **config_to_instance(train_config)
    )
    trainer.test_run(train_iter, validation_iter)
    trainer.train(train_iter, validation_iter)
