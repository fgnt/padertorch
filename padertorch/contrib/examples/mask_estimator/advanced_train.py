"""
Advanced training script for a mask_estimator. Uses sacred and configurable
to create a config, instantiate the model and Trainer and write everything
to a model file.
May be called as follows:
python -m padertorch.contrib.examples.mask_estimator.advanced_train


"""
from pathlib import Path
import os

import sacred
from padercontrib.database.chime import Chime3
from padercontrib.database.keys import OBSERVATION, NOISE_IMAGE, SPEECH_IMAGE
from paderbox.io import dump_json
from paderbox.utils.nested import deflatten
from paderbox.utils.pretty import pprint
from padertorch.configurable import Configurable
from padertorch.configurable import config_to_instance, recursive_class_to_str
from padertorch.contrib.jensheit.data import SequenceProvider, MaskTransformer
from paderbox.transform.module_stft import STFT
from padertorch.contrib.jensheit.utils import get_experiment_name
from padertorch.contrib.jensheit.utils import compare_configs
from padertorch.models.mask_estimator import MaskEstimatorModel
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred.utils import apply_backspaces_and_linefeeds
from copy import deepcopy

model_dir = Path(os.environ['STORAGE_ROOT'])
ex = sacred.Experiment('Train Mask Estimator')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    model_class = MaskEstimatorModel
    trainer_opts = deflatten({
        'model.factory': model_class,
        'optimizer.factory': Adam,
        'stop_trigger': (int(1e5), 'iteration'),
        'summary_trigger': (500, 'iteration'),
        'checkpoint_trigger': (500, 'iteration'),
        'storage_dir': None,
    })
    provider_opts = deflatten({
        'factory': SequenceProvider,
        'database.factory': Chime3,
        'audio_keys': [OBSERVATION, NOISE_IMAGE, SPEECH_IMAGE],
        'transform.factory': MaskTransformer,
        'transform.stft': dict(factory=STFT, shift=256, size=1024),
    })
    trainer_opts['model']['transformer'] = provider_opts['transform']

    storage_dir = None
    add_name = None
    if storage_dir is None:
        ex_name = get_experiment_name(trainer_opts['model'])
        if add_name is not None:
            ex_name += f'_{add_name}'
        observer = sacred.observers.FileStorageObserver(
            str(model_dir / ex_name))
        storage_dir = observer.basedir
    else:
        sacred.observers.FileStorageObserver.create(storage_dir)

    trainer_opts['storage_dir'] = storage_dir

    if (Path(storage_dir) / 'init.json').exists():
        trainer_opts, provider_opts = compare_configs(
            storage_dir, trainer_opts, provider_opts)

    Trainer.get_config(
        trainer_opts
    )
    Configurable.get_config(
        provider_opts
    )
    validate_checkpoint = 'ckpt_latest.pth'
    validation_kwargs = dict(
        metric = 'loss',
        maximize = False,
        max_checkpoints=1,
    )
    validation_length = 1000  # number of examples taken from the validation iterator



@ex.capture
def initialize_trainer_provider(task, trainer_opts, provider_opts, _run):

    storage_dir = Path(trainer_opts['storage_dir'])

    trainer_opts = deepcopy(trainer_opts)
    provider_opts = deepcopy(provider_opts)
    if (storage_dir / 'init.json').exists():
        assert task in ['restart', 'validate'], task
    elif task in ['train', 'create_checkpoint']:
        dump_json(dict(trainer_opts=recursive_class_to_str(trainer_opts),
                       provider_opts=recursive_class_to_str(provider_opts)),
                  storage_dir / 'init.json')
    else:
        raise ValueError(task, storage_dir)
    pprint('provider_opts:')
    pprint(provider_opts)
    pprint('trainer_opts:')
    pprint(trainer_opts)

    trainer = Trainer.from_config(trainer_opts)
    assert isinstance(trainer, Trainer)
    provider = config_to_instance(provider_opts)
    return trainer, provider


@ex.command
def restart(validation_kwargs):
    trainer, provider = initialize_trainer_provider(task='restart')
    train_iterator = provider.get_train_iterator()
    validation_iterator = provider.get_eval_iterator(
        num_examples=validation_kwargs.pop('validation_length')
    )
    trainer.load_checkpoint()
    trainer.test_run(train_iterator, validation_iterator)
    trainer.register_validation_hook(
        validation_iterator, **validation_kwargs)
    trainer.train(train_iterator, resume=True)


@ex.command
def validate(_config):
    import numpy as np
    import os
    from padertorch.contrib.jensheit.evaluation import evaluate_masks
    from functools import partial
    from paderbox.io import dump_json
    from concurrent.futures import ThreadPoolExecutor
    assert len(ex.current_run.observers) == 1, (
        'FileObserver` missing. Add a `FileObserver` with `-F foo/bar/`.'
    )
    storage_dir = Path(ex.current_run.observers[0].basedir)
    assert not (storage_dir / 'results.json').exists(), (
        f'model_dir has already bin evaluatet, {storage_dir}')
    trainer, provider = initialize_trainer_provider(task='validate')
    trainer.model.cpu()
    eval_iterator = provider.get_eval_iterator()
    evaluation_json = dict(snr=dict(), pesq=dict())
    provider.opts.multichannel = True
    batch_size = 1
    provider.opts.batch_size = batch_size
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        for example_id, snr, pesq in executor.map(
                partial(evaluate_masks, model=trainer.model,
                        transform=provider.transformer.stft), eval_iterator):
            evaluation_json['snr'][example_id] = snr
            evaluation_json['pesq'][example_id] = pesq
    evaluation_json['pesq_mean'] = np.mean(
        [value for value in evaluation_json['pesq'].values()])
    evaluation_json['snr'] = np.mean(
        [value for value in evaluation_json['snr'].values()])
    dump_json(evaluation_json, storage_dir / 'results.json')


@ex.command
def create_checkpoint(_config):
    # This may be useful to merge to separatly trained models into one
    raise NotImplementedError


@ex.automain
def train(validation_kwargs, validation_length):
    trainer, provider = initialize_trainer_provider(task='train')
    train_iterator = provider.get_train_iterator()
    validation_iterator = provider.get_eval_iterator(
        num_examples=validation_length
    )
    trainer.test_run(train_iterator, validation_iterator)
    trainer.register_validation_hook(
        validation_iterator, **validation_kwargs)
    trainer.train(train_iterator)
