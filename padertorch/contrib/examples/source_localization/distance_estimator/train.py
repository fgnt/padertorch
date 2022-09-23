"""
Training script for a distance estimator using CRNNs
like proposed in "On Source-Microphone Distance Estimation Using Convolutional
Recurrent Neural Networks" (Speech Communication; 14th ITG Conference from
29 September 2021 - 01 October 2021).

Saves checkpoints and summaries to
$STORAGE_ROOT/source_localization/distance_estimator_{feature}_{id}
which requires the environment variable $STORAGE_ROOT to be set.

May be called with:
python -m padertorch.contrib.examples.source_localization.distance_estimator.
train with rir_json=/PATH/TO/RIR-JSON, libri_json=/PATH/TO/LIBRISPEECH-JSON
"""

import os
from pathlib import Path

import numpy as np
from sacred import Experiment as Exp, SETTINGS
from sacred.observers import FileStorageObserver

from data import AudioPreparation, FeatureExtraction, DataProvider
import lazy_dataset
from lazy_dataset.database import JsonDatabase
from model import CRNN, DistanceEstimator
from paderbox.utils.nested import deflatten
from paderbox.transform.module_stft import get_stft_center_frequencies
import padertorch as pt
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Exp('distance_estimator_training')


@ex.config
def config():
    rir_json = None
    libri_json = None
    audio_key = None

    msg = 'You have to specify a path to a JSON describing your RIR-' \
          'database, add "with rir_json=/PATH/TO/RIR-JSON" as suffix'
    assert rir_json is not None, msg
    assert Path(rir_json).exists(), f'"{rir_json}" does not exist.'

    msg = "Please state either a LibriSpeech path or an audio key for use " \
          "with an own database"
    assert not (libri_json is None and audio_key is None), msg
    assert not (libri_json is not None and audio_key is not None), msg
    if libri_json is not None:
        assert Path(libri_json).exists(), f'"{libri_json}" does not exist.'

    feature = "stft"
    check_feature(feature)

    if 'STORAGE_ROOT' not in os.environ:
        raise EnvironmentError(
            'You have to specify an STORAGE_ROOT environmental variable')
    resume = False
    if resume:
        storage_dir = get_storage_dir_resume(feature)
    else:
        storage_dir = pt.io.get_new_storage_dir(
            'source_localization',
            prefix='distance_estimator_' + ("_".join(feature.split())))

    metric = 'mae'

    # Pairs of microphones used for distance estimation
    # (distances are pair-wisely estimated)
    mic_pairs = [(0, 3), (1, 4), (2, 5)]
    d_mic = 0.05

    f_min = 0
    f_max = 8000
    fft_length = 1024
    sample_rate = 16000

    d_min = 0.5
    d_max = 7.9
    quant_step = .1
    num_cls = np.ceil((d_max - d_min) / quant_step) + 1
    output_size = int(num_cls)
    sig_len = 16000  # samples

    input_chs = calc_n_input_chs(feature)
    n_freq_bins = calc_n_freq_bins(f_min, f_max, fft_length, sample_rate)
    batch_size = 16

    # If the memory usage of your model is too large, you can use a smaller
    # batch size and use virtual mini batches to imitate a larger batch size
    virtual_minibatch_size = 1

    pool_config = {
        'pool_type': 'max',
        'kernel_size': (4, 2)
    }
    pool_layers_2d = [None, pool_config] * 3
    n_chs_2d = [64, 64, 64, 64, 64, 64]
    kernel_sized_2d = [(7, 3)] * 6
    n_chs_1d = [512, 512]
    kernel_sized_1d = [(3,)] * 2
    pool_layers_1d = [None] * 2
    hidden_units_gru = 256
    hidden_units_fcn = 256
    cnn_conf = {
        'cnn_2d': {
            'n_chs_input': input_chs,
            'n_chs': n_chs_2d,
            'kernel_sizes': kernel_sized_2d,
            'pool_layers': pool_layers_2d,
        },
        'cnn_1d': {
            'n_chs': n_chs_1d,
            'kernel_sizes': kernel_sized_1d,
            'pool_layers': pool_layers_1d
        },
        'n_freq_bins': n_freq_bins
    }
    gru_conf = {
        'hidden_size': hidden_units_gru,
        'n_layers': 2,
        'dropout_prob': .5
    }
    fcn_conf = {
        'hidden_size': hidden_units_fcn,
        'output_size': output_size,
        'dropout': .5
    }

    crnn_conf = {
        'cnn': cnn_conf,
        'gru': gru_conf,
        'fcn': fcn_conf
    }
    crnn_conf = CRNN.get_config(crnn_conf)
    trainer = deflatten({
        'model.factory': DistanceEstimator,
        'model.net': crnn_conf,
        'model.num_cls': num_cls,
        'model.quant_step': quant_step,
        'model.d_min': d_min,
        'optimizer': {
            'factory': Adam,
            'lr': 3e-4,
            'gradient_clipping': 15.,
            'weight_decay': 3e-5
        },
        'stop_trigger': (500000, 'iteration'),
        'summary_trigger': (1000, 'iteration'),
        'checkpoint_trigger': (10000, 'iteration'),
        'storage_dir': storage_dir,
        'virtual_minibatch_size': virtual_minibatch_size
    })
    Trainer.get_config(trainer)
    trainer['model']['net']['cnn'].pop('n_freq_bins')

    audio_preparator = {
        'factory': AudioPreparation,
        'source_sig_path': libri_json,
        'audio_key': audio_key,
        'signal_length': sig_len
    }

    data_provider_conf = deflatten({
        'feature_extractor.factory': FeatureExtraction,
        'feature_extractor.audio_preparator': audio_preparator,
        'feature_extractor.feature': feature,
        'feature_extractor.fft_length': fft_length,
        'feature_extractor.sample_rate': sample_rate,
        'feature_extractor.d_mic': d_mic,
        'feature_extractor.f_min': f_min,
        'feature_extractor.f_max': f_max,
        'max_workers': 8,
        'batch_size': batch_size,
        'd_min': d_min,
        'quant_step': .1
    })

    data_provider_conf['prefetch_buffer'] = data_provider_conf['batch_size']
    DataProvider.get_config(data_provider_conf)
    ex.observers.append(FileStorageObserver(storage_dir))


def check_feature(feature):
    allowed_features = ["stft", "diffuseness", "ild", "ipd", "phase", "mag"]
    msg = f'Wrong feature specified: "{feature}" not in {allowed_features}'
    assert all([i in allowed_features for i in feature.split()]), msg


def get_storage_dir_resume(feature):
    """Get the latest distance_estimator subdirectory,
     if the training should be resumed."""
    feature = "_".join(feature.split())
    storage_root = Path(os.environ['STORAGE_ROOT']).expanduser().resolve()
    task_dir = storage_root / 'source_localization'
    dirs = list(task_dir.glob('distance_estimator_' + feature + "_*"))
    latest_id = sorted([int(path.name.split('_')[-1]) for path in dirs])[-1]
    return task_dir / f'distance_estimator_{feature}_{latest_id}'


def calc_n_input_chs(feature):
    feature_lut = {"stft": 4,
                   "diffuseness": 1,
                   "phase": 2,
                   "mag": 1,
                   "ild": 1,
                   "ipd": 2}
    return sum([feature_lut[key] for key in feature.split()])


def calc_n_freq_bins(f_min, f_max, fft_length, sample_rate):
    freq_bins = get_stft_center_frequencies(fft_length, sample_rate)
    return len([f for f in freq_bins if f_min <= f <= f_max])


@ex.capture
def prepare_data_iterator(data_provider_conf, rir_json,
                          dataset_key, mic_pairs=None):
    """
    Prepare the datasets for training, validation and testing.

        Args:
            data_provider_conf (dict):
                Create a DataProvider instance with Configurable.
                All attributes of the class DataProvider have to be specified
                in this dict and filled with values.
            rir_json (str):
                Path to the json of the database with the RIRs
            dataset_key (str):
                Key for the dataset that should be prepared
            mic_pairs (list of tuples):
                The pairs of microphones which microphone signals should be
                considered for the feature extraction
                and used for the distance estimation, since distances are
                pair-wisely estimated.
                Can be specified like:
                "[(mic_1_1, mic_1_2), (mic_2_1, mic_2_2) ...]".
    """

    def add_mic_pair(iterable, mic_pair):
        rir_set = {}
        mic_pair_set = {}
        for example_id, example in enumerate(iterable):
            rir_set[f'rir_{example_id}'] = example
            mic_pair_set[f'rir_{example_id}'] = {
                'mic_pair': mic_pair
            }
        rir_set = lazy_dataset.new(rir_set)
        mic_pair_set = lazy_dataset.new(mic_pair_set)
        iterable = rir_set.zip(mic_pair_set)
        iterable = iterable.map(lambda example: {**example[0], **example[1]})
        return iterable

    rir_db = JsonDatabase(rir_json)

    dataset = rir_db.get_dataset(dataset_key)

    if mic_pairs is not None:
        dataset = [add_mic_pair(dataset, mic_pair) for mic_pair in mic_pairs]
        dataset = lazy_dataset.Dataset.concatenate(*dataset)

    if dataset_key == "train":
        data_provider_conf["feature_extractor"]["audio_preparator"][
            "random_speech_samples"] = True
    else:
        data_provider_conf["feature_extractor"]["audio_preparator"][
            "random_speech_samples"] = False
    data_provider = DataProvider.from_config(data_provider_conf)
    iterator = data_provider.prepare_iterable(dataset)
    return iterator


@ex.automain
def train(trainer, metric, resume):
    train_iterator = prepare_data_iterator(dataset_key="train")
    valid_iterator = prepare_data_iterator(dataset_key="dev")
    trainer = Trainer.from_config(trainer)
    trainer.test_run(train_iterator, valid_iterator)
    trainer.register_validation_hook(valid_iterator, metric=metric)
    trainer.train(train_iterator, resume=resume)
