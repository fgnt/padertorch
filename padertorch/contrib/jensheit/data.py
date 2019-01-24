from typing import Dict
from typing import List

import numpy as np
from dataclasses import dataclass, asdict
from dataclasses import field
from numpy import random
from padertorch.contrib.jensheit import Parameterized, dict_func
from scipy import signal

from paderbox.database.chime import Chime3
from paderbox.database.iterator import AudioReader
from paderbox.database.keys import *
from paderbox.speech_enhancement.mask_module import biased_binary_mask
from paderbox.transform import stft, istft
from paderbox.utils.mapping import Dispatcher
from padertorch.data.utils import Padder
from padertorch.modules.mask_estimator import MaskKeys as M_K
from padertorch.data.transforms import Compose

WINDOW_MAP = Dispatcher(
    blackman=signal.blackman,
    hamming=signal.hamming,
    hann=signal.hann
)


class STFT(Parameterized):
    @dataclass
    class Options:
        size: int = 1024
        shift: int = 256
        window: str = 'blackman'
        window_length: int = None
        fading: bool = True
        symmetric_window: bool = False

    def __call__(self, signal):
        return stft(signal, pad=True, **dict(
            asdict(self.opts), **dict(window=WINDOW_MAP[self.opts.window])))

    def inverse(self, signal):
        return istft(signal, **dict(asdict(self.opts),
                                    **dict(
                                        window=WINDOW_MAP[self.opts.window])))


class MaskTransformer(Parameterized):
    @dataclass
    class Options:
        stft: Dict = dict_func({
            'cls': STFT,
            'kwargs': {}
        })
        low_cut: int = 5
        high_cut: int = -5

    def __init__(self, stft, **kwargs):
        super().__init__(**kwargs)
        self.stft = stft

    def __call__(self, example):
        def maybe_add_channel(signal):
            if signal.ndim == 1:
                return np.expand_dims(signal, axis=0)
            elif signal.ndim == 2:
                return signal
            else:
                raise ValueError('Either the signal has ndim 1 or 2',
                                 signal.shape)

        example[M_K.OBSERVATION_STFT] = self.stft(maybe_add_channel(
            example[OBSERVATION]))
        example[M_K.OBSERVATION_ABS] = np.abs(example[M_K.OBSERVATION_STFT]
                                              ).astype(np.float32)
        example[NUM_FRAMES] = example[M_K.OBSERVATION_STFT].shape[-2]
        if SPEECH_IMAGE in example and NOISE_IMAGE in example:
            speech = self.stft(maybe_add_channel(example[SPEECH_IMAGE]))
            noise = self.stft(maybe_add_channel(example[NOISE_IMAGE]))
            target_mask, noise_mask = biased_binary_mask(
                np.stack([speech, noise], axis=0),
                low_cut=self.opts.low_cut,
                high_cut=self.opts.high_cut if self.opts.high_cut >= 0
                else speech.shape[-1] + self.opts.high_cut
            )
            example[M_K.SPEECH_MASK_TARGET] = target_mask.astype(np.float32)
            example[M_K.NOISE_MASK_TARGET] = noise_mask.astype(np.float32)
        return example


class SequenceProvider(Parameterized):
    @dataclass
    class Options:
        database: Dict = dict_func({
            'cls': Chime3,
            'kwargs': {}
        })
        audio_keys: List = field(default_factory=lambda: [OBSERVATION])
        shuffle: bool = True
        batch_size: int = 1
        num_workers: int = 4
        buffer_size: int = 20
        backend: str = 't'
        drop_last: bool = False
        collate: Dict = dict_func({
            'cls': Padder,
            'kwarg': dict()
        })
        transformer: Dict = dict_func({})

    def __init__(self, database, transformer, collate, **kwargs):
        self.database = database
        self.transformer = transformer
        self.collate = collate
        super().__init__(**kwargs)

    def to_train_structure(self, example):
        """Function to be mapped on an iterator."""
        out_dict = example[AUDIO_DATA]
        out_dict[EXAMPLE_ID] = example[EXAMPLE_ID]
        out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        return out_dict

    def to_eval_structure(self, example):
        """Function to be mapped on an iterator."""
        return self.to_train_structure(example)

    def to_predict_structure(self, example):
        """Function to be mapped on an iterator."""
        out_dict = dict()
        out_dict[OBSERVATION] = example[AUDIO_DATA][OBSERVATION]
        out_dict[EXAMPLE_ID] = example[EXAMPLE_ID]
        out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        return out_dict

    def read_audio(self, example):
        """Function to be mapped on an iterator."""
        return AudioReader(
            audio_keys=self.opts.audio_keys,
            read_fn=self.database.read_fn
        )(example)

    def get_iterator(self, datasets, shuffle=False):
        iterator = self.database.get_iterator_by_names(datasets)
        if shuffle:
            iterator = iterator.shuffle()
        return iterator.map(Compose(
            self.read_audio, self.database.add_num_samples
        ))

    def transformer(self, example):
        return example

    def get_train_iterator(self, filter_fn=lambda x: True):
        iterator = self.get_iterator(
            self.database.datasets_train, shuffle=self.opts.shuffle).map(
            Compose(self.to_train_structure, self.transformer)
        )
        return iterator.batch(self.opts.batch_size, self.opts.drop_last) \
            .map(self.collate).prefetch(self.opts.num_workers,
                                        self.opts.buffer_size,
                                        self.opts.backend)

    def get_eval_iterator(self, num_examples=-1, transform_fn=lambda x: x,
                          filter_fn=lambda x: True):
        iterator = self.get_iterator(
            self.database.datasets_eval, shuffle=False).map(
            Compose(self.to_eval_structure, self.transformer)
        )
        return iterator.batch(self.opts.batch_size, self.opts.drop_last) \
            .map(self.collate)[:num_examples].prefetch(
            self.opts.num_workers, self.opts.buffer_size, self.opts.backend
        )


class MaskProvider(SequenceProvider):
    @dataclass
    class Options(SequenceProvider.Options):
        reference_channel: int = 0
        multichannel: bool = False
        audio_keys: List = field(default_factory=lambda: [
            OBSERVATION, SPEECH_IMAGE, NOISE_IMAGE
        ])
        transformer: Dict = dict_func({
            'cls': MaskTransformer,
            'kwargs': {}
        })
        collate: Dict = dict_func({
            'cls': Padder,
            'kwargs': dict(to_torch=False,
                           sort_by_key=NUM_SAMPLES,
                           padding=False,
                           padding_keys=None
                           )
        })
        num_channels = None
        choose_channel = None

    def get_iterator(self, datasets, shuffle=False):
        iterator = super().get_iterator(datasets, shuffle)
        if not self.opts.multichannel:
            iterator = iterator.map(self.remove_channels)
        return iterator

    def remove_channels(self, example):
        if self.opts.num_channels is None:
            num_channels = len(example[AUDIO_PATH][OBSERVATION])
        else:
            num_channels = self.opts.num_channels
        if self.opts.choose_channel is None:
            channel = random.randint(0, num_channels)
        else:
            channel = self.opts.choose_channel
        example[AUDIO_PATH] = {key: example[AUDIO_PATH][key][channel]
                               for key in self.opts.audio_keys}
        return example

    def to_train_structure(self, example):
        out_dict = super().to_train_structure(example)
        if isinstance(example[NUM_SAMPLES], dict):
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES][OBSERVATION]
        else:
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        return out_dict

    def to_eval_structure(self, example):
        out_dict = super().to_eval_structure(example)
        if isinstance(example[NUM_SAMPLES], dict):
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES][OBSERVATION]
        else:
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        return out_dict
