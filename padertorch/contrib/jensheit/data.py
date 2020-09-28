from copy import deepcopy
from functools import partial
from random import shuffle
from typing import Dict
from typing import List

import numpy as np
from dataclasses import dataclass
from dataclasses import field
from paderbox.array import segment_axis
from paderbox.transform import STFT
from paderbox.utils.mapping import Dispatcher
from padercontrib.database.iterator import AudioReader
from padercontrib.database.keys import *
from padertorch.contrib.jensheit import Parameterized, dict_func
from padertorch.contrib.jensheit.batch import Padder
from padertorch.contrib.jensheit.mask_estimator_example.modul import MaskKeys as M_K
from pb_bss.extraction.mask_module import biased_binary_mask
from scipy import signal
from collections import Generator

WINDOW_MAP = Dispatcher(
    blackman=signal.blackman,
    hamming=signal.hamming,
    hann=signal.hann
)


class MaskTransformer(Parameterized):
    @dataclass
    class opts:
        stft: Dict = dict_func({
            'factory': STFT,
            'shift': 256,
            'size': 1024,
        })
        low_cut: int = 5
        high_cut: int = -5

    def __init__(self, stft, **kwargs):
        super().__init__(**kwargs)
        self.stft = stft

    def inverse(self, signal):
        return self.stft.inverse(signal)

    def __call__(self, example):
        if isinstance(example, (list, tuple, Generator)):
            return [self.transform(ex) for ex in example]
        else:
            return self.transform(example)

    def transform(self, example):
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
    is_training = False
    @dataclass
    class opts:
        reference_channel: int = 0
        collate: Dict = dict_func(dict(
            factory=Padder,
            to_torch=False,
            sort_by_key=NUM_SAMPLES,
            padding=False,
            padding_keys=None
        ))
        audio_keys: List = field(default_factory=lambda: [OBSERVATION])
        segment_channels: bool = False
        shuffle: bool = True
        batch_size: int = 1
        batch_size_eval: int = 5
        num_workers: int = 4
        buffer_size: int = 20

        backend: str = 't'
        drop_last: bool = False
        time_segments: int = None
        time_segments_random_offset: bool = False

    def __init__(self, database, collate, transform=None, **kwargs):
        self.database = database
        self.transform = transform if transform is not None else lambda x: x
        self.collate = collate
        super().__init__(**kwargs)

    def to_train_structure(self, example):
        """Function to be mapped on an iterator."""
        out_dict = example[AUDIO_DATA]
        out_dict['audio_keys'] = list(example[AUDIO_DATA].keys())
        if SENSOR_POSITION in example:
            out_dict[SENSOR_POSITION] = example[SENSOR_POSITION]
        out_dict[EXAMPLE_ID] = example[EXAMPLE_ID]
        out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        if isinstance(example[NUM_SAMPLES], dict):
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES][OBSERVATION]
        else:
            out_dict[NUM_SAMPLES] = example[NUM_SAMPLES]
        return out_dict

    def to_eval_structure(self, example):
        """Function to be mapped on an iterator."""
        return self.to_train_structure(example)

    def to_predict_structure(self, example):
        """Function to be mapped on an iterator."""
        return self.to_train_structure(example)

    def read_audio(self, example):
        """Function to be mapped on an iterator."""
        return AudioReader(
            audio_keys=self.opts.audio_keys,
            read_fn=self.database.read_fn
        )(example)

    def segment(self, example, exclude_keys=None):
        if exclude_keys is None:
            exclude_keys = []
        elif isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        segment_len = shift = self.opts.time_segments
        num_samples = example[NUM_SAMPLES]
        audio_keys = [key for key in example['audio_keys']
                      if key not in exclude_keys]

        offset = 0
        if self.opts.time_segments_random_offset:
            max_offset = num_samples % segment_len
            if max_offset > 0:
                offset = np.random.randint(0, max_offset)

        for key in audio_keys:
            example[key] = segment_axis(
                example[key][..., offset:num_samples], segment_len,
                shift=shift, axis=-1, end='cut'
            )
        lengths = ([example[key].shape[-2] for key in audio_keys])
        assert lengths.count(lengths[-2]) == len(lengths), {
            audio_keys[idx]: leng for idx, leng in enumerate(lengths)}
        length = lengths[0]
        if length == 0:
            from lazy_dataset.core import FilterException
            print('was to short')
            raise FilterException
        out_list = list()
        example[NUM_SAMPLES] = self.opts.time_segments
        for idx in range(length):
            new_example = deepcopy(example)
            for key in audio_keys:
                new_example[key] = new_example[key][..., idx, :]
            out_list.append(new_example)
        shuffle(out_list)
        return out_list

    def segment_channels(self, example, exclude_keys=None):
        if not isinstance(example, (tuple, list)):
            example = [example]
        if exclude_keys is None:
            exclude_keys = []
        elif isinstance(exclude_keys, str):
            exclude_keys = [exclude_keys]
        from copy import deepcopy
        out_list = list()
        num_channels = example[0][OBSERVATION].shape[0]
        for ex in example:
            audio_keys = [key for key, value in ex.items()
                          if isinstance(value, np.ndarray)
                          if key not in exclude_keys]
            for idx in range(num_channels):
                new_example = deepcopy(ex)
                for key in audio_keys:
                    signal = new_example[key]
                    if signal.shape[0] < num_channels:
                        signal = signal.swapaxes(0, 1)
                    assert signal.shape[0] == num_channels, signal.shape
                    new_example[key] = signal[idx, None]
                out_list.append(new_example)
        shuffle(out_list)
        return out_list

    def get_map_iterator(self, iterator, batch_size=None,
                         prefetch=True, unbatch=False, segment_channels=False):
        iterator = iterator.map(self.transform)

        if segment_channels:
            iterator = iterator.map(segment_channels)
            unbatch = True
        if prefetch:
            iterator = iterator.prefetch(
                self.opts.num_workers, self.opts.buffer_size,
                self.opts.backend, catch_filter_exception=True
            )
        if unbatch:
            iterator = iterator.unbatch()
        if batch_size is not None:
            iterator = iterator.batch(batch_size, self.opts.drop_last)
            iterator = iterator.map(self.collate)
        else:
            if self.opts.batch_size is not None:
                iterator = iterator.batch(self.opts.batch_size,
                                          self.opts.drop_last)
                iterator = iterator.map(self.collate)
        return iterator

    def train(self):
        self._train = True

    def get_train_iterator(self, time_segment=None):
        self.is_training = True
        iterator = self.database.get_dataset_train()
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)
        exclude_keys = None
        iterator = iterator.map(self.to_train_structure)


        unbatch = False
        if self.opts.shuffle:
            iterator = iterator.shuffle(reshuffle=True)
        if self.opts.time_segments is not None or time_segment is not None:
            assert not (self.opts.time_segments and time_segment)
            iterator = iterator.map(
                partial(self.segment, exclude_keys=exclude_keys))
            unbatch = True

        if self.opts.segment_channels:
            segment_channels = partial(self.segment_channels,
                                       exclude_keys=exclude_keys)
        else:
            segment_channels = False
        return self.get_map_iterator(iterator, self.opts.batch_size,
                                     segment_channels=segment_channels,
                                     unbatch=unbatch)

    def get_eval_iterator(self, num_examples=-1):
        self.is_training = False

        iterator = self.database.get_dataset_validation()
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)

        iterator = iterator.map(self.to_eval_structure)[:num_examples]
        return self.get_map_iterator(iterator, self.opts.batch_size_eval)

    def get_predict_iterator(self, dataset=None, num_examples=None,
                             iterable_apply_fn=None,
                             filter_fn=None):
        self.is_training = False
        if dataset is None:
            iterator = self.database.get_dataset_test()
        else:
            iterator = self.database.get_dataset(dataset)
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)
        iterator = iterator.map(self.to_predict_structure)
        if num_examples is not None:
            iterator = iterator[:num_examples]
        if iterable_apply_fn is not None:
            iterator = iterator.apply(iterable_apply_fn)
        iterator = self.get_map_iterator(iterator, prefetch=False)
        if filter_fn is not None:
            iterator = iterator.filter(filter_fn)
        return iterator
