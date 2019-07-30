from typing import Dict
from typing import List

import numpy as np
from dataclasses import dataclass, asdict
from dataclasses import field
from scipy import signal

from paderbox.database.iterator import AudioReader
from paderbox.database.keys import *
from paderbox.speech_enhancement.mask_module import biased_binary_mask
from paderbox.transform import stft, istft
from paderbox.utils.mapping import Dispatcher
from padertorch.contrib.jensheit import Parameterized, dict_func
from padertorch.data.utils import Padder
from padertorch.modules.mask_estimator import MaskKeys as M_K


WINDOW_MAP = Dispatcher(
    blackman=signal.blackman,
    hamming=signal.hamming,
    hann=signal.hann
)


class STFT(Parameterized):
    @dataclass
    class opts:
        size: int = 1024
        shift: int = 256
        window: str = 'blackman'
        window_length: int = None
        fading: bool = True
        symmetric_window: bool = False
        pad: bool = True

    def __call__(self, signal):
        return stft(signal, **dict(
            asdict(self.opts), **dict(window=WINDOW_MAP[self.opts.window])))

    def inverse(self, signal):
        opts = {key: value for key, value in asdict(self.opts).items()
                if not key == 'pad'}
        return istft(signal, **dict(
            opts, **dict(window=WINDOW_MAP[self.opts.window])))


class MaskTransformer(Parameterized):
    @dataclass
    class opts:
        stft: Dict = dict_func({
            'factory': STFT,
        })
        low_cut: int = 5
        high_cut: int = -5

    def __init__(self, stft, **kwargs):
        super().__init__(**kwargs)
        self.stft = stft

    def inverse(self, signal):
        return self.stft.inverse(signal)

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
        shuffle: bool = True
        batch_size: int = 1
        batch_size_eval: int = 5
        num_workers: int = 4
        buffer_size: int = 20
        multichannel: bool = True
        backend: str = 't'
        drop_last: bool = False
        time_segments: int = None

    def __init__(self, database, collate, transform=None, **kwargs):
        self.database = database
        self.transform = transform if transform is not None else lambda x: x
        self.collate = collate
        super().__init__(**kwargs)

    def to_train_structure(self, example):
        """Function to be mapped on an iterator."""
        out_dict = example[AUDIO_DATA]
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

    def segment(self, example):
        from paderbox.utils.numpy_utils import segment_axis_v2
        from copy import deepcopy
        segment_len = shift = self.opts.time_segments
        num_samples = example[NUM_SAMPLES]
        for key in self.opts.audio_keys:
            example[key]= segment_axis_v2(
                example[key][..., :num_samples], segment_len,
                shift=shift, axis=-1, end='cut')
        lengths = ([example[key].shape[-2] for key in self.opts.audio_keys])
        assert lengths.count(lengths[0]) == len(lengths), {
            self.opts.audio_keys[idx]: leng for idx, leng in enumerate(lengths)}
        length = lengths[0]
        out_list = list()
        example[NUM_SAMPLES] = self.opts.time_segments
        for idx in range(length):
            new_example = deepcopy(example)
            for key in self.opts.audio_keys:
                new_example[key] = new_example[key][..., idx, :]
            out_list.append(new_example)
        return out_list

    def get_map_iterator(self, iterator, batch_size,
                         training=False, prefetch=True):
        iterator = iterator.map(self.transform)
        if prefetch:
            iterator = iterator.prefetch(
                self.opts.num_workers,self.opts.buffer_size,
                self.opts.backend, catch_filter_exception=True
            )
        if training and self.opts.time_segments is not None:
            iterator = iterator.unbatch()
        if batch_size is not None:
            iterator = iterator.batch(batch_size, self.opts.drop_last)
            iterator = iterator.map(self.collate)
        return iterator

    def get_train_iterator(self, time_segment=None):
        iterator = self.database.get_iterator_by_names(
            self.database.datasets_train)
        iterator = iterator.map(self.read_audio)\
            .map(self.database.add_num_samples)\
            .map(self.to_train_structure)
        if self.opts.shuffle:
            iterator = iterator.shuffle()
        if self.opts.time_segments is not None or time_segment is not None:
            assert not (self.opts.time_segments and time_segment)
            iterator = iterator.map(self.segment)
        return self.get_map_iterator(iterator, self.opts.batch_size,
                                     training=True)

    def get_eval_iterator(self, num_examples=-1, transform_fn=lambda x: x,
                          filter_fn=lambda x: True):
        iterator = self.database.get_iterator_by_names(
            self.database.datasets_eval)
        iterator = iterator.map(self.read_audio)\
            .map(self.database.add_num_samples)\
            .map(self.to_eval_structure)[:num_examples]
        return self.get_map_iterator(iterator, self.opts.batch_size_eval)

    def get_predict_iterator(self, num_examples=-1,
                             dataset=None,
                             iterable_apply_fn=None,
                             filter_fn=lambda x: True):
        if dataset is None:
            dataset = self.database.datasets_test
        iterator = self.database.get_iterator_by_names(dataset)
        iterator = iterator.map(self.read_audio)\
            .map(self.database.add_num_samples)\
            .map(self.to_eval_structure)[:num_examples]
        if iterable_apply_fn is not None:
            iterator = iterator.apply(iterable_apply_fn)
            prefetch = False
        else:
            prefetch = True
        iterator = self.get_map_iterator(iterator, prefetch=prefetch)
        return iterator.filter(filter_fn)

