from functools import partial
from random import shuffle
from typing import Dict
from typing import List

import numpy as np
import torch
from dataclasses import dataclass
from dataclasses import field
from scipy import signal

from padercontrib.database.iterator import AudioReader
from padercontrib.database.keys import *
from pb_bss.extraction.mask_module import biased_binary_mask
from paderbox.transform import stft, istft
from paderbox.transform.module_stft import _samples_to_stft_frames
from paderbox.transform.module_stft import _stft_frames_to_samples
from paderbox.utils.mapping import Dispatcher
from padertorch.configurable import Configurable
from padertorch.contrib.jensheit import Parameterized, dict_func
from padertorch.data.utils import pad_tensor, collate_fn
from padertorch.modules.mask_estimator import MaskKeys as M_K

WINDOW_MAP = Dispatcher(
    blackman=signal.blackman,
    hamming=signal.hamming,
    hann=signal.hann
)


class Padder(Configurable):
    def __init__(
            self,
            to_torch: bool = True,
            sort_by_key: str = None,
            padding: bool = True,
            padding_keys: list = None
    ):
        """

        :param to_torch: if true converts numpy arrays to torch.Tensor
            if they are not strings or complex
        :param sort_by_key: sort the batch by a key from the batch
            packed_sequence needs sorted batch with decreasing sequence_length
        :param padding: if False only collates the batch,
            if True all numpy arrays with one variable dim size are padded
        :param padding_keys: list of keys, if no keys are specified all
            keys from the batch are used
        """
        assert not to_torch ^ (padding and to_torch)
        self.to_torch = to_torch
        self.padding = padding
        self.padding_keys = padding_keys
        self.sort_by_key = sort_by_key

    def pad_batch(self, batch):
        if isinstance(batch[0], np.ndarray):
            if batch[0].ndim > 0:
                dims = np.array(
                    [[idx for idx in array.shape] for array in batch]).T
                axis = [idx for idx, dim in enumerate(dims)
                        if not all(dim == dim[0])]

                assert len(axis) in [0, 1], (
                    f'only one axis is allowed to differ, '
                    f'axis={axis} and dims={dims}'
                )
                if len(axis) == 1:
                    axis = axis[0]
                    pad = max(dims[axis])
                    array = np.stack([pad_tensor(vec, pad, axis)
                                      for vec in batch], axis=0)
                else:
                    array = np.stack(batch, axis=0)
                complex_dtypes = [np.complex64, np.complex128]
                if self.to_torch and not array.dtype.kind in {'U', 'S'} \
                        and not array.dtype in complex_dtypes:
                    return torch.from_numpy(array)
                else:
                    return array
            else:
                return np.array(batch)
        elif isinstance(batch[0], int):
            return np.array(batch)
        else:
            return batch

    def sort(self, batch):
        return sorted(batch, key=lambda x: x[self.sort_by_key], reverse=True)

    def __call__(self, unsorted_batch):
        # assumes batch to be a list of dicts
        # ToDo: do we automatically sort by sequence length?

        if self.sort_by_key:
            batch = self.sort(unsorted_batch)
        else:
            batch = unsorted_batch

        nested_batch = collate_fn(batch)

        if self.padding:
            if self.padding_keys is None:
                padding_keys = nested_batch.keys()
            else:
                assert len(self.padding_keys) > 0, \
                    'Empty padding key list was provided default should be None'
                padding_keys = self.padding_keys

            def nested_padding(value, key):
                if isinstance(value, dict):
                    return {k: nested_padding(v, k) for k, v in value.items()}
                else:
                    if key in padding_keys:
                        return self.pad_batch(value)
                    else:
                        return value

            return {key: nested_padding(value, key) for key, value in
                    nested_batch.items()}
        else:
            assert self.padding_keys is None or len(self.padding_keys) == 0, (
                'Padding keys have to be None or empty if padding is set to '
                'False, but they are:', self.padding_keys
            )
            return nested_batch


class STFT:
    def __init__(self, size: int = 512, shift: int = 160,
                 window: str = 'blackman', window_length: int = 400,
                 fading: bool = True,  pad: bool = True,
                 symmetric_window: bool = False):
        self.size = size
        self.shift = shift
        self.window = window
        self.window_length = window_length
        self.fading = fading
        self.pad = pad
        self.symmetric_window = symmetric_window

    def __call__(self, signal):
        return stft(signal, pad=self.pad, size=self.size, shift=self.shift,
                    window_length=self.window_length, fading=self.fading,
                    symmetric_window=self.symmetric_window,
                    window=WINDOW_MAP[self.window])

    def inverse(self, signal):
        return istft(signal, size=self.size, shift=self.shift,
                     window_length=self.window_length, fading=self.fading,
                     symmetric_window=self.symmetric_window,
                     window=WINDOW_MAP[self.window])

    def frames_to_samples(self, nframes):
        return _stft_frames_to_samples(nframes, self.window_length, self.shift)

    def samples_to_frames(self, nsamples):
        return _samples_to_stft_frames(nsamples, self.window_length,
                                       self.shift)


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
        num_channels: int = 6
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
        from paderbox.utils.numpy_utils import segment_axis_v2
        from copy import deepcopy
        segment_len = shift = self.opts.time_segments
        num_samples = example[NUM_SAMPLES]
        audio_keys = [key for key in example['audio_keys']
                      if not key in exclude_keys]
        for key in audio_keys:
            example[key] = segment_axis_v2(
                example[key][..., :num_samples], segment_len,
                shift=shift, axis=-1, end='cut')
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
        for ex in example:
            audio_keys = [key for key, value in ex.items()
                          if isinstance(value, np.ndarray)
                          if not key in exclude_keys]
            for idx in range(self.opts.num_channels):
                new_example = deepcopy(ex)
                for key in audio_keys:
                    signal = new_example[key]
                    if signal.shape[0] < self.opts.num_channels:
                        signal = signal.swapaxes(0, 1)
                    assert signal.shape[
                               0] == self.opts.num_channels, signal.shape
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

    def get_train_iterator(self, time_segment=None):

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
        if not self.opts.multichannel:
            segment_channels = partial(self.segment_channels,
                                       exclude_keys=exclude_keys)
        else:
            segment_channels = False
        return self.get_map_iterator(iterator, self.opts.batch_size,
                                     segment_channels=segment_channels,
                                     unbatch=unbatch)

    def get_eval_iterator(self, num_examples=-1, transform_fn=lambda x: x,
                          filter_fn=lambda x: True):

        iterator = self.database.get_dataset_validation()
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)

        iterator = iterator.map(self.to_eval_structure)[:num_examples]
        return self.get_map_iterator(iterator, self.opts.batch_size_eval)

    def get_predict_iterator(self, num_examples=-1,
                             dataset=None,
                             iterable_apply_fn=None,
                             filter_fn=lambda x: True):
        if dataset is None:
            iterator = self.database.get_dataset_test()
        else:
            iterator = self.database.get_dataset(dataset)
        iterator = iterator.map(self.read_audio) \
            .map(self.database.add_num_samples)

        iterator = iterator.map(self.to_predict_structure)[:num_examples]
        if iterable_apply_fn is not None:
            iterator = iterator.apply(iterable_apply_fn)
        iterator = self.get_map_iterator(iterator, prefetch=False)
        if filter_fn is not None:
            iterator = iterator.filter(filter_fn)
        return iterator
