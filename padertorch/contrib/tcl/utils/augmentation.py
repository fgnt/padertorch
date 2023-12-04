from scipy.signal import fftconvolve

import paderbox as pb
import lazy_dataset
import numpy as np
from typing import List, Tuple, Iterable, Union, Dict
from functools import partial
import padertorch as pt


class AugmentationHelper:
    def __init__(self,
                 augmentation_sets: Dict = None,
                 p_augment: float = 0.,
                 p_reverb=None,
                 augmentation_type: Union[str, Iterable] = ('noise', 'music', 'speech'),
                 deterministic: bool = False,
                 augmentation_key='observation',
                 target_key='speech_image'
                 ):
        self.augmentation_dataset = augmentation_sets
        for k, v in self.augmentation_dataset.items():
            if isinstance(v, list):
                self.augmentation_dataset[k] = lazy_dataset.concatenate(*v)
            assert isinstance(self.augmentation_dataset[k], lazy_dataset.Dataset), \
                f'expected dataset of type lazy_dataset.Dataset, got {repr(v)} for dataset {k}'
        self.p_augment = p_augment
        if p_reverb is None:
            self.p_reverb = p_augment
        else:
            self.p_reverb = p_reverb
        self.deterministic = deterministic
        self.augmentation_key = augmentation_key
        self.augment_options = {
            'noise': {},
            'music': {},
            'speech_single': {"snr": (10, 20)},
            'speech': {"snr": (13, 20), 'n_examples': (3, 7)},
        }
        self.augment_types = []
        self.reverb = False
        for aug in augmentation_type:
            if aug == 'reverb':
                self.reverb = True
            else:
                self.augment_types.append(aug)
        self.target_key = target_key

    def __call__(self, example):
        example['audio_data'][self.target_key] = example['audio_data'][self.augmentation_key][None,...]
        if self.deterministic:
            rng = pb.utils.random_utils.str_to_random_state(example['example_id'])
        else:
            rng = np.random.RandomState()
        if self.reverb:
            if rng.uniform() < self.p_reverb:
                example = self.reverb_augmentation(example, rng)
        if rng.uniform() < self.p_augment:
            aug_type = rng.choice(self.augment_types)
            example = self.additive_augmentation(example, rng, aug_type)

        return example

    def replace_signal(self, example, rng):
        augmentation_type = rng.choice(self.augment_types)
        augmentation_example = self.augmentation_dataset[augmentation_type].random_choice(1, rng_state=rng,)[0]
        augmentation_data = pb.io.load_audio(augmentation_example['audio_path'][self.augmentation_key])
        example['audio_data'][self.target_key] = augmentation_data
        return example

    def pad_and_sum(self, audio_data: List, rng: np.random.RandomState = np.random):
        max_len = max([len(ex) for ex in audio_data])
        output_array = np.zeros(max_len)
        for ex in audio_data:
            audio_len = len(ex)
            if max_len - audio_len > 0:
                offset = rng.randint(0, max_len-audio_len)
            else:
                offset = 0
            output_array[offset:offset+audio_len] += ex
        return output_array


    def get_scaling_factor(self, observation, augmentation, snr):
        observation_power = np.mean(observation ** 2, keepdims=True)
        augmentation_power = np.mean(augmentation ** 2, keepdims=True)

        current_snr = 10 * np.log10(observation_power / augmentation_power)
        factor = 10 ** (-(snr - current_snr) / 20)
        if isinstance(factor, np.ndarray):
            if factor.ndim > 1:
                factor = factor[0]
        return factor

    def reverb_augmentation(self, example, rng: np.random.RandomState):
        rir = self.augmentation_dataset['reverb'].random_choice(1, rng_state=rng)[0]
        try:
            rir = pb.io.load(rir['audio_path']['rir'])
            position = rng.randint(0, len(rir))
            channel = rng.randint(0, rir[position].shape[0])
            rir = rir[position][channel, :]
        except KeyError:
            print('Found no key "rir" under "audio_path" of augmentation dataset. Make sure to set reverberation dataset as'
                  'last dataset for now!')
        example['audio_data']['rir'] = rir
        example['audio_data'][self.augmentation_key] = fftconvolve(rir, example['audio_data'][self.augmentation_key])
        return example

    def additive_augmentation(self, example, rng: np.random.RandomState, augmentation_type, snr: Tuple = (0, 15), n_examples: Union[int, Tuple] = 1):
        #  Hard-coded for MUSAN data, for now
        if isinstance(n_examples, tuple):
            n_examples = rng.randint(*n_examples)

        augmentation_example = self.augmentation_dataset[augmentation_type].random_choice(n_examples, rng_state=rng,)
        if n_examples > 1:
            augmentation_example = pt.data.utils.collate_fn(augmentation_example.batch(n_examples)[0])
            augmentation_example = pb.io.recursive_load_audio(augmentation_example['audio_path']['observation'])
            augmentation_example = self.pad_and_sum(augmentation_example, rng=rng)
        else:
            augmentation_example = pb.io.recursive_load_audio(augmentation_example[0]['audio_path']['observation'])
        snr = rng.uniform(*snr)
        scale = self.get_scaling_factor(example['audio_data'][self.augmentation_key], augmentation_example, snr)
        if scale.ndim > 1:
            scale = scale[0, :]
        augmentation_example *= scale
        example_len = len(example['audio_data'][self.augmentation_key])
        if len(augmentation_example) >= example_len:
            example['audio_data'][self.augmentation_key] = example['audio_data'][self.augmentation_key] + augmentation_example[:example_len]
        else:
            offset = rng.randint(0, example_len-len(augmentation_example))
            example['audio_data'][self.augmentation_key][offset:offset+len(augmentation_example)] = \
                example['audio_data'][self.augmentation_key][offset:offset+len(augmentation_example)] + augmentation_example[:example_len]
        return example
