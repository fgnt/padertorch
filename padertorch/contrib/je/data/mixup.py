from lazy_dataset import Dataset, FilterException
import numpy as np
import numbers


class MixUpDataset(Dataset):
    def __init__(self, input_dataset, mixin_dataset, p, mixin_keys=None):
        """
        Combines examples from input_dataset and mixin_dataset into tuples.

        Args:
            input_dataset: lazy dataset providing example dict with key audio_length.
            mixin_dataset: lazy dataset providing example dict with key audio_length.
            p: list of probabilities of the number of mixture components.
        """
        self.input_dataset = input_dataset
        self.mixin_dataset = mixin_dataset
        self.p = p
        if mixin_keys is None:
            self.mixin_keys = sorted(self.mixin_dataset.keys())
        elif isinstance(mixin_keys, (list, tuple, set)):
            self.mixin_keys = sorted(mixin_keys)
        elif isinstance(mixin_keys, dict):
            self.mixin_keys = {
                key: sorted(keys) for key, keys in mixin_keys.items()
            }

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        for key in self.keys():
            yield self[key]

    def keys(self):
        return self.input_dataset.keys()

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            mixin_dataset=self.mixin_dataset.copy(freeze=freeze),
            p=self.p,
            mixin_keys=self.mixin_keys,
        )

    @property
    def indexable(self):
        return True

    def __getitem__(self, item):
        n_components = np.random.choice(len(self.p), p=self.p) + 1
        if isinstance(item, str):
            key = item
        elif isinstance(item, numbers.Integral):
            key = self.keys()[item]
        else:
            return super().__getitem__(item)
        if isinstance(self.mixin_keys, list):
            mixin_keys = self.mixin_keys
        elif isinstance(self.mixin_keys, dict):
            mixin_keys = self.mixin_keys[key]
        n_components = min(n_components, len(mixin_keys))
        if n_components > 1:
            mixin_idx = np.random.choice(len(mixin_keys), n_components-1, replace=False)
        else:
            mixin_idx = []
        components = [self.input_dataset[item]] + [self.mixin_dataset[mixin_keys[i]] for i in mixin_idx]
        return components


class SuperposeEvents:
    def __init__(self, sample_rate, min_overlap=1, max_length=None):
        self.sample_rate = sample_rate
        self.min_overlap = min_overlap
        self.max_length = max_length

    def __call__(self, components):
        assert len(components) > 0
        start_indices = [0]
        stop_indices = [components[0]['audio_data'].shape[-1]]
        for comp in components[1:]:
            l = comp['audio_data'].shape[-1]
            min_start = max(
                -int(l*(1-self.min_overlap)),
                max(stop_indices) - self.max_length * self.sample_rate
            )
            max_start = min(
                components[0]['audio_data'].shape[-1] - int(np.ceil(self.min_overlap*l)),
                min(start_indices) + self.max_length * self.sample_rate - l
            )
            if max_start < min_start:
                raise FilterException
            start_indices.append(
                int(min_start + np.random.rand() * (max_start - min_start + 1))
            )
            stop_indices.append(start_indices[-1] + l)
        start_indices = np.array(start_indices)
        stop_indices = np.array(stop_indices)
        stop_indices -= start_indices.min()
        start_indices -= start_indices.min()

        mixed_audio = np.zeros((*components[0]['audio_data'].shape[:-1], stop_indices.max()))
        for comp, start, stop in zip(components, start_indices, stop_indices):
            mixed_audio[..., start:stop] += comp['audio_data']
        mix = {
            'example_id': '+'.join([comp['example_id'] for comp in components]),
            'audio_data': mixed_audio,
            'audio_length': mixed_audio.shape[-1] / self.sample_rate,
            'events': [event for comp in components for event in comp['events']],
            'dataset': '+'.join([comp['dataset'] for comp in components])
        }
        if "events_start_times" in components[0]:
            mix["events_start_times"] = [
                mixin_start_sample/self.sample_rate + event_start_time
                for mixin_start_sample, comp in zip(start_indices, components)
                for event_start_time in comp['events_start_times']
            ]
        if "events_stop_times" in components[0]:
            mix["events_stop_times"] = [
                mixin_start_sample/self.sample_rate + event_stop_time
                for mixin_start_sample, comp in zip(start_indices, components)
                for event_stop_time in comp['events_stop_times']
            ]
        for key in components[0].keys():
            if key not in mix:
                mix[key] = [comp[key] for comp in components]
        return mix
