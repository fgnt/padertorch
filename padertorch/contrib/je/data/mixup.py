from lazy_dataset import Dataset, FilterException
import numpy as np
import numbers


class MixUpDataset(Dataset):
    """
    >>> ds = MixUpDataset(range(10), SampleMixupComponents((.0,1.)), (lambda x: x), buffer_size=2)
    >>> list(ds)
    """
    def __init__(self, input_dataset, sample_fn, mixup_fn, buffer_size=100):
        """
        Combines examples from input_dataset and mixin_dataset into tuples.

        Args:
            input_dataset: lazy dataset providing example dict with key audio_length.
            sample_fn: sample_fn(buffer) returning a list of examples from buffer for mixup.
        """
        self.input_dataset = input_dataset
        self.buffer = []
        self.buffer_size = buffer_size
        self.sample_fn = sample_fn
        self.mixup_fn = mixup_fn

    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        for example in self.input_dataset:
            self.buffer.append(example)
            if len(self.buffer) > self.buffer_size:
                examples = self.sample_fn(self.buffer)
                if len(examples) == 1:
                    yield examples[0]
                elif len(examples) > 1:
                    yield self.mixup_fn(examples)
                else:
                    raise ValueError('sample_fn has to return at least one example')
                self.buffer.pop(0)
            else:
                yield example

    def copy(self, freeze=False):
        return self.__class__(
            input_dataset=self.input_dataset.copy(freeze=freeze),
            sample_fn=self.sample_fn,
            mixup_fn=self.mixup_fn,
            buffer_size=self.buffer_size,
        )

    @property
    def indexable(self):
        return False


class SampleMixupComponents:
    """
    >>> sample_fn = SampleMixupComponents((0,1.))
    >>> buffer = list(range(10))
    >>> sample_fn(buffer)
    >>> buffer
    """
    def __init__(self, mixup_prob):
        self.mixup_prob = mixup_prob

    def __call__(self, buffer):
        examples = [buffer[-1]]
        num_mixins = np.random.choice(len(self.mixup_prob), p=self.mixup_prob)
        num_mixins = min(num_mixins, len(buffer) - 1)
        if num_mixins > 0:
            idx = np.random.choice(len(buffer)-1, num_mixins, replace=False)
            examples.extend(buffer[i] for i in idx)
        return examples


class SuperposeEvents:
    """
    >>> mixup_fn = SuperposeEvents(min_overlap=0.5)
    >>> example1 = {'example_id': '0', 'dataset': '0', 'stft': np.ones((1, 10, 9, 2)), 'events': np.array([0,1,0,0,1]), 'events_alignment': np.array([0,1,0,0,1])[:,None].repeat(10,axis=1)}
    >>> example2 = {'example_id': '1', 'dataset': '1', 'stft': -np.ones((1, 8, 9, 2)), 'events': np.array([0,0,1,0,0]), 'events_alignment': np.array([0,0,1,0,0])[:,None].repeat(8,axis=1)}
    >>> mixup_fn([example1, example2])
    """
    def __init__(self, min_overlap=1., max_length=None):
        self.min_overlap = min_overlap
        self.max_length = max_length

    def __call__(self, components):
        assert len(components) > 0
        start_indices = [0]
        stop_indices = [components[0]['stft'].shape[1]]
        for comp in components[1:]:
            seq_len = comp['stft'].shape[1]
            min_start = -int(seq_len*(1-self.min_overlap))
            max_start = components[0]['stft'].shape[1] - int(np.ceil(self.min_overlap*seq_len))
            if self.max_length is not None:
                assert seq_len <= self.max_length, (seq_len, self.max_length)
                min_start = max(
                    min_start, max(stop_indices) - self.max_length
                )
                max_start = min(
                    max_start, min(start_indices) + self.max_length - seq_len
                )
            start_indices.append(
                int(min_start + np.random.rand() * (max_start - min_start + 1))
            )
            stop_indices.append(start_indices[-1] + seq_len)
        start_indices = np.array(start_indices)
        stop_indices = np.array(stop_indices)
        stop_indices -= start_indices.min()
        start_indices -= start_indices.min()

        stft_shape = list(components[0]['stft'].shape)
        stft_shape[1] = stop_indices.max()
        mixed_stft = np.zeros(stft_shape, dtype=components[0]['stft'].dtype)
        if 'events' in components[0] and components[0]['events'].ndim >= 2:
            assert all([('events' in comp and comp['events'].ndim == 2) for comp in components])
            alignment_shape = list(components[0]['events'].shape)
            alignment_shape[1] = stop_indices.max()
            mixed_alignment = np.zeros(alignment_shape)
        else:
            mixed_alignment = None
        for comp, start, stop in zip(components, start_indices, stop_indices):
            mixed_stft[:, start:stop] += comp['stft']
            if mixed_alignment is not None:
                mixed_alignment[:, start:stop] += comp['events']

        mix = {
            'example_id': '+'.join([comp['example_id'] for comp in components]),
            'dataset': '+'.join(sorted(set([comp['dataset'] for comp in components]))),
            'stft': mixed_stft,
            'seq_len': mixed_stft.shape[1],
        }
        if mixed_alignment is not None:
            mix['events'] = (mixed_alignment > .5).astype(components[0]['events'].dtype)
        elif all(['events' in comp for comp in components]):
            mix['events'] = (np.sum([comp['events'] for comp in components], axis=0) > .5).astype(components[0]['events'].dtype)
        return mix
