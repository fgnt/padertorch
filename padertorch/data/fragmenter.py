from copy import deepcopy
import numpy as np
from paderbox.utils.nested import nested_op, flatten, deflatten


class ChannelFragmenter(object):
    """
    Takes the first dimension of the values corresponding to fragment_keys
    and adds it to the examples queue in the iterator.
    Mostly used with pb.database.iterator.FragmentIterator

    >>> channel_fragmenter = ChannelFragmenter(['a', 'b'])
    >>> example = {'a': np.array([[1,2],[3,4]]), 'b': np.array([[5,6],[7,8]])}
    >>> channel_fragmenter(example)
    [{'a': array([1, 2]), 'b': array([5, 6])}, {'a': array([3, 4]), 'b': array([7, 8])}]
    >>> channel_fragmenter = ChannelFragmenter(['a'], copy_keys='c')
    >>> example = {'a': np.array([[1,2],[3,4]]), 'b': np.array([[1,2],[3,4]]), 'c': 10}
    >>> channel_fragmenter(example)
    [{'c': 10, 'a': array([1, 2])}, {'c': 10, 'a': array([3, 4])}]
    """
    def __init__(self, fragment_keys, copy_keys=None, keep_dim=False):
        self.fragment_keys = fragment_keys
        self.copy_keys = copy_keys
        self.keep_dim = keep_dim

    def __call__(self, example):
        copies = flatten(
            {key: example[key] for key in self.copy_keys}
            if self.copy_keys is not None else example
        )

        def fragment_channels(x):
            if self.keep_dim:
                return [x_ for x_ in x ]
            else:
                return [x[idx:idx+1] for idx in range(len(x))]

        features = nested_op(
            fragment_channels,
            flatten({key: example[key] for key in self.fragment_keys})
        )
        num_fragments = [len(features[key]) for key in list(features.keys())]
        assert all([n == num_fragments[0] for n in num_fragments])
        num_fragments = num_fragments[0]
        fragments = list()
        for i in range(num_fragments):
            fragment = deepcopy(copies)
            for key in features.keys():
                fragment[key] = features[key][i]
            fragment = deflatten(fragment)
            fragments.append(fragment)
        return fragments


class TimeFragmenter(object):
    """
    >>> time_fragmenter = TimeFragmenter({'a':2, 'b':1})
    >>> example = {'a': np.array([1,2,3,4]), 'b': np.array([1,2])}
    >>> time_fragmenter(example)
    [{'a': array([1, 2]), 'b': array([1])}, {'a': array([3, 4]), 'b': array([2])}]
    >>> example = {'a': np.array([1,2,3,4,5]), 'b': np.array([1,2,3])}
    >>> time_fragmenter(example)
    [{'a': array([1, 2]), 'b': array([1])}, {'a': array([3, 4]), 'b': array([2])}, {'a': array([5]), 'b': array([3])}]
    >>> time_fragmenter = TimeFragmenter({'a':2, 'b':1}, drop_last=True, copy_keys=['c'])
    >>> example = {'a': np.array([1,2,3,4,5]), 'b': np.array([1,2,3]), 'c': 10}
    >>> time_fragmenter(example)
    [{'c': 10, 'a': array([1, 2]), 'b': array([1])}, {'c': 10, 'a': array([3, 4]), 'b': array([2])}]
    >>> time_fragmenter = TimeFragmenter({'a':1, 'b':1}, {'a':2, 'b':1}, drop_last=True)
    >>> example = {'a': np.array([1,2,3,4]), 'b': np.array([1,2,3])}
    >>> time_fragmenter(example)
    [{'a': array([1, 2]), 'b': array([1])}, {'a': array([2, 3]), 'b': array([2])}, {'a': array([3, 4]), 'b': array([3])}]
    """
    def __init__(self, fragment_steps, fragment_lengths=None,
                 training=False, drop_last=False, copy_keys=None):
        self.fragment_steps = fragment_steps
        self.fragment_lengths = fragment_lengths \
            if fragment_lengths is not None else fragment_steps
        self.drop_last = drop_last
        self.training = training
        self.copy_keys = copy_keys

    def __call__(self, example):
        copies = flatten(
            {key: example[key] for key in self.copy_keys}
            if self.copy_keys is not None else example
        )

        if self.training:
            start = np.random.rand()
            for fragment_step in self.fragment_steps.values():
                start = int(int(start*fragment_step) / fragment_step)
        else:
            start = 0.

        def fragment_time(key, x):
            fragment_step = self.fragment_steps[key]
            fragment_length = self.fragment_lengths[key]
            first_idx = int(start * fragment_step)
            x = x[..., first_idx:]

            if self.drop_last:
                tail = int((x.shape[-1] - fragment_length) % fragment_step)
                if tail > 0:
                    x = x[..., :-tail]
            fragments = list()
            for start_idx in np.arange(0, x.shape[-1], fragment_step):
                fragments.append(
                    x[..., int(start_idx): int(start_idx) + int(fragment_length)]
                )
            return fragments

        features = flatten({
            key: nested_op(lambda x: fragment_time(key, x), example[key])
            for key in self.fragment_steps.keys()
        })
        num_fragments = np.array([len(features[key]) for key in list(features.keys())])
        assert all(num_fragments == num_fragments[0])
        fragments = list()
        for i in range(int(num_fragments[0])):
            fragment = deepcopy(copies)
            for key in features.keys():
                fragment[key] = features[key][i]
            fragment = deflatten(fragment)
            fragments.append(fragment)
        return fragments
