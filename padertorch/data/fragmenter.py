from copy import deepcopy
import numpy as np
from paderbox.utils.nested import nested_op, flatten, deflatten


class Fragmenter(object):
    """
    Build fragments of the values corresponding to fragment_keys 
    along axis and adds it to the examples queue in the iterator.
    Mostly used with pb.database.iterator.FragmentIterator

    >>> time_fragmenter = Fragmenter({'a':2, 'b':1}, axis=-1)
    >>> example = {'a': np.arange(8).reshape((2, 4)), 'b': np.array([1,2])}
    >>> from pprint import pprint
    >>> pprint(time_fragmenter(example))
    [{'a': array([[0, 1],
           [4, 5]]), 'b': array([1])},
     {'a': array([[2, 3],
           [6, 7]]), 'b': array([2])}]
    >>> time_fragmenter = Fragmenter(\
            {'a':1, 'b':1}, {'a':2, 'b':1}, drop_last=True)
    >>> example = {'a': np.arange(8).reshape((2, 4)), 'b': np.array([1,2,3])}
    >>> pprint(time_fragmenter(example))
    [{'a': array([[0, 1],
           [4, 5]]), 'b': array([1])},
     {'a': array([[1, 2],
           [5, 6]]), 'b': array([2])},
     {'a': array([[2, 3],
           [6, 7]]), 'b': array([3])}]
    >>> channel_fragmenter = Fragmenter(\
            {'a':1}, axis=0, copy_keys=['b'])
    >>> example = {'a': np.arange(8).reshape((2, 4)), 'b': np.array([1,2,3,4])}
    >>> pprint(channel_fragmenter(example))
    [{'a': array([[0, 1, 2, 3]]), 'b': array([1, 2, 3, 4])},
     {'a': array([[4, 5, 6, 7]]), 'b': array([1, 2, 3, 4])}]
    >>> channel_fragmenter = Fragmenter(\
            {'a':1}, axis=0, squeeze=True, copy_keys=['b'])
    >>> example = {'a': np.arange(8).reshape((2, 4)), 'b': np.array([1,2,3,4])}
    >>> pprint(channel_fragmenter(example))
    [{'a': array([0, 1, 2, 3]), 'b': array([1, 2, 3, 4])},
     {'a': array([4, 5, 6, 7]), 'b': array([1, 2, 3, 4])}]
    """
    def __init__(
            self, fragment_steps, fragment_lengths=None, axis=-1,
            squeeze=False, drop_last=False, copy_keys=None
    ):
        self.fragment_steps = fragment_steps
        self.fragment_lengths = fragment_lengths \
            if fragment_lengths is not None else fragment_steps
        self.axis = axis
        self.squeeze = squeeze
        self.drop_last = drop_last
        self.copy_keys = copy_keys

    def __call__(self, example, random_onset=False):
        copies = flatten(
            {key: example[key] for key in self.copy_keys}
            if self.copy_keys is not None else example
        )

        if random_onset:
            start = np.random.rand()
            for fragment_step in self.fragment_steps.values():
                start = int(int(start*fragment_step) / fragment_step)
        else:
            start = 0.

        def fragment(key, x):
            fragment_step = self.fragment_steps[key]
            fragment_length = self.fragment_lengths[key]
            start_idx = int(start * fragment_step)
            if start_idx > 0:
                slc = [slice(None)] * len(x.shape)
                slc[self.axis] = slice(
                    int(start_idx), x.shape[self.axis]
                )
                x = x[slc]

            end_index = x.shape[self.axis]
            if self.drop_last:
                end_index -= (fragment_length - 1)
            fragments = list()
            for start_idx in np.arange(0, end_index, fragment_step):
                if fragment_length == 1 and self.squeeze:
                    fragments.append(x.take(start_idx, axis=self.axis))
                else:
                    slc = [slice(None)] * len(x.shape)
                    slc[self.axis] = slice(
                        int(start_idx), int(start_idx) + int(fragment_length)
                    )
                    fragments.append(x[tuple(slc)])
            return fragments

        features = flatten({
            key: nested_op(lambda x: fragment(key, x), example[key])
            for key in self.fragment_steps.keys()
        })
        num_fragments = np.array(
            [len(features[key]) for key in list(features.keys())]
        )
        assert all(num_fragments == num_fragments[0]), (list(features.keys()), num_fragments)
        fragments = list()
        for i in range(int(num_fragments[0])):
            fragment = deepcopy(copies)
            for key in features.keys():
                fragment[key] = features[key][i]
            fragment = deflatten(fragment)
            fragments.append(fragment)
        return fragments
