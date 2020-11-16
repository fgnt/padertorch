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


def get_segment_offsets(num_samples, length, step=None,
                        offset_mode='start', num_segments=None):
    """

    Args:
        num_samples:
        segment_length:
        shift:
        offset_mode:
        num_segments:

    Returns:
    >>> get_segment_offsets(25, 10, None, offset_mode='start')
    array([ 0, 10])
    >>> get_segment_offsets(25, 10, 3, offset_mode='start')
    array([ 0,  3,  6,  9, 12, 15])
    >>> get_segment_offsets(24, 10, 3, offset_mode='start')
    array([ 0,  3,  6,  9, 12])
    >>> get_segment_offsets(24, 10, 3, offset_mode='end')
    array([ 2,  5,  8, 11, 14])
    >>> get_segment_offsets(24, 10, 3, offset_mode='center')
    array([ 1,  4,  7, 10, 13])
    >>> get_segment_offsets(24, 10, 3, offset_mode='center', num_segments=2)
    array([1, 4])
    >>> get_segment_offsets(24, 10, 3, offset_mode='center', num_segments=1)
    array([1])
    """
    assert num_samples >= length, (num_samples, length)
    if step is None:
        step = length
    assert step > 0, step

    if offset_mode == 'start':
        start = 0
    elif offset_mode == 'random':
        start = np.random.randint(num_samples - length)
    elif offset_mode in ['end', 'center']:
        remainder = (num_samples - length) % step
        if offset_mode == 'end':
            start = remainder
        else:
            start = remainder // 2
    else:
        raise ValueError('Unknown offset mode', offset_mode)
    offsets = np.arange(start, num_samples - length + 1, step)
    if num_segments:
        max_segments = (num_samples - length) // step + 1
        assert num_segments <= max_segments, (num_segments, max_segments)
        offsets = offsets[:num_segments]
    return offsets


def segment(x, length, step=None, axis=-1,
            offset_mode='start', num_segments=None):
    """

    Args:
        x:
        segment_length:
        step:
        axis:
        offset_mode:
        num_segments:

    Returns:

    >>> segment(np.arange(0, 15), 10, 3, -1)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
    """
    ndim = x.ndim
    axis = axis % ndim

    offsets = get_segment_offsets(x.shape[axis], length, step,
                                  offset_mode, num_segments)
    # slice the array to remove samples not addressed with
    # this offsets-segment_length combination
    slc = [slice(None)] * ndim
    slc[axis] = slice(offsets[0], offsets[-1] + length)
    x = x[tuple(slc)]

    return segment_axis(x, length, step, end='cut')
