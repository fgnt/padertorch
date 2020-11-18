from copy import deepcopy
import numpy as np
from paderbox.utils.nested import nested_op, flatten, deflatten
from paderbox.array import segment_axis


possible_boundary_modes = [
    'begin',
    'end',
    'center',
    'random',
    'random_max_segments',
]


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


def get_segment_boundaries(num_samples, length, shift=None,
                           mode='begin', choose_one=False, rng=np.random):
    """

    Args:
        num_samples: num samples of signal for which boundaries are caclulated
        length: segment length
        shift: shift between segments
        mode: Defines the position of the boundaries in the signal:
            begin: boundaries start at sample zero
                so that only values at the end are cut
            end: boundaries end at num_samples
                so that only values at the beginning are cut
            center: boundaries are chosen such that the same number of samples
                are discarded at the end and the beginning of the signal
            random: Starts the first segment randomly at a value
                between 0 and shift. This may reduce the number of segments.
            random_max_segments: Randomly chooses the first segment such that
                the maximum number of segments are created
        choose_one: If True only outputs one segment boundary
        rng: random number generator (numpy.random)

    Returns:
        2xB numpy array with start and end values for B boundaries

    >>> np.random.seed(0)
    >>> get_segment_boundaries(25, 10, None, mode='begin')
    array([[ 0, 10],
           [10, 20]])
    >>> get_segment_boundaries(25, 10, 3, mode='begin')
    array([[ 0,  3,  6,  9, 12, 15],
           [10, 13, 16, 19, 22, 25]])
    >>> get_segment_boundaries(24, 10, 3, mode='begin')
    array([[ 0,  3,  6,  9, 12],
           [10, 13, 16, 19, 22]])
    >>> get_segment_boundaries(24, 10, 3, mode='random')
    array([[ 0,  3,  6,  9, 12],
           [10, 13, 16, 19, 22]])
    >>> get_segment_boundaries(24, 10, 3, mode='random_max_segments')
    array([[ 1,  4,  7, 10, 13],
           [11, 14, 17, 20, 23]])
    >>> get_segment_boundaries(24, 10, 3, mode='center')
    array([[ 1,  4,  7, 10, 13],
           [11, 14, 17, 20, 23]])
    >>> get_segment_boundaries(24, 10, 3, mode='center', choose_one=True)
    array([[ 1],
           [11]])
    """
    assert num_samples >= length, (num_samples, length)
    if shift is None:
        shift = length
    assert shift > 0, shift

    if mode == 'begin':
        start = 0
    elif mode == 'random':
        start = rng.randint(shift)
    elif mode == 'random_max_segments':
        start = rng.randint((num_samples - length) % shift)
    elif mode in ['end', 'center']:
        remainder = (num_samples - length) % shift
        if mode == 'end':
            start = remainder
        else:
            start = remainder // 2
    else:
        raise ValueError('Unknown offset mode', mode,
                         'choose on of', possible_boundary_modes)
    start = np.arange(start, num_samples - length + 1, shift)
    if choose_one:
        start = rng.choice(start, 1)
    stop = start + length
    boundaries = np.stack([start, stop], axis=0)
    return boundaries


def segment(x, axis=-1, boundary=None, length: int=None, shift=None,
            mode='begin', choose_one=False, rng=np.random):
    """
    Segments a signal x along an axis. Either with predefined segment
    boundaries if boundary is set or with internally calculated boundaries if
    length is set.

    Args:
        x: signal to be segmented
        axis: axis which is segmented
        boundary: 2xB numpy array with start and end values
            for B pre calculated boundaries, which may be used to segment
            multiple audios in the same way.
        length: segment length
        shift: shift between segments
        mode: Defines the position of the boundaries in the signal:
            begin: boundaries start at sample zero
                so that only values at the end are cut
            end: boundaries end at num_samples
                so that only values at the beginning are cut
            center: boundaries are chosen such that the same number of samples
                are discarded at the end and the beginning of the signal
            random: Starts the first segment randomly at a value
                between 0 and shift
            random_max_segments: Randomly chooses the first segment such that
                the maximum number of segments are created
        choose_one: If True only outputs one segment boundary
        rng: random number generator (numpy.random)

    Returns:

    >>> segment(np.arange(0, 15), -1, None, 10, 3)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
    """

    if boundary is None:
        boundary = get_segment_boundaries(x.shape[axis], length, shift,
                                            mode, choose_one, rng)
    else:
        assert length is None, (
            'Either use predefined boundaries or set a segment length'
            'because only one of these information can be used.'
        )
        if isinstance(boundary, (list, tuple)):
            boundary = np.array(boundary)
            if boundary.ndim == 1:
                boundary = boundary[..., None]
            assert boundary.ndim == 2, boundary.shape
        elif x.__class__.__module__ == 'numpy':
            assert x.ndim == 2, x.shape
        elif x.__class__.__module__ == 'torch':
            assert x.dim() == 2, x.shape
        else:
            raise TypeError('The signal x has an unknown type', type(x))

        assert boundary.shape[0] == 2, 'The first dimension has to be 2 to' \
                                       ' describe start and stop of a segment'

    # slice the array to remove samples not addressed with
    # this offsets-segment_length combination
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(boundary[0, 0], boundary[1, -1])
    x = x[tuple(slc)]

    return segment_axis(x, length, shift, end='cut')
