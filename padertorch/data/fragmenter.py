from copy import deepcopy
from typing import Union

import numpy as np
from paderbox.array import segment_axis
from paderbox.utils.nested import nested_op, flatten, deflatten

possible_anchor_modes = [
    'left',
    'right',
    'center',
    'centered_cutout',
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


def get_anchor(
        num_samples: int, length: int, shift: int=None,
        mode: str = 'left', rng=np.random
):
    """
    Calculates anchor for the boundaries for segmentation of a signal
    with length `num_sammples` in case of a fixed segment length
    `length` and shift `shift`

    Args:
        num_samples: num samples of signal for which boundaries are caclulated
        length: segment length
        shift: shift between segments, defaults to length
        mode: Defines the position of the boundaries in the signal:
            left: anchor is set to zero
                so that only values at the end are cut
            right: anchor is set to `num_samples`
                so that only values at the beginning are cut
            center: anchor is set to `num_samples // 2`
            centered_cutout: the anchor is chosen such that the same number
                of samples are discarded at the end and the beginning
            random: anchor is set to a random value between
                `0` and `num_samples`.
                 This may reduce the number of possible segments.
            random_max_segments: Randomly chooses the anchor such that
                the maximum number of segments are created
        rng: random number generator (`numpy.random`)

    Returns:
       integer value describing the anchor
    >>> np.random.seed(3)
    >>> get_anchor(24, 10, 3, mode='left')
    0
    >>> get_anchor(24, 10, 3, mode='right')
    14
    >>> get_anchor(24, 10, 3, mode='center')
    12
    >>> get_anchor(24, 10, 3, mode='centered_cutout')
    1
    >>> get_anchor(24, 10, 3, mode='random')
    10
    >>> get_anchor(24, 10, 3, mode='random_max_segments')
    3
    """
    assert num_samples >= length, (num_samples, length)
    if shift is None:
        shift = length
    assert shift > 0, shift

    if mode == 'left':
        return 0
    elif mode == 'right':
        return num_samples - length
    elif mode == 'center':
        return num_samples // 2
    elif mode == 'centered_cutout':
        remainder = (num_samples - length) % shift
        return remainder // 2
    elif mode == 'random':
        return rng.randint(num_samples - length)
    elif mode == 'random_max_segments':
        start = rng.randint((num_samples - length) % shift + 1)
        anchors = np.arange(start, num_samples - length + 1, shift)
        return int(np.random.choice(anchors))
    else:
        raise ValueError('Unknown mode', mode,
                         'choose on of', possible_anchor_modes)


def get_segment_boundaries(
        anchor: Union[str, int], num_samples: int, length: int,
        shift: int = None, rng=np.random
):
    """
    Calculates boundaries for segmentation of a signal with length
    `num_sammples` in case of a fixed segment length `length` and shift `shift`

    Args:
        anchor: anchor from which the segmentation boundaries are calculated.
            if it is a string `get_anchor` is called to calculate an integer
            using `anchor` as anchor mode definition.
        num_samples: num samples of signal for which boundaries are caclulated
        length: segment length
        shift: shift between segments, defaults to length
        rng: random number generator (`numpy.random`)

    Returns:
        Bx2 numpy array with start and end values for B boundaries

    >>> np.random.seed(3)
    >>> get_segment_boundaries('left', 24, 10, 3).T
    array([[ 0,  3,  6,  9, 12],
           [10, 13, 16, 19, 22]])
    >>> get_segment_boundaries('right', 24, 10, 3).T
    array([[ 2,  5,  8, 11, 14],
           [12, 15, 18, 21, 24]])
    >>> get_segment_boundaries('center', 24, 10, 3).T
    array([[ 0,  3,  6,  9, 12],
           [10, 13, 16, 19, 22]])
    >>> get_segment_boundaries('centered_cutout', 24, 10, 3).T
    array([[ 1,  4,  7, 10, 13],
           [11, 14, 17, 20, 23]])
    >>> get_segment_boundaries('random', 24, 10, 3).T
    array([[ 1,  4,  7, 10, 13],
           [11, 14, 17, 20, 23]])
    >>> get_segment_boundaries('random_max_segments', 24, 10, 3).T
    array([[ 0,  3,  6,  9, 12],
           [10, 13, 16, 19, 22]])
    """
    assert num_samples >= length, (num_samples, length)
    if shift is None:
        shift = length
    assert shift > 0, shift
    if isinstance(anchor, str):
        anchor = get_anchor(num_samples, length, shift, mode=anchor, rng=rng)
    assert isinstance(anchor, int), (anchor, type(anchor))

    start = anchor % shift
    start = np.arange(start, num_samples - length + 1, shift)
    stop = start + length
    boundaries = np.stack([start, stop], axis=-1)
    return boundaries


def segment(
        x, anchor: Union[str, int], length: int, shift: int = None,
        axis: int = -1, rng=np.random
):
    """
    Segments a signal `x` along an axis. Either with a predefined anchor for
    the segment boundaries if anchor is set or with an internally calculated
    anchor if anchor is a string.

    Args:
        x: signal to be segmented
        anchor: anchor from which the segmentation boundaries are calculated.
            if it is a string `get_anchor` is called to calculate an integer
            using `anchor` as anchor mode definition.
        length: segment length
        shift: shift between segments, defaults to length
        axis: axis over which to segment
        rng: random number generator (`numpy.random`)

    Returns:

    >>> np.random.seed(3)
    >>> segment(np.arange(0, 15), 'left', 10, 3)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
    >>> segment(np.arange(0, 15), 'random', 10, 3)
    array([[ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])
    >>> segment(np.arange(0, 15), 5, 10, 3)
    array([[ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
           [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14]])
    """
    if x.__class__.__module__ == 'numpy':
        ndim = x.ndim
    elif x.__class__.__module__ == 'torch':
        ndim = x.dim()
    elif isinstance(x, list):
        x = np.array(x)
        ndim = x.ndim
    else:
        raise TypeError('Unknown type for input signal x', type(x))
    axis = axis % ndim

    num_samples = x.shape[axis]
    assert num_samples >= length, (num_samples, length)
    if shift is None:
        shift = length
    assert shift > 0, shift
    if isinstance(anchor, str):
        anchor = get_anchor(num_samples, length, shift, mode=anchor, rng=rng)
    assert isinstance(anchor, int), (anchor, type(anchor))

    start = anchor % shift

    # slice the array to remove samples discarded with the specified anchor
    slc = [slice(None)] * ndim
    slc[axis] = slice(start, None)
    x = x[tuple(slc)]

    return np.moveaxis(segment_axis(
        x, length, shift, end='cut', axis=axis), axis, 0)
