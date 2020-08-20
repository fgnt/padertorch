from copy import deepcopy

import lazy_dataset
import numpy as np

from dataclasses import dataclass


def _getitem_on_axis(array, item, axis):
    slicer = [slice(None)] * array.ndim
    slicer[axis] = item
    return array[tuple(slicer)]


@dataclass
class Chunk:
    """
    Computes chunks as in the original DPRNN paper. This implementation
    is based on Yis description:

            "For an utterance, we truncate it into 4s chunks with 50%
        overlap. That said, for a 6s mixture, it is truncated into 2
        chunks [0, 4]s and [2, 6]s, but for a 5s mixture it only has 1
        chunk [0,4]s (no zero-padding at the end for another chunk).
        You can definitely split every utterance into even more chunks
        with for example random start/end indices, or to apply
        zero-padding at the end (or even randomly for any chunk less
        than 4s long). There's infinite way to do such data augmentation
        and we did not play too much with it, and all our models are
        trained with the same setting."

    If an utterance is shorter than `chunk_length`, a
    `lazy_dataset.FilterException` is raised.

    This chunking returns a list of chunked examples that can be unbatched.
    Everything that is not listed in `chunk_keys` is simply copied from the
    input example to the output examples. The key `num_samples` is updated
    with the `chunk_size`.

    Examples:
        >>> c = Chunk(chunk_size=32000, chunk_keys=('x', 'y'))
        >>> ex = {'x': np.arange(65000), 'y': np.arange(65000), 'num_samples': 65000, 'gender': 'm'}
        >>> chunked = c(ex)
        >>> type(chunked)
        <class 'list'>
        >>> len(chunked)
        3
        >>> chunked[0]['num_samples']
        32000
        >>> chunked[0]['x'].shape
        (32000,)
        >>> chunked[0]['gender']
        'm'
        >>> for entry in chunked:
        ...     print(entry['x'][[0, -1]])
        [    0 31999]
        [16000 47999]
        [32000 63999]

        Chunking can be disabled by setting `chunk_size=-1`
        >>> Chunk(chunk_size=-1, chunk_keys=('x', 'y'))(ex)[0] == ex
        True

        Check the corner cases.
        >>> ex = {'x': np.arange(64000), 'y': np.arange(64000)}
        >>> for entry in c(ex):
        ...     print(entry['x'][[0, -1]])
        [    0 31999]
        [16000 47999]
        [32000 63999]
        >>> ex = {'x': np.arange(63999), 'y': np.arange(63999)}
        >>> for entry in c(ex):
        ...     print(entry['x'][[0, -1]])
        [    0 31999]
        [16000 47999]

    Args:
        chunk_size: The size of the cut chunks in samples. If set to `-1`,
            the original example is returned in a list of length one.
        chunk_keys: The keys in the passed example dict to chunk. The all
            must have the same size along `axis`
        axis: The axis to chunk along


    """
    chunk_size: int
    chunk_keys: tuple
    axis: int = 0

    def __call__(self, example):
        # Shortcut if chunking is disabled
        if self.chunk_size == -1:
            return [example]

        to_chunk = {k: example.pop(k) for k in self.chunk_keys}
        to_chunk_lengths = [c.shape[self.axis] for c in to_chunk.values()]
        assert to_chunk_lengths[1:] == to_chunk_lengths[:-1], (
            'The shapes along the chunk dimension of all entries to chunk must '
            'be equal!\n'
            f'chunk_keys: {self.chunk_keys}'
            f'to_chunk_lengths: {to_chunk_lengths}'
        )
        to_chunk_length = to_chunk_lengths[0]

        # Discard examples that are shorter than `chunk_size`
        if to_chunk_length < self.chunk_size:
            raise lazy_dataset.FilterException()

        # Cut overlapping chunks
        chunks = []

        shift = self.chunk_size // 2
        for chunk_beginning in range(
                0,
                to_chunk_length - self.chunk_size + 1,  # only full sizes
                shift,
        ):
            chunk_end = chunk_beginning + self.chunk_size
            chunk = deepcopy(example)
            chunk.update({
                k: _getitem_on_axis(v, slice(chunk_beginning, chunk_end), axis=self.axis)
                for k, v in to_chunk.items()
            })
            chunk.update(num_samples=self.chunk_size)
            chunks.append(chunk)

        return chunks


@dataclass
class RandomChunkSingle:
    """
    Cuts one (single) random chunk from the data of the input example. The
    remaining data is discarded. If an example is shorter than `min_length`,
    it is discarded. If `min_length` is not set (`None`) it defaults to
    `chunk_size`. If the example is shorter than `chunk_size` but longer than
    `min_length` it gets padded with 0s.

    Args:
        chunk_size: The size of the cut chunks in samples. If set to `-1`,
            the original example is returned.
        chunk_keys: The keys in the passed example dict to chunk. The all
            must have the same size along `axis`
        axis: The axis to chunk along
        min_length: Examples that are shorter than `min_length` are discarded.
            Examples that are longer than `min_length` but shorter than
            `chunk_size` are randomly zero-padded to `chunk_size`. Defaults to
            `chunk_size`.

    Examples:
        >>> np.random.seed(3)
        >>> c = RandomChunkSingle(chunk_size=4, chunk_keys=('x',))
        >>> c({'x': np.arange(6)})
        {'x': array([2, 3, 4, 5]), 'num_samples': 4}
        >>> c({'x': np.arange(4)})
        {'x': array([0, 1, 2, 3]), 'num_samples': 4}
        >>> c({'x': np.arange(3)})
        Traceback (most recent call last):
        ...
        lazy_dataset.core.FilterException

        # Check the Padding
        >>> c = RandomChunkSingle(chunk_size=4, min_length=2, chunk_keys=('x',))
        >>> c({'x': np.arange(4) + 1})
        {'x': array([1, 2, 3, 4]), 'num_samples': 4}
        >>> c({'x': np.arange(3) + 1})
        {'x': array([1, 2, 3, 0]), 'num_samples': 4}
        >>> c({'x': np.arange(2) + 1})
        {'x': array([0, 1, 2, 0]), 'num_samples': 4}
    """
    chunk_size: int
    chunk_keys: tuple
    axis: int = 0
    min_length: int = None

    def __post_init__(self):
        assert self.chunk_size == -1 or self.chunk_size > 0, (
            f'Invalid chunk size: {self.chunk_size}'
        )
        if self.min_length is None:
            self.min_length = self.chunk_size

    def __call__(self, example):
        # Shortcut if chunking is disabled
        if self.chunk_size == -1:
            return example

        to_chunk = {k: example.pop(k) for k in self.chunk_keys}
        to_chunk_lengths = [c.shape[self.axis] for c in to_chunk.values()]
        assert to_chunk_lengths[1:] == to_chunk_lengths[:-1], (
            f'The shapes along the chunk dimension of all entries to chunk '
            f'must be equal! {to_chunk_lengths}'
        )
        to_chunk_length = to_chunk_lengths[0]

        if to_chunk_length < self.min_length:
            raise lazy_dataset.FilterException()
        elif to_chunk_length < self.chunk_size:
            # Pad
            from paderbox.array.padding import pad_axis
            pad_width = [
                ((self.chunk_size - to_chunk_length) // 2),
                ((self.chunk_size - to_chunk_length) // 2) + 1,
                # +1 for odd case, in even case does not hurt
            ]
            to_chunk = {
                k: pad_axis(v, pad_width, axis=self.axis)
                for k, v in to_chunk.items()}
            start = 0
        elif to_chunk_length >= self.chunk_size:
            start = np.random.randint(0, to_chunk_length - self.chunk_size + 1)
        else:
            raise RuntimeError(
                to_chunk_length, self.min_length, self.chunk_size)

        chunk = deepcopy(example)
        chunk.update({
            k: _getitem_on_axis(v, slice(start, start + self.chunk_size), axis=self.axis).copy()
            for k, v in to_chunk.items()
        })
        chunk.update(num_samples=self.chunk_size)

        return chunk
