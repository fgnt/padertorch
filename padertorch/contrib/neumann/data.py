from typing import Union


def sort_by(key: Union[str, callable], reverse: bool = True):
    """
    Sorts the example in a batch by `key`. Meant to be mapped to a lazy dataset
    after batching and before collating like
    `dataset.batch(4).map(sort_by('num_samples')).map(collate_fn)`.

    Examples:
        >>> batch = [{'value': x} for x in [5, 1, 3, 2]]
        >>> sort_by('value')(batch)
        [{'value': 5}, {'value': 3}, {'value': 2}, {'value': 1}]

    Args:
        key: Key to sort by
        reverse: If `True`, sorts in reverse order. The default `True` is
            required if sorting by length for `PackedSequence`s.
    """
    if not callable(key):
        key_ = key

        def key(example): return example[key_]

    def _sort_by(example):
        return sorted(example, key=key, reverse=reverse)

    return _sort_by
