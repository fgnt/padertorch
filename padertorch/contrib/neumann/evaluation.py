from typing import Optional, Iterable

from logging import getLogger
import paderbox as pb
import numpy as np
import pb_bss
import operator
import re

logger = getLogger('evaluation')


def compute_means(
        results: dict,
        mean_keys: Optional[Iterable] = None,
        exclude_keys: tuple = (r'.*selection', ),
) -> dict:
    """

    Args:
        results: Input data dict. Structure should be:
            `{'dataset_name': {'example_id': {...nested values...}}}`
        mean_keys: Keys (if nested, separate with '.') to compute a mean over.
            If `None`, computes mean over all keys found in the data.
        exclude_keys: Keys or key patterns to exclude when inferring mean keys
            from data. Has no effect if `mean_keys is not None`.

    Returns:
        {'dataset_name': {... nested means ...}}
    """
    means = {}
    for dataset, dataset_results in results.items():
        means[dataset] = {}

        # Flatten to structure {'example_id': {'path.to.sub.entry': value}}
        flattened = {
            k: pb.utils.nested.flatten(v) for k, v in
            dataset_results.items()
        }

        if mean_keys is None:
            # Try to infer mean keys from first element in data
            _mean_keys = list(filter(lambda x: not any(
                re.fullmatch(pattern, x) for pattern in exclude_keys
            ), next(iter(flattened.values())).keys()))
        else:
            _mean_keys = mean_keys

        for mean_key in _mean_keys:
            means[dataset][mean_key] = np.mean(np.array([
                v[mean_key] for v in flattened.values()
            ]))
        means[dataset] = pb.utils.nested.deflatten(means[dataset])

    return means
