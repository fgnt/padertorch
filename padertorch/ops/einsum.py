import string

import torch

__all__ = [
    'einsum'
]


def einsum(operation: str, *operands):
    """Allows capital letters and collects operands as in `np.einsum`."""
    remaining_letters = set(string.ascii_lowercase)
    remaining_letters = remaining_letters - set(operation)
    for capital_letter, replacement in zip(set.intersection(
            set(string.ascii_uppercase),
            set(operation)
    ), remaining_letters):
        operation = operation.replace(capital_letter, replacement)
    return torch.einsum(operation, operands)
