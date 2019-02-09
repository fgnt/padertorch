import numpy as np
import torch


__all__ = [
    'example_to_device',
    'Sorter',
]


def example_to_device(example, device=None):
    """
    Moves a nested structure to the device.
    Numpy arrays are converted to torch.Tensor, except complex numpy arrays
    that aren't supported in the moment in torch.

    The original doctext from torch for `.to`:
    Tensor.to(device=None, dtype=None, non_blocking=False, copy=False) → Tensor
        Returns a Tensor with the specified device and (optional) dtype. If
        dtype is None it is inferred to be self.dtype. When non_blocking, tries
        to convert asynchronously with respect to the host if possible, e.g.,
        converting a CPU Tensor with pinned memory to a CUDA Tensor. When copy
        is set, a new Tensor is created even when the Tensor already matches
        the desired conversion.

    Args:
        example:
        device: None, 'cpu', 0, 1, ...

    Returns:
        example on device

    """

    if isinstance(example, dict):
        return example.__class__({
            key: example_to_device(value, device=device)
            for key, value in example.items()
        })
    elif isinstance(example, (tuple, list)):
        return example.__class__([
            example_to_device(element, device=device)
            for element in example
        ])
    elif torch.is_tensor(example):
        return example.to(device=device)
    elif isinstance(example, np.ndarray):
        if example.dtype in [np.complex64, np.complex128]:
            # complex is not supported
            return example
        else:
            # TODO: Do we need to ensure tensor.is_contiguous()?
            # TODO: If not, the representer of the tensor does not work.
            return example_to_device(
                torch.from_numpy(example), device=device
            )
    elif hasattr(example, '__dataclass_fields__'):
        return example.__class__(
            **{
                f: example_to_device(getattr(example, f), device=device)
                for f in example.__dataclass_fields__
            }
        )
    else:
        return example


class Sorter:
    # pb.database.keys.NUM_SAMPLES is 'num_samples'
    def __init__(self, key=lambda example: example['num_samples']):
        self.key = key

    def __call__(self, examples):
        return tuple(sorted(
            examples,
            key=self.key,
            reverse=True,
        ))
