import numpy as np
import torch


def batch_to_device(batch, use_cuda=False, gpu_device=None):
    if isinstance(batch, dict):
        return batch.__class__({
            key: batch_to_device(value, use_cuda, gpu_device)
            for key, value in batch.items()
        })
    elif isinstance(batch, (tuple, list)):
        return batch.__class__([
            batch_to_device(element, use_cuda, gpu_device)
            for element in batch
        ])
    elif torch.is_tensor(batch):
        if use_cuda:
            return batch.cuda(gpu_device)
        else:
            return batch.cpu()
    elif isinstance(batch, np.ndarray):
        if batch.dtype in [np.complex64, np.complex128]:
            # complex is not supported
            return batch
        else:
            # TODO: Do we need to ensure tensor.is_contiguous()?
            # TODO: If not, the representer of the tensor does not work.
            return batch_to_device(
                torch.from_numpy(batch), use_cuda, gpu_device
            )
    elif hasattr(batch, '__dataclass_fields__'):
        return batch.__class__(
            **{
                f: batch_to_device(getattr(batch, f), use_cuda, gpu_device)
                for f in batch.__dataclass_fields__
            }
        )
    else:
        return batch
