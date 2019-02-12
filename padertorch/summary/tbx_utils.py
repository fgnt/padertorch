import numpy as np
import torch

__all__ = [
    'mask_to_image',
    'stft_to_image',
]


def mask_to_image(mask, batch_first=False):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        mask: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    if torch.is_tensor(mask):
        mask = mask.cpu().detach().numpy()
    image = np.clip(mask * 255, 0, 255)
    image = image.astype(np.uint8)
    if image.ndim == 2:
        image = image[None]
    elif image.ndim == 3:
        if batch_first:
            image = image
        else:
            image = image.transpose(1,0,2)
    else:
        raise ValueError('Either the signal has ndim 2 or 3',
                         image.shape)
    return image[:1].transpose(0,2,1)[:, ::-1]


def stft_to_image(signal, batch_first=False):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        mask: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    if torch.is_tensor(signal):
        signal = signal.cpu().detach().numpy()
    return spectrogram_to_image(np.abs(signal), batch_first=batch_first)


def spectrogram_to_image(signal, batch_first=False):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        mask: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    if torch.is_tensor(signal):
        signal = signal.cpu().detach().numpy()
    return mask_to_image(signal / np.max(signal), batch_first=batch_first)
