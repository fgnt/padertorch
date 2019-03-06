import operator

import numpy as np

import torch

from padertorch.utils import to_numpy


__all__ = [
    'mask_to_image',
    'stft_to_image',
    'spectrogram_to_image',
    'review_dict',
]


def _remove_batch_axis(array, batch_first):
    if array.ndim == 2:
        pass
    elif array.ndim == 3:
        if batch_first:
            array = array[0]
        else:
            array = array[:, 0]
    else:
        raise ValueError('Either the signal has ndim 2 or 3',
                         array.shape)
    return array


def mask_to_image(mask, batch_first=False):
    """
    For more details of the output shape, see the tensorboardx docs
    Args:
        mask: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(color, features, frames)

    """
    mask = to_numpy(mask, detach=True)

    image = np.clip(mask * 255, 0, 255)
    image = image.astype(np.uint8)

    image = _remove_batch_axis(image, batch_first=batch_first)

    return image[None].transpose(0, 2, 1)[:, ::-1]


def stft_to_image(signal, batch_first=False, color='viridis'):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        signal: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    signal = to_numpy(signal, detach=True)

    return spectrogram_to_image(
        np.abs(signal), batch_first=batch_first, color=color
    )


_spectrogram_to_image_cmap = {}


def spectrogram_to_image(signal, batch_first=False, color='viridis'):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        mask: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    signal = to_numpy(signal, detach=True)

    signal = signal / np.max(signal)

    signal = _remove_batch_axis(signal, batch_first=batch_first)

    visible_dB = 50

    # remove problematic small numbers
    floor = 10 ** (-visible_dB / 20)
    signal = np.maximum(signal, floor)

    # Scale such that X dB are visible (i.e. in the range 0 to 1)
    signal = (20 / visible_dB) * np.log10(signal) + 1

    signal = (signal * 255).astype(np.uint8)

    if color is not None:
        try:
            cmap = _spectrogram_to_image_cmap[color]
        except KeyError:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(color)
            _spectrogram_to_image_cmap[color] = cmap

        return cmap(signal).transpose(2, 1, 0)[:, ::-1, :]
    else:
        # gray image
        return signal.transpose(1, 0)[None, ::-1, :]


def review_dict(
        *,
        loss: torch.Tensor=None,
        losses: dict=None,
        scalars: dict=None,
        histograms: dict=None,
        audios: dict=None,
        images: dict=None,
):
    """
    This is a helper function to build the review dict.
    The main purpose is for auto completion of the review dict, prevent typos
    and documentation what is expected for the values.

    ToDo: Text for expected shapes

    Args:
        loss:
            Scalar torch.Tensor. If not None, expect losses to be None.
        losses:
            Dict of scalar torch.Tensor. If not None, expect loss to be None.
        scalars:
            Dict of scalars that are reported to tensorboard. Losses and loss
            are also reported as scalars.
        histograms:
            Dict of ???.
        audios:
            Dict of ???.
        images:
            Dict of torch.Tensor with Shape(batch, features, frames, 1).

    Returns:
        dict of the args that are not None

    """

    review = locals()

    for k, v in list(review.items()):
        if v is None:
            del review[k]

    assert operator.xor(loss is None, losses is None), (loss, losses)

    return review
