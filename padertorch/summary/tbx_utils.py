import operator

import numpy as np
import torch

from padertorch.utils import to_numpy

__all__ = [
    'mask_to_image',
    'stft_to_image',
    'spectrogram_to_image',
    'review_dict',
    'audio',
]


def _remove_batch_axis(array, batch_first, ndim=2):
    if array.ndim == ndim:
        pass
    elif array.ndim == ndim + 1:
        if batch_first is True:
            array = array[0]
        elif batch_first is False:
            array = array[:, 0]
        elif batch_first is None:
            raise ValueError(
                'Remove "remove batch axis" is disabled '
                '(i.e. batch_first=None)\n'
                f'It looks like the array still has a batch axis.\n'
                f'Shape: {array.shape}'
            )
    else:
        raise ValueError(f'Either the signal has ndim {ndim} or {ndim + 1}',
                         array.shape)
    return array


def _apply_origin(image, origin):
    """
    When origin is 'lower' flip the top and bottom from the image

    Args:
        signal: Shape(..., features/y-axis, frames/x-axis)
        origin: 'upper' or 'lower' (For speech usually lower)

    Returns:
        Shape(..., features, frames)

    """
    assert origin in ['upper', 'lower'], origin
    if origin == 'lower':
        image = image[..., ::-1, :]
    return image


def mask_to_image(mask, batch_first=False, color=None, origin='lower'):
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

    return _colorize(_apply_origin(image.T, origin), color)


def stft_to_image(signal, batch_first=False, color='viridis', origin='lower'):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        signal: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]

    Returns: Shape(features, frames)

    """
    signal = to_numpy(signal, detach=True)

    return spectrogram_to_image(
        np.abs(signal), batch_first=batch_first, color=color, origin=origin,
    )


_spectrogram_to_image_cmap = {}


def _colorize(image, color):
    """
    >>> i = np.arange(15).reshape([3, 5])
    >>> _colorize(i, True).shape
    (4, 3, 5)
    >>> _colorize(i, None).shape
    (1, 3, 5)
    >>> i = np.arange(6).reshape([2, 3])
    >>> _colorize(i, True)
    array([[[0.267004, 0.26851 , 0.269944],
            [0.271305, 0.272594, 0.273809]],
    <BLANKLINE>
           [[0.004874, 0.009605, 0.014625],
            [0.019942, 0.025563, 0.031497]],
    <BLANKLINE>
           [[0.329415, 0.335427, 0.341379],
            [0.347269, 0.353093, 0.358853]],
    <BLANKLINE>
           [[1.      , 1.      , 1.      ],
            [1.      , 1.      , 1.      ]]])
    >>> _colorize(i, None)
    array([[[0, 1, 2],
            [3, 4, 5]]])
    """
    if color is None:
        return image[None, :, :]
    else:
        if color is True:
            color = 'viridis'
        try:
            cmap = _spectrogram_to_image_cmap[color]
        except KeyError:
            try:
                import matplotlib.pyplot as plt
                cmap = plt.cm.get_cmap(color)
                _spectrogram_to_image_cmap[color] = cmap
            except ImportError:
                from warnings import warn
                gray_scale = lambda x: x[None, ...]
                warn('Since matplotlib is not installed, all images are '
                     'switched to grey scale')
                _spectrogram_to_image_cmap[color] = gray_scale
                cmap = gray_scale
                # gray image
                # return gray_scale(image)
        return np.moveaxis(cmap(image), -1, 0)


def spectrogram_to_image(
        signal, batch_first=False, color='viridis', origin='lower'):
    """
        For more details of the output shape, see the tensorboardx docs
    Args:
        signal: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]
        color: A color map name. The name is forwarded to
               `matplotlib.pyplot.cm.get_cmap` to get the color map.


    Returns: Shape(features, frames)
    """
    signal = to_numpy(signal, detach=True)

    signal = signal / (np.max(signal) + np.finfo(signal.dtype).tiny)

    signal = _remove_batch_axis(signal, batch_first=batch_first)

    visible_dB = 50

    # remove problematic small numbers
    floor = 10 ** (-visible_dB / 20)
    signal = np.maximum(signal, floor)

    # Scale such that X dB are visible (i.e. in the range 0 to 1)
    signal = (20 / visible_dB) * np.log10(signal) + 1

    signal = (signal * 255).astype(np.uint8)

    return _colorize(_apply_origin(signal.T, origin=origin), color)


def audio(signal, sampling_rate: int = 16000, batch_first=False,
          normalize=True):
    """

    Args:
        signal: Shape (samples, batch [optional]). If `batch_first = True`,
            (batch [optional], samples).
        sampling_rate: Sampling rate of the audio signal
        batch_first: If `True`, the optional batch dimension is assumed to be
            the first axis, otherwise the second one.
        normalize: If `True`, the signal is normalized to a max amplitude of
            0.95 to prevent clipping
    """
    signal = to_numpy(signal, detach=True)

    signal = _remove_batch_axis(signal, batch_first=batch_first, ndim=1)

    # Normalize so that there is no clipping
    if normalize:
        denominator = np.max(np.abs(signal))
        if denominator > 0:
            signal = signal / denominator
            signal *= 0.95

    return signal, sampling_rate


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
            Dict of either one dimensional numpy arrays with the raw audio data
            with a sampling rate of 16k or tuples of length 2 with
            (audio data, sampling rate).
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
