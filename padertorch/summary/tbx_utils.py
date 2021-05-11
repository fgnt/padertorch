import operator
from typing import Union, Optional, Tuple

import numpy as np
import torch

from padertorch.utils import to_numpy

_T_input = Union[torch.Tensor, np.ndarray]

__all__ = [
    'mask_to_image',
    'stft_to_image',
    'spectrogram_to_image',
    'review_dict',
    'audio',
    'figure',
    'figure_to_image',
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


def mask_to_image(
        mask: _T_input, batch_first: bool = False,
        color: Optional[str] = None,
        origin: str = 'lower'
) -> np.ndarray:
    """
    Creates an image from a mask `Tensor` or `ndarray`.

    For more details of the output shape, see the tensorboardx docs

    Note:
        Clips mask to range [0, 1]. Any values outside of this range will be
        ignored.

    Args:
        mask: Mask to plot
        batch_first: If `True`, `signal` is expected to have shape
            `(batch [optional], frames, features)`. If `False`, the batch axis
            is assumed to be in the second position, i.e.,
            `(frames, batch [optional], features)`.
        color: A color map name. The name is forwarded to
               `matplotlib.pyplot.cm.get_cmap` to get the color map. If `None`,
               grayscale is used.
        origin: Origin of the plot. Can be `'upper'` or `'lower'`.

    Returns:
        Colorized image with shape (color (1 or 3), features, frames)
    """
    mask = to_numpy(mask, detach=True)

    clipped_values = np.sum((mask < 0) | (mask > 1))
    if clipped_values:
        import warning
        warning.warn(
            f'Mask value passed to mask_to_image out of range ([0, 1])! '
            f'{clipped_values} values are clipped!'
        )

    image = np.clip(mask * 255, 0, 255)
    image = image.astype(np.uint8)

    image = _remove_batch_axis(image, batch_first=batch_first)

    return _colorize(_apply_origin(image.T, origin), color)


def stft_to_image(
        signal: _T_input,
        batch_first: bool = False,
        color: str = 'viridis',
        origin: str = 'lower',
        visible_dB: float = 50,
) -> np.ndarray:
    """
    Creates an image from an STFT signal.
    For more details of the output shape, see the tensorboardx docs

    Args:
        signal: Shape (frames, batch [optional], features)
        batch_first: if true mask shape (batch [optional], frames, features]
        color: A color map name. The name is forwarded to
               `matplotlib.pyplot.cm.get_cmap` to get the color map. If `None`,
               grayscale is used.
        origin: Origin of the plot. Can be `'upper'` or `'lower'`.
        visible_dB: How many dezibel are visible in the image.
                    Note: `paderbox.visualization.plot.stft` uses
                          `visible_dB == 60` internally. So by default it shows
                          10 dB more.

    Returns:
        Colorized image with shape (color (1 or 3), features, frames)


    Small test to see the effect of `visible_dB`:

        >>> visible_dB = 60
        >>> 10 ** (-visible_dB / 20)
        0.001

        >>> data = [1, 0.004, 0.003, 0.001_05, 0.001]
        >>> np.squeeze(stft_to_image(np.array(data)[:, None], color=None))
        array([255,  10,   0,   0,   0], dtype=uint8)

        >>> np.squeeze(stft_to_image(
        ...     np.array(data)[:, None], color=None, visible_dB=60))
        array([255,  51,  40,   1,   0], dtype=uint8)

    """
    signal = to_numpy(signal, detach=True)

    return spectrogram_to_image(
        signal.real ** 2 + signal.imag ** 2,
        batch_first=batch_first,
        color=color,
        origin=origin,
        visible_dB=visible_dB,
    )


class _Colorize:
    """
    >>> _colorize = _Colorize()
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
    >>> print(_colorize(i, None))
    [[[0 1 2]
      [3 4 5]]]
    >>> import mock
    >>> _colorize = _Colorize()
    >>> with mock.patch.dict('sys.modules', {'matplotlib.pyplot': None}):
    ...     print(_colorize(i, True))
    [[[0 1 2]
      [3 4 5]]]
    """

    def __init__(self):
        self.color_to_cmap = {}

    def __call__(self, image, color):
        if color is None:
            return image[None, :, :]
        else:
            if color is True:
                color = 'viridis'
            try:
                cmap = self.color_to_cmap[color]
            except KeyError:
                try:
                    import matplotlib.pyplot as plt
                    cmap = plt.cm.get_cmap(color)
                    self.color_to_cmap[color] = cmap
                except ImportError:
                    from warnings import warn
                    gray_scale = lambda x: x[..., None]
                    warn('Since matplotlib is not installed, all images are '
                         'switched to grey scale')
                    self.color_to_cmap[color] = gray_scale
                    cmap = gray_scale
            return np.moveaxis(cmap(image), -1, 0)


_colorize = _Colorize()


def spectrogram_to_image(
        signal: _T_input,
        batch_first: bool = False,
        color: str = 'viridis',
        origin: str = 'lower',
        log: bool = True,
        visible_dB: float = 50,
) -> np.ndarray:
    """
    Creates an image from a spectrogram (power).

    Note:
        When The input is the absolute value of the STFT, the value for
        visible_dB is effectively two times larger (i.e. default 100) and
        the image looks more noisy.

    For more details of the output shape, see the tensorboardx docs

    Args:
        signal: Spectrogram to plot.
        batch_first: If `True`, `signal` is expected to have shape
            `(batch [optional], frames, features)`. If `False`, the batch axis
            is assumed to be in the second position, i.e.,
            `(frames, batch [optional], features)`.
        color: A color map name. The name is forwarded to
               `matplotlib.pyplot.cm.get_cmap` to get the color map.
        origin: Origin of the plot. Can be `'upper'` or `'lower'`.
        log: If `True`, the spectrogram is plotted in log domain and shows a
            50dB range. The 50dB can be changed with the argument `visible_dB`.
        visible_dB: Only used when `log` is `True`. Specifies how many dB will
            be visible in the plot. Assumes the input is the power of the STFT
            signal, i.e., the abs square of it.

    Returns:
        Colorized image with shape (channels (3), features, frames)

    """
    signal = to_numpy(signal, detach=True)

    signal = signal / (np.max(np.abs(signal)) + np.finfo(signal.dtype).tiny)

    signal = _remove_batch_axis(signal, batch_first=batch_first)

    if log:
        # remove problematic small numbers
        floor = 10 ** (-visible_dB / 10)
        signal = np.maximum(signal, floor)

        # Scale such that X dB are visible (i.e. in the range 0 to 1)
        signal = (10 / visible_dB) * np.log10(signal) + 1

    signal = (signal * 255).astype(np.uint8)

    return _colorize(_apply_origin(signal.T, origin=origin), color)


def audio(
        signal: _T_input,
        sampling_rate: int = 16000,
        batch_first: bool = False,
        normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Adds an audio signal to tensorboard.

    Args:
        signal: Time-domain signal with shape (samples, batch [optional]).
            If `batch_first = True`, (batch [optional], samples).
        sampling_rate: Sampling rate of the audio signal
        batch_first: If `True`, `signal` is expected to have shape
            `(batch [optional], samples)`. If `False`, the batch axis
            is assumed to be in the second position, i.e.,
            `(samples, batch [optional])`.
        normalize: If `True`, the signal is normalized to a max amplitude of
            0.95 to prevent clipping.

    Returns:
        A tuple consisting of the signal and the sampling rate. See tensorboardX
        docs for further information on the return type.
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


def figure_to_image(
        fig: 'matplotlib.figure.Figure' = None,
        close=True,
) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array that can be handled by
    tensorboardX.

    Uses `tensorboardX.utils.figure_to_image` with some sanity checks. It works
    even without explicitly switching to the 'Agg' backend before using this
    function.

    Notes:
        Use this function only if you want to convert the figure to an image
        manually and don't want to wait for tensorboardX to convert it (e.g., if
        you have a very large number of points that would take up too much
        memory). Otherwise, use the `padertorch.summary.figure` function and
        pass it under the `figures` key in the review dict.

        Matplotlib is in general quite slow, so you shouldn't be computing
        matplotlib plots in every iteration. Use the `pt.Model.create_snapshot`
        flag to reduce the computational overhead.

    Warnings:
        Make sure that you close unused plots to reduce memory consumption by
        either closing the plot manually or setting `close=True` (default)!

    Args:
        fig: matplotlib figure object. If `None`, it defaults to `plt.gcf()`.
        close: If `True`, closes the figure.

    Returns:
        Numpy array with shape (color (4), height, width)

    Examples:
        >>> from matplotlib import pyplot as plt
        >>> plt.plot([1, 3, 2, 4])  # doctest: +ELLIPSIS
        [<matplotlib.lines.Line2D object ...>]
        >>> image = figure_to_image()
        >>> image.shape
        (3, 480, 640)
    """
    from tensorboardX.utils import figure_to_image as tbX_figure_to_image
    return tbX_figure_to_image(figure(fig, close=close), close=False)


def figure(
        fig: 'matplotlib.figure.Figure' = None,
        close=True,
) -> 'matplotlib.figure.Figure':
    """
    Checks a figure to be passed on to tensorboardX.

    Makes sure that:
        - The figure is a figure (i.e., subclass of `matplotlib.figure.Figure`)
        - The figure is not empty (i.e., contains at least one axis)

    Args:
        fig: The figure to plot. This is directly passed to tensorboardX.
            Defaults to `plt.gcf()` if `None`.
        close: If `True`, closes the figure to prevent modifications.

    Returns:
        The input figure
    """
    from matplotlib.figure import Figure
    if fig is None:
        from matplotlib import pyplot as plt
        fig = plt.gcf()

    assert isinstance(fig, Figure), fig
    assert len(fig.axes) > 0, (
        'Empty plot detected. You probably wanted to plot something.'
    )

    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return fig


def review_dict(
        *,
        loss: torch.Tensor = None,
        losses: dict = None,
        scalars: dict = None,
        histograms: dict = None,
        audios: dict = None,
        images: dict = None,
        figures: dict = None,
        texts: dict = None,
):
    """
    This is a helper function to build the review dict.
    The main purpose is for auto completion of the review dict, prevent typos
    and documentation what is expected for the values.

    Every argument to this function is a `dict`. Its keys represent the tags
    used in tensorboard and its values the values displayed.

    Args:
        loss:
            Scalar `torch.Tensor`s. If not `None`, expect `losses` to be `None`.
        losses:
            `dict` of scalar `torch.Tensor`s. If not `None`, expect `loss` to be
            `None`.
        scalars:
            `dict` of scalars that are reported to tensorboard. `losses` and
            `loss` are also reported as scalars.
        histograms:
            `dict` of scalars, `list`s, `tuple`s `torch.Tensor`s or
            `np.ndarray`s of any shape to build a histogram from. Any
            multi-dimensional tensors or arrays are flattened before a histogram
            is created from them.
        audios:
            `dict` of audio data. The audio data can have two formats:
                - A one dimensional numpy array with raw audio data at a
                    sampling rate of 16kHz
                - A `tuple` of length 2 with (audio data, sampling rate). Use
                    `padertorch.summary.audio` to create these tuples.
        images:
            `dict` of `torch.Tensor`s or `np.ndarray`s with shape
                (color (1 or 3), height, width).
        figures:
            `dict` of `matplotlib.figure.Figure`s.
        texts:
            `dict` of `str`.

    Returns:
        `dict` of the args that are not `None`
    """

    review = locals()

    for k, v in list(review.items()):
        if v is None:
            del review[k]

    assert operator.xor(loss is None, losses is None), (loss, losses)

    return review
