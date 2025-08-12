from typing import Union, List, Optional

import numpy as np

from paderbox.visualization import plot

import padertorch
from padertorch.utils import to_list
from padertorch.contrib.je.modules.conv_utils import compute_pad_size


def compute_receptive_field_region(
    kernel_sizes, *,
    dilations=1,
    strides=1,
    pad_types='both',
    center_index=0,
    is_transpose=False,
):
    """

    [1]: https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region
    """
    dilations = to_list(dilations, len(kernel_sizes))
    strides = to_list(strides, len(kernel_sizes))
    pad_types = to_list(pad_types, len(kernel_sizes))

    padding = list(map(
        lambda x: compute_pad_size(*x),
        zip(kernel_sizes, dilations, strides, pad_types)
    ))

    # Compute receptive field regions according to (5) and (6) [1]
    padding = np.array(padding)[:, 0]
    strides = np.array(strides)
    kernel_sizes = np.array(kernel_sizes)

    if is_transpose:
        strides = 1 / strides
        padding = padding * (strides >= 1)  # No padding when upsampling
        kernel_sizes[strides < 1] = 1  # Kernel has no effect on receptive field when upsampling
    else:
        strides = np.array(np.concatenate(([1], strides[:-1])))
    total_stride = np.prod(strides)
    receptive_field_start = np.math.floor(
        center_index * total_stride - np.sum(padding * np.cumprod(strides))
    )
    receptive_field_stop = np.math.floor(
        center_index * total_stride
        - np.sum((1 + padding - kernel_sizes) * np.cumprod(strides))
    )
    return receptive_field_start, receptive_field_stop


def plot_receptive_field_at_time_index(
    signal: np.ndarray, time_index: int,
    cnn: "padertorch.contrib.je.modules.conv.CNN1d",
    *,
    cnn_transpose:\
        Optional["padertorch.contrib.je.modules.conv.CNNTranspose1d"] = None,
    plot_fn=plot.spectrogram,
    ax=None, **kwargs
):
    """
    Plot the receptive field of a single time index at the output of CNN1d.

    Args:
        signal: Signal at the input of CNN of shape (t, ...).
        time_index: Time index at the output of the CNN for which to plot
            the receptive field of the input.
        cnn: padertorch.contrib.je.modules.conv.CNN1d instance.
        plot_fn: Function to plot the signal. Defaults to
            paderbox.visualiazation.plot.spectrogram.
        ax: Axis for plotting. Defaults to None.
        kwargs: Passed to plot_fn.
    """
    # receptive_field = cnn.get_receptive_field()
    if cnn.is_transpose():
        raise NotImplementedError(
            'Single transposed CNNs are not supported!'
            'Pass it via cnn_transpose=... with a matching CNN1d instance.'
        )
    if cnn_transpose is not None:
        start, stop = compute_receptive_field_region(
            cnn_transpose.kernel_sizes,
            dilations=cnn_transpose.dilations,
            strides=cnn_transpose.strides,
            pad_types=cnn_transpose.pad_types,
            center_index=time_index,
            is_transpose=True,
        )
        start = np.maximum(start, 0)
        stop = np.minimum(stop, signal.shape[0])
        _, start_u = compute_receptive_field_1d(
            kernel_sizes=cnn.kernel_sizes,
            dilations=cnn.dilations,
            strides=cnn.strides,
            pad_types=cnn.pad_types,
            center_index=start,
        )
        rfield_v, start_v = compute_receptive_field_1d(
            kernel_sizes=cnn.kernel_sizes,
            dilations=cnn.dilations,
            strides=cnn.strides,
            pad_types=cnn.pad_types,
            center_index=stop,
        )
        start = start_u
        receptive_field = start_v + rfield_v - start
    else:
        receptive_fields, start = compute_receptive_field_1d(
            kernel_sizes=cnn.kernel_sizes,
            dilations=cnn.dilations,
            strides=cnn.strides,
            pad_types=cnn.pad_types,
            center_index=time_index,
        )
        receptive_field = receptive_fields[0]
    stop = np.minimum(start + receptive_field, signal.shape[0])
    start = np.maximum(start, 0)
    plot_sig = signal[start:stop]
    plot_fn(plot_sig, ax=ax, **kwargs)
    return start, stop
