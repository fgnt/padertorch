import typing as tp

import numpy as np
from padertorch.contrib.je.modules.conv_utils import compute_pad_size
from padertorch.utils import to_list


def compute_receptive_field_1d(
    kernel_sizes, *,
    dilations: tp.Union[int, tp.List[int] ] =1,
    strides: tp.Union[int, tp.List[int]] = 1,
    pad_types='both',
    pool_sizes=1,
    pool_strides=1,
    center_index=None,
):
    """Compute the receptive field size for a 1D convolutional neural network.

    [1]: https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region
    [2]: https://www.baeldung.com/cs/cnn-receptive-field-size
    """
    dilations = to_list(dilations, len(kernel_sizes))
    strides = to_list(strides, len(kernel_sizes))
    pad_types = to_list(pad_types, len(kernel_sizes))
    pool_sizes = to_list(pool_sizes, len(kernel_sizes))
    pool_strides = to_list(pool_strides, len(kernel_sizes))

    padding = list(map(
        lambda x: compute_pad_size(*x),
        zip(kernel_sizes, dilations, strides, pad_types)
    ))

    receptive_fields = []
    _stride = 1
    r = 0

    # Compute receptive field size according to recursion formula (18) [1]
    for k, d, s, pk, ps, in zip(
        kernel_sizes, dilations, strides, pool_sizes, pool_strides
    ):
        k = d * (k - 1) + 1
        r = r + _stride * (k - 1)
        receptive_fields.insert(0, r + 1)
        _stride = _stride * s
        # Support pooling which behaves like a kernel + stride
        if pk > 1:
            r = r + _stride * (pk - 1)
            receptive_fields[0] = r
            _stride = _stride * ps

    if center_index is None:
        return receptive_fields

    # Compute receptive field start according to recursion formula (22) [1]
    strides = np.array(np.concatenate(([1], strides[:-1])))
    padding = np.array(padding)[:, 0]
    receptive_field_start = (
        center_index * _stride - np.sum(padding * np.cumprod(strides))
    )

    return receptive_fields, receptive_field_start
