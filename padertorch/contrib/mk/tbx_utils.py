import typing

import numpy as np
from padertorch.utils import to_numpy
from padertorch.summary.tbx_utils import spectrogram_to_image
import torch
from torch import Tensor
from torchvision.utils import make_grid


def tensor_to_image(signal: Tensor, input_type: str):
    x = to_numpy(signal, detach=True)
    if input_type == 'image':
        x = (x * 255).astype(np.uint8)
    elif input_type == 'spectrogram':
        x = spectrogram_to_image(
            x.transpose(-1, -2), batch_first=None, log=False
        )
    else:
        raise ValueError(f'Unknown input type {input_type}')
    return x


def batch_image_to_grid(
    batch_image: torch.Tensor,
    input_shape_format: str = 'b c h w',
    height_axis: str = 'h',
    width_axis: str = 'w',
    stack: typing.Optional[str] = None,
    origin: str = 'upper',
    normalize: bool = True,
    scale_each: bool = False,
):
    """
    >>> batch_image = torch.rand(4, 3, 32, 32)
    >>> grid = batch_image_to_grid(batch_image)
    >>> grid.shape
    torch.Size([3, 138, 36])
    >>> grid = batch_image_to_grid(\
            torch.rand(4, 32, 32),\
            input_shape_format='b h w'\
        )
    >>> grid.shape
    torch.Size([138, 36])

    Args:
        batch_image: Batched images of shape (batch, channel, heigth, width) or
            (batch, height, width).
        input_shape_format: Format of the input shape. Should be a string of
            space-separated dimension names, e.g., 'b c h w'.
        height_axis: Name of the height (frequency) axis.
        width_axis: Name of the width (time) axis.
        stack: How to stack the images. `height_axis` for horizontal,
            `width_axis` for vertical stacking.
        origin: Origin of the plot. Can be `'upper'` or `'lower'`.
        normalize: See make_grid
        scale_each: See make_grid
    """
    if origin not in ('upper', 'lower'):
        raise ValueError(f'"origin" should be "upper" or "lower" but got {origin}')

    if stack is None:
        stack = height_axis

    if stack not in (height_axis, width_axis):
        raise ValueError(
            f'"stack" should be "{height_axis}" or '
            f'"{width_axis}" but got {stack}'
        )

    dims = input_shape_format.split()
    if len(dims) != batch_image.ndim:
        raise ValueError(f'Shape format {input_shape_format} does not match input shape {batch_image.shape}')

    if batch_image.ndim == 3:
        # Add channel dimension
        batch_image = batch_image.unsqueeze(1)
        dims.insert(1, 'c')

    if origin == 'lower':
        # Reverse the order of the height (frequency) dimension
        batch_image = batch_image.flip(dims.index(height_axis))

    grid = make_grid(
        batch_image,
        normalize=normalize,
        scale_each=scale_each,
        nrow=1 if stack==height_axis else batch_image.shape[0],
    )
    if batch_image.shape[1] == 1:
        # Remove color dimension
        grid = grid[0]
    return grid
