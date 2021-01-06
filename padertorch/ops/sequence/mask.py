import torch
from typing import Union, List


def compute_mask(x: torch.Tensor,
                 sequence_lengths: Union[List[int], int] = None,
                 batch_axis: int = 0, sequence_axis: int = 1):
    """
    This function calculates a mask which indicates the position
    of non-padded values.
    It can be used to do subsequent operations only on non-padded values.

    >>> x, seq_len = 2*torch.ones((3,1,10,4)), [1, 2, 3]
    >>> mask = compute_mask(x, sequence_lengths=seq_len,
    ...                     batch_axis=0, sequence_axis=-1)
    >>> mask[:,0]
    tensor([[[1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.],
             [1., 0., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 0.]]])
    >>> x, seq_len = 2*torch.ones((2,1,10,4)), 2
    >>> mask = compute_mask(x, sequence_lengths=seq_len,
    ...                     batch_axis=0, sequence_axis=-1)
    >>> mask[:,0]
    tensor([[[1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.]],
    <BLANKLINE>
            [[1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 0., 0.]]])

    Args:
        x: tensor to be masked
        sequence_lengths: int or list of ints stating sequence length
            for each sequence in the mini-batch.
            A single integer will be broadcasted and used for all sequences
            If None a one-mask is returned, i.e., no values in x are masked.
        batch_axis: axis along which sequences are stacked
        sequence_axis: axis which may contain padding (of different lengths
            for each sequence)

    Returns:

    """
    if sequence_lengths is None:
        return torch.ones_like(x)
    elif isinstance(sequence_lengths, int):
        sequence_lengths = [sequence_lengths] * x.shape[batch_axis]
    if batch_axis < 0:
        batch_axis = x.dim() + batch_axis
    if sequence_axis < 0:
        sequence_axis = x.dim() + sequence_axis
    sequence_lengths = torch.Tensor(sequence_lengths).long().to(x.device)
    for dim in range(batch_axis + 1, x.dim()):
        sequence_lengths = sequence_lengths.unsqueeze(-1)
    idx = torch.arange(x.shape[sequence_axis]).to(x.device)
    for dim in range(sequence_axis + 1, x.dim()):
        idx = idx.unsqueeze(-1)
    mask = (idx < sequence_lengths).float().expand(x.shape)
    return mask
