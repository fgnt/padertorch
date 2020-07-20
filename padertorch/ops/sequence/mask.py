import torch


def compute_mask(x, sequence_lengths, batch_axis=0, sequence_axis=1):
    """
    >>> x, seq_len = 2*torch.ones((3,1,10,4)), [1, 2, 3]
    >>> mask = compute_mask(x, sequence_lengths=seq_len, batch_axis=0, sequence_axis=-1)
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

    Args:
        x:
        sequence_lengths:
        batch_axis:
        sequence_axis:

    Returns:

    """
    if sequence_lengths is None:
        return torch.ones_like(x)
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
