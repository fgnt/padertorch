import torch


def overlap_add(
        tensor,
        shift,
):
    """

    >>> overlap_add(torch.arange(12).to(torch.float).reshape(3, 4), 4)
    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
    >>> overlap_add(torch.arange(12).to(torch.float).reshape(3, 4), 2)
    tensor([ 0.,  1.,  6.,  8., 14., 16., 10., 11.])
    >>> overlap_add(torch.ones(12).to(torch.float).reshape(3, 4), 2)
    tensor([1., 1., 2., 2., 2., 2., 1., 1.])
    >>> overlap_add(torch.ones(2, 3, 4, 5).to(torch.float), 2).shape
    torch.Size([2, 3, 11])
    """
    *independent, frames, frequencies = tensor.shape

    samples = frequencies + frames * shift - shift
    tensor = tensor.reshape(-1, frames, frequencies).transpose(-1, -2)
    out = torch.nn.Fold(
        output_size=(1, samples),
        kernel_size=(1, frequencies),
        dilation=1,
        padding=0,
        stride=(1, shift),
    )(tensor)
    return out.squeeze(-3).squeeze(-2).reshape(*independent, samples)
