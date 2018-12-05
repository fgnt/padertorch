import torch
from functools import partial


def sequence_elementwise(function, x, *args, **kwargs):
    """Expects the desired function and a `Tensor` or `PackedSequence`."""
    if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        return torch.nn.utils.rnn.PackedSequence(
            function(x.data, *args, **kwargs),
            x.batch_sizes
        )
    else:
        return function(x, *args, **kwargs)


abs = partial(sequence_elementwise, torch.abs)
ceil = partial(sequence_elementwise, torch.ceil)
clamp = partial(sequence_elementwise, torch.clamp)
exp = partial(sequence_elementwise, torch.exp)
log = partial(sequence_elementwise, torch.log)
log10 = partial(sequence_elementwise, torch.log10)
log1p = partial(sequence_elementwise, torch.log1p)
sigmoid = partial(sequence_elementwise, torch.sigmoid)
sqrt = partial(sequence_elementwise, torch.sqrt)
