import torch
import padertorch as pt
from padertorch.ops.reduction import sequence_reduction
from torch.nn.utils.rnn import PackedSequence

class Normalization(pt.Module):
    """ Computes lp-statistics of a sequence
    :param num_features: number of features of weight and bias
    :param order: Order of the normalization. Can be l1 or l2. For l1, the
        tensor is normalized by \\(\\frac{\\gamma(x-\\mu)}{}\\)
    :param statistics_axis: axis along which to compute the statics.
    :param independent_axis: int, tuple of ints specifying independent
        dimensions. Determines shape of the kernel for
        shift and scale operations.
    :param norm_epsilon: prevents division by 0
    :param keep_dims: keep the number of dimensions
    :param affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True
            and always uses batch statistics in both training and eval modes.
    :return: tuple of lp-statistics
    """
    def __init__(self, num_features=None, order='l2', statistics_axis=0,
                 independent_axis=-1, norm_epsilon=1e-5, affine=True):
        assert num_features is not None
        super().__init__()
        self.num_features = num_features
        self.norm_epsilon = norm_epsilon
        self.affine = affine
        self.statistics_axis = statistics_axis
        self.independent_axis = independent_axis
        self.norm_epsilon = norm_epsilon
        self.order = order
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def _check_input_dim(self, input):
        assert input.ndim >= self.statistics_axis

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.uniform_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_statistics(self, tensor):
        if self.order == 'l1':
            mean = sequence_reduction(
                torch.mean,
                tensor,
                axis=self.statistics_axis,
                keepdims=True
            )
            norm = sequence_reduction(
                torch.mean,
                torch.abs(tensor),
                axis=self.statistics_axis,
                keepdims=True
            )
        elif self.order in ['l2', 'mean']:
            mean = sequence_reduction(
                torch.mean,
                tensor,
                axis=self.statistics_axis,
                keepdims=True
            )
            norm = sequence_reduction(
                torch.mean,
                tensor - mean,
                axis=self.statistics_axis,
                keepdims=True
            )
        else:
            raise ValueError(f'chosen order {self.order} in is not'
                             f' known in {self}')

        norm += self.norm_epsilon
        if self.order == 'l2':
            norm = pt.ops.pointwise.sqrt(norm)
        if self.order == 'mean':
            norm = None

        return mean, norm

    def forward(self, tensor):
        if isinstance(tensor, PackedSequence):
            raise NotImplementedError
        mean, norm = self.get_statistics(tensor)
        if mean is not None:
            tensor -= mean
        if norm is not None:
            tensor /= norm
        if self.affine:
            dims = tensor.ndim * [1]
            dims[self.independent_axis] = -1
            tensor = (tensor + self.bias.view(dims)) * self.weight.view(dims)
        return tensor
