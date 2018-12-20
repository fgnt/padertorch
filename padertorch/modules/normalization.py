import torch
import padertorch as pt


class Normalization(pt.Module):
    """ Computes lp-statistics of a sequence
    :param num_features: number of features of weight and bias
    :param order: Order of the normalization. Can be l1 or l2. For l1, the
        tensor is normalized by \\(\\frac{\\gamma(x-\\mu)}{}\\)
    :param statistics_axis: axis along which to compute the statics.
    :param norm_epsilon: prevents division by 0
    :param keep_dims: keep the number of dimensions
    :param momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
    :param affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True
    :param track_running_stats: a boolean value that when set to ``True``,
            this module tracks the running mean and variance, and when set
            to ``False``, this module does not track such statistics
            and always uses batch statistics in both training and eval modes.
    :return: tuple of lp-statistics
    """
    def __init__(self, num_features=None, order='l2', data_format='tbf',
                 statistics_axis='t',
                 norm_epsilon=1e-5, keep_dims=False, affine=True):
        assert num_features is not None
        super().__init__()
        self.num_features = num_features
        self.norm_epsilon = norm_epsilon
        self.affine = affine
        self.statistics_axis = statistics_axis
        self.norm_epsilon = norm_epsilon
        self.keep_dims = keep_dims
        self.order = order
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_frames_tracked',
                                 torch.Tensor(0, dtype=torch.LongTensor))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def _check_input_dim(self, input):
        assert input.ndim >= self.statistics_axis

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            torch.nn.init.uniform_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, tensor):
        # expects last dimension of to be the feature dimension
        if isinstance(tensor, torch.nn.utils.rnn.PackedSequence):
            inputs = pt.ops.unpack_sequence(tensor)
            packed = True
        elif isinstance(tensor, torch.Tensor):
            input = [tensor]
            packed = False
        else:
            raise ValueError()
        ndim = len(tensor[0].ndim)
        assert ndim <= self.statistics_axis, f'ndim {ndim} <= len("{self.statistics_axis}")'
        output =  [self.normalize(x) for x in tensor]
        if packed:
            output = pt.ops.pack_sequence(output)
        return output

    def get_statistics(self, tensor):
        if self.opts.order == 'l1':
            mean = torch.mean(tensor, dim=self.opts.statistics_axis,
                              keepdims=self.opts.keepdims)
            norm = torch.mean(torch.abs(tensor), self.optsstatistics_axis,
                              keepdims=self.opts.keepdims)
        elif self.opts.order in ['l2', 'mean']:
            mean = torch.mean(tensor, dim=self.opts.statistics_axis,
                              keepdims=self.opts.keepdims)
            norm = torch.mean(tensor, self.optsstatistics_axis,
                              keepdims=self.opts.keepdims)
        else:
            raise ValueError(f'chosen order {self.opts.order} in is not'
                             f' known in {self}')

        norm += self.opts.norm_epsilon
        if self.opts.order == 'l2':
            norm = torch.sqrt(norm)
        if self.opts.order == 'mean':
            norm = None

        return mean, norm

    def normalize(self, tensor):
        mean, norm = self.get_statistics(tensor)
        if mean is not None:
            tensor -= mean
        if norm is not None:
            tensor /= norm
        if self.affine:
            dims = (tensor.ndim - 1) * [1]
            tensor = (tensor + self.bias.view(*dims,-1)
                      ) * self.weight.view((*dims,-1))
        return tensor
