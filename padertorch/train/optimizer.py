import torch
from torch import optim
from padertorch.data import example_to_device


class Optimizer:
    optimizer_cls = None
    optimizer = None
    parameters = None

    def __init__(
            self, gradient_clipping, swa_start=None, swa_freq=None, swa_lr=None,
            **kwargs
    ):
        self.gradient_clipping = gradient_clipping
        self.optimizer_kwargs = kwargs
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr

    def set_parameters(self, parameters):
        self.parameters = tuple(parameters)
        self.optimizer = self.optimizer_cls(
            self.parameters, **self.optimizer_kwargs
        )
        if self.swa_start is not None:
            from torchcontrib.optim import SWA
            assert self.swa_freq is not None
            assert self.swa_lr is not None
            self.optimizer = SWA(
                self.optimizer, swa_start=self.swa_start,
                swa_freq=self.swa_freq, swa_lr=self.swa_lr
            )

    def check_if_set(self):
        assert self.optimizer is not None, \
            'The optimizer is not initialized, call set_parameter before' \
            ' using any of the optimizer functions'

    def zero_grad(self):
        self.check_if_set()
        return self.optimizer.zero_grad()

    def step(self):
        self.check_if_set()
        return self.optimizer.step()

    def swap_swa_sgd(self):
        self.check_if_set()
        from torchcontrib.optim import SWA
        assert isinstance(self.optimizer, SWA), self.optimizer
        return self.optimizer.swap_swa_sgd()

    def clip_grad(self):
        self.check_if_set()
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm
        grad_clips = self.gradient_clipping
        return torch.nn.utils.clip_grad_norm_(
            self.parameters, grad_clips
        )

    def to(self, device):
        if device is None:
            return
        self.check_if_set()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        assert device is None or isinstance(device, int), device
        if device is None:
            device = torch.device('cuda')
        return self.to(device)

    def load_state_dict(self, key):
        self.check_if_set()
        return self.optimizer.load_state_dict(key)

    def state_dict(self):
        self.check_if_set()
        return self.optimizer.state_dict()


class Adam(Optimizer):
    optimizer_cls = optim.Adam

    def __init__(
            self,
            gradient_clipping=1e10,
            swa_start=None,
            swa_freq=None,
            swa_lr=None,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False
    ):
        super().__init__(
            gradient_clipping,
            swa_start=swa_start,
            swa_freq=swa_freq,
            swa_lr=swa_lr,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


class SGD(Optimizer):
    optimizer_cls = optim.SGD

    def __init__(
            self,
            gradient_clipping=1e10,
            swa_start=None,
            swa_freq=None,
            swa_lr=None,
            lr=1e-3,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False
    ):
        super().__init__(
            gradient_clipping,
            swa_start=swa_start,
            swa_freq=swa_freq,
            swa_lr=swa_lr,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )


def batch_norm_update(
        model, dataset, feature_key, batch_dim=0,
        device=0 if torch.cuda.is_available() else 'cpu'
):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        dataset: dataset to compute the activation statistics on.
            Each data batch should be either a dict, or a list/tuple.

        model: model for which we seek to update BatchNorm statistics.

        feature_key: key to get an input tensor to read batch_size from

        device: If set, data will be transferred to :attr:`device`
            before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()

    model.to(device)

    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    with torch.no_grad():
        for i, example in enumerate(dataset):
            b = example[feature_key].size(batch_dim)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            example = example_to_device(example, device)
            model(example)

            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
