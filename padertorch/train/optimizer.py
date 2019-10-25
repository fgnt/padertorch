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
            assert self.swa_freq is not None, self.swa_freq
            assert self.swa_lr is not None, self.swa_lr
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

    def load_state_dict(self, state_dict):
        self.check_if_set()
        return self.optimizer.load_state_dict(state_dict)

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


class LRScheduler:
    """
     Basic class for LRScheduler, can be given to the Trainer and will be
     applied during evaluation.
     Does not work if no validation_iterator is supplied in Trainer.train.
    """

    lr_scheduler_cls = None
    lr_scheduler = None

    def __init__(self, use_metrics=False, metric_key=None):

        self.scheduler_kwargs = dict()
        self.use_metrics = use_metrics
        assert not (use_metrics and metric_key is None)
        self.metric_key = metric_key

    def set_optimizer(self, optimizer_wrapper):
        assert isinstance(optimizer_wrapper, Optimizer), optimizer_wrapper
        assert optimizer_wrapper.swa_start is None, optimizer_wrapper
        optimizer_wrapper.check_if_set()
        self.optimizer = optimizer_wrapper.optimizer
        self.lr_scheduler = self.lr_scheduler_cls(
            self.optimizer, **self.scheduler_kwargs)

    def step(self, metrics, epoch):
        if self.use_metrics:
            if epoch == 0:
                return
            self.lr_scheduler.step(metrics[self.metric_key], epoch)
        else:
            self.lr_scheduler.step(epoch=epoch)

    def load_state_dict(self, state_dict):
        return self.lr_scheduler.load_state_dict(state_dict)

    def state_dict(self):
        return self.lr_scheduler.state_dict()


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.
    """
    from torch.optim.lr_scheduler import ExponentialLR
    lr_scheduler_cls = ExponentialLR

    def __init__(self, gamma=0.5, last_epoch=-1):
        super().__init__(use_metrics=False)

        self.scheduler_kwargs = dict(gamma=gamma, last_epoch=last_epoch)


class StepLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every
    step_size epochs. Notice that such decay can happen simultaneously with
    other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.
    """
    from torch.optim.lr_scheduler import StepLR
    lr_scheduler_cls = StepLR

    def __init__(self, step_size=5, gamma=0.1, last_epoch=-1):
        super().__init__(use_metrics=False)

        self.scheduler_kwargs = dict(step_size=step_size, gamma=gamma,
                                     last_epoch=last_epoch)


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a ‘patience’ number
    of epochs, the learning rate is reduced.
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    lr_scheduler_cls = ReduceLROnPlateau

    def __init__(self, metric_key='loss', mode='min',
                 factor=0.1, patience=1, verbose=True, threshold=1e-4,
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(use_metrics=True, metric_key=metric_key)

        self.scheduler_kwargs = dict(
            mode=mode, factor=factor, patience=patience, verbose=verbose,
            threshold=threshold, threshold_mode=threshold_mode,
            cooldown=cooldown, min_lr=min_lr, eps=eps
        )
