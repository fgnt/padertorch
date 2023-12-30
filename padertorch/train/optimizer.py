import itertools
import torch
from torch import optim


class Optimizer:
    optimizer_cls = None
    optimizer = None
    parameters = None

    def __init__(
            self, gradient_clipping, **kwargs
    ):
        self.gradient_clipping = gradient_clipping
        self.optimizer_kwargs = kwargs

    def set_parameters(self, parameters):
        self.parameters = tuple(parameters)
        self.optimizer = self.optimizer_cls(
            self.parameters, **self.optimizer_kwargs
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

    def clip_grad(self):
        self.check_if_set()
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm
        grad_clips = self.gradient_clipping
        if isinstance(self.parameters[0], dict) and 'params' in self.parameters[0]:
            params = itertools.chain(*[param_group['params'] for param_group in self.parameters])
        else:
            params = self.parameters
        return torch.nn.utils.clip_grad_norm_(params, grad_clips)

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
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False
    ):
        super().__init__(
            gradient_clipping,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


class AdamW(Optimizer):
    optimizer_cls = optim.AdamW

    def __init__(
            self,
            gradient_clipping=1e10,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False
    ):
        super().__init__(
            gradient_clipping,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


class Adadelta(Optimizer):
    optimizer_cls = optim.Adadelta

    def __init__(
            self,
            gradient_clipping=1e10,
            lr=1.0,
            rho=0.9,
            eps=1e-6,
            weight_decay=0
    ):
        super().__init__(
            gradient_clipping,
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )


class SGD(Optimizer):
    optimizer_cls = optim.SGD

    def __init__(
            self,
            gradient_clipping=1e10,
            lr=1e-3,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False
    ):
        super().__init__(
            gradient_clipping,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
