import torch
from pytorch_sanity.parameterized import Parameterized
from torch import optim


class Optimizer(Parameterized):
    optimizer_cls = None
    optimizer = None

    def __init__(self, gradient_clipping):
        self.gradient_clipping = gradient_clipping

    def set_parameter(self, parameters):
        self.optimizer = self.optimizer_cls(
            parameters, **self.optimizer_kwargs
        )

    def check_if_set(self):
        assert self.optimizer is not None, \
            'The optimizer is not initialized, call set_params before' \
            ' using any of the optimizer functions'

    def zero_grad(self):
        self.check_if_set()
        return self.optimizer.zero_grad()

    def step(self):
        self.check_if_set()
        return self.optimizer.step()

    def clip_grad(self, params, prefix: str = None):
        self.check_if_set()
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm
        if isinstance(self.gradient_clipping, dict):
            grad_clips = self.gradient_clipping[prefix]
        else:
            grad_clips = self.gradient_clipping
        return torch.nn.utils.clip_grad_norm_(
            params, grad_clips
        )

    def cuda(self, device):
        self.check_if_set()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(device)

    def cpu(self):
        self.check_if_set()
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

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
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False
    ):
        super().__init__(gradient_clipping)
        self.optimizer_kwargs = dict(
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
            lr=1e-3,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False
    ):
        super().__init__(gradient_clipping)
        self.optimizer_kwargs = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
