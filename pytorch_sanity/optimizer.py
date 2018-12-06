from pytorch_sanity.parameterized import Parameterized
from pytorch_sanity.mapping import OPTIMIZER_MAP
import torch


class Optimizer(Parameterized):
    def __init__(self,
                 optimizer_name='adam',
                 *,
                 gradient_clipping=None,
                 lr=1e-3,
                 weight_decay=0,
                 adam_kwargs=None,
                 sdm_kwargs=None,
                 adagrad_kwargs=None):
        self.gradient_clipping = gradient_clipping
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_kwargs = adam_kwargs
        self.sdm_kwargs = sdm_kwargs
        self.adagrad_kwargs = adagrad_kwargs
        self.optimizer = None

    @classmethod
    def get_signature(cls):
        defaults = super().get_signature()
        defaults['gradient_clipping'] = 1e10
        defaults['adam_kwargs'] = dict(
            betas=(0.9, 0.999), eps=1e-8, amsgrad=False
        )
        defaults['sdm_kwargs'] = dict(
            momentum=0, dampening=0, nesterov=False
        )
        defaults['adagrad_kwargs'] = dict(
            lr_decay=0, initial_accumulator_value=0
        )

    def set_params(self, params):
        self.optimizer = OPTIMIZER_MAP[self.optimizer_name](
            params, lr=self.lr, weight_decay=self.weight_decay,
            **getattr(self, self.optimizer_name + '_kwargs'))

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
        # Todo: report clipped and unclipped
        # Todo: allow clip=None but still report grad_norm
        if isinstance(self.gradient_clipping, dict):
            grad_clips = self.gradient_clipping[prefix]
        else:
            grad_clips = self.gradient_clipping
        return torch.nn.utils.clip_grad_norm_(
            params, grad_clips
        )
