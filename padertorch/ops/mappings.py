import torch
from torch import optim
from paderbox.utils.mapping import Dispatcher
import numpy as np

__all__ = [
    'ACTIVATION_FN_MAP',
    'POOLING_FN_MAP',
    'DTYPE_MAP',
]

ACTIVATION_FN_MAP = Dispatcher(
    relu=torch.nn.ReLU,
    prelu=torch.nn.PReLU,
    leaky_relu=torch.nn.LeakyReLU,
    elu=torch.nn.ELU,
    tanh=torch.nn.Tanh,
    sigmoid=torch.nn.Sigmoid,
    softmax=torch.nn.Softmax,  # Defaults to softmax along last dimension
    identity=torch.nn.Sequential,  # https://github.com/pytorch/pytorch/issues/9160
)

POOLING_FN_MAP = Dispatcher(
    median=np.median,
    average=np.mean,
    min=np.min,
    max=np.max,
)

TORCH_POOLING_FN_MAP = Dispatcher(
    median=torch.median,
    average=torch.mean,
    min=torch.min,
    max=torch.max,
)

DTYPE_MAP = Dispatcher(
    float32=np.float32,
    float64=np.float64,
    complex64=np.complex64,
    complex128=np.complex128,
)

OPTIMIZER_MAP = Dispatcher(
    sgd=optim.SGD,
    adam=optim.Adam,
    adagrad=optim.Adagrad
)
