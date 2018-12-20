import torch.nn.functional as F
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
    relu=F.relu,
    leaky_relu=torch.nn.LeakyReLU(.1),
    elu=F.elu,
    tanh=F.tanh,
    sigmoid=torch.sigmoid,
    softmax=F.softmax,  # Defaults to softmax along last dimension
    identity=lambda x: x,
)

POOLING_FN_MAP = Dispatcher(
    median=np.median,
    average=np.mean,
    min=np.min,
    max=np.max,
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
