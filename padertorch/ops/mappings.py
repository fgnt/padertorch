import torch
from torch import optim
from paderbox.utils.mapping import Dispatcher
import numpy as np

__all__ = [
    'ACTIVATION_FN_MAP',
]

class _CallableDispatcher(Dispatcher):
    """
       If the input is a callable it is returned.
       Otherwise, it is basically a dict
       with a better error message on key error.
       >>> from padertorch.ops.mappings import _CallableDispatcher
       >>> d = _CallableDispatcher(abc=1, bcd=2)
       >>> d['acd']  #doctest: +ELLIPSIS
       Traceback (most recent call last):
       ...
       paderbox.utils.mapping.DispatchError: Invalid option 'acd'.
       Close matches: ['bcd', 'abc'].
       >>> from padertorch.ops.mappings import _CallableDispatcher
       >>> d = _CallableDispatcher(abc=1, bcd=2)
       >>> d[np.median]  #doctest: +ELLIPSIS
       <function median at ...
       """

    def __getitem__(self, item):
        if callable(item):
            return item
        else:
            return super().__getitem__(item)


ACTIVATION_FN_MAP = _CallableDispatcher(
    relu=torch.nn.ReLU,
    prelu=torch.nn.PReLU,
    leaky_relu=torch.nn.LeakyReLU,
    elu=torch.nn.ELU,
    tanh=torch.nn.Tanh,
    sigmoid=torch.nn.Sigmoid,
    softmax=torch.nn.Softmax,  # Defaults to softmax along last dimension
    identity=torch.nn.Identity,
)


# These mappings are not used at the moment if required they can be added again
# but the naming convention shuld be updated.
# NP_REDUCE_MAP = _CallableDispatcher(
#     median=np.median,
#     mean=np.mean,
#     min=np.min,
#     max=np.max,
# )
#
# REDUCE_MAP = _CallableDispatcher(
#     median=torch.median,
#     mean=torch.mean,
#     min=torch.min,
#     max=torch.max,
# )
#
# DTYPE_MAP = Dispatcher(
#     float32=np.float32,
#     float64=np.float64,
#     complex64=np.complex64,
#     complex128=np.complex128,
# )
#
# OPTIMIZER_MAP = _CallableDispatcher(
#     sgd=optim.SGD,
#     adam=optim.Adam,
#     adagrad=optim.Adagrad
# )
