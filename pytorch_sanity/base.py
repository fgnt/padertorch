import abc

from torch import nn

from pytorch_sanity.parameterized import Parameterized


__all__ = [
    'Module',
    'Model',
]


class Module(nn.Module, Parameterized, abc.ABC):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Anything
        """
        pass


class Model(Module, Parameterized, abc.ABC):
    """
    Model that can be trained by padertorch.trainer.Trainer
    """
    @abc.abstractmethod
    def forward(self, inputs):
        """

        :param inputs: whatever is required here (Probably dict or tuple).
        :return: ouputs (dict,tuple,list,tensor,...)
        """
        pass

    @abc.abstractmethod
    def review(self, inputs, outputs):
        """

        :param inputs: whatever is required here. (By default trainer provides
        output of iterator. Can be modified by overwriting train_step.)
        :param outputs: outputs of forward function
        :return: dict with possible sub-dicts
        losses/scalars/histograms/images/audios/figures
        """
        pass
