import abc

from torch import nn

from padertorch.configurable import Configurable


__all__ = [
    'Module',
    'Model',
]


class Module(nn.Module, Configurable, abc.ABC):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Anything
        """
        pass


class Model(Module, Configurable, abc.ABC):
    """
    Model that can be trained by padertorch.trainer.Trainer
    """
    @abc.abstractmethod
    def forward(self, inputs):
        """

        :param inputs: whatever is required here (Probably dict or tuple).
        :return: outputs (dict,tuple,list,tensor,...)
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
