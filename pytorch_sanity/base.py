from torch import nn
import abc


class Module(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Anything
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_config(cls, update_dict=None):
        """
        Provides configuration to allow Instantiation with
        module = Module(**Module.get_config())
        :param update_dict: dict with values to be modified w.r.t. defaults.
        Sub-configurations are updated accordingly if top-level-keys are
        changed. An Exception is raised if update_dict has unused entries.
        :return: config
        """
        pass


class Model(Module, abc.ABC):
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
