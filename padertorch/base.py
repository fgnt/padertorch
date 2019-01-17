import abc

from torch import nn

from padertorch.configurable import Configurable


__all__ = [
    'Module',
    'Model',
]


class Module(nn.Module, Configurable, abc.ABC):
    """
    Abstract base class for configurable Modules.
    """
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Anything
        """
        pass


class Model(Module, Configurable, abc.ABC):
    """
    Abstract base class for configurable Models which can be trained by
    padertorch.trainer.Trainer.
    """
    @abc.abstractmethod
    def forward(self, inputs):
        """

        Args:
            inputs:
                Single example from train_iterator or validation_iterator that
                is provided to `pt.Trainer`.

        Returns:
            Whatever self.review expects as second argument.
            Usually something like a prediction.

        """
        pass

    @abc.abstractmethod
    def review(self, inputs, outputs):
        """

        Args:
            inputs:
                Same as `inputs` argument of `self.forward`.

                Single example from train_iterator or validation_iterator that
                is provided to `pt.Trainer`.

                In case of multi model `pt.Trainer.step` is overwritten.
                Than see that new function.
            outputs:
                Output of `self.forward`


        Returns:
            dict with possible sub-dicts for tensorboard
                losses:
                    Will be added to scalars logging of tensorboard.
                    Combined with `pt.Trainer.loss_weights` these losses build
                    the loss. On the loss is backward called for gradient
                    update.
                loss:
                    Scalar objective. Only allowed when no losses is provided.
                    Otherwise it will be computed from losses.
                scalars: dict of scalars for tensorboard
                histograms: see tensorboardX documentation
                images: see tensorboardX documentation
                audios: see tensorboardX documentation
                figures: see tensorboardX documentation


        Hints:
         - The contextmanager `torch.no_grad()` disables backpropagation for
           metric computations (i.e. scalars in tensorboard)
         - `self.training` (bool) indicate training or validation mode


        """
        pass
