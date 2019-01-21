import abc
from pathlib import Path
import json

import torch
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

    @classmethod
    def from_config_and_checkpoint(
            cls,
            config_path: Path,
            checkpoint_path: Path,
            in_config_path: str='trainer.kwargs.model',
            in_checkpoint_path: str='model',
    ) -> 'Module':
        config_path = Path(config_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        assert config_path.is_file(), config_path
        assert checkpoint_path.is_file(), checkpoint_path

        # Load config
        assert config_path.is_file(), f'Expected {config_path} is file.'
        with config_path.open() as fp:
            configurable_config = json.load(fp)
        for part in in_config_path.split('.'):
            configurable_config = configurable_config[part]
        module = cls.from_config(configurable_config)

        # Load weights
        checkpoint = torch.load(checkpoint_path)
        for part in in_checkpoint_path.split('.'):
            checkpoint = checkpoint[part]
        module.load_state_dict(checkpoint)

        return module

    @classmethod
    def from_storage_dir(
            cls,
            storage_dir: Path,
            checkpoint_name: str='ckpt_best_loss.pth',
            in_config_path: str='trainer.kwargs.model',
            in_checkpoint_path: str='model',
    ) -> 'Module':
        """
        Assumes fixed folder structure in the default configuration. If this
        folder structure is not your folder structure, use the function
        `from_config_and_checkpoint` directly.

        Assumes this structure:
        storage_dir
        ├── checkpoints
        │   └── ckpt_best_loss.pth
        └── config.json

        Args:
            storage_dir: Path which was provided during training.
            checkpoint_name: In case the name is not default.
            in_config_path: In case you want to load an inner module.
            in_checkpoint_path: In case you want to load an inner module.

        Returns:

        """
        return cls.from_config_and_checkpoint(
            config_path=storage_dir / 'config.json',
            checkpoint_path=storage_dir / 'checkpoints' / checkpoint_name,
            in_config_path=in_config_path,
            in_checkpoint_path=in_checkpoint_path
        )


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
