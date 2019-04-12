"""Provide Module and Model abstract base classes.

This module defines abstract base classes which allow subclassed modules
to be used in models and subclassed models to be configurable and trainable
by instances of pytorch.train.Trainer.
"""
import io
import abc
from pathlib import Path

import torch
from torch import nn

from paderbox.utils.nested import deflatten
from padertorch.configurable import Configurable


__all__ = [
    'Module',
    'Model',
]


class Module(nn.Module, Configurable, abc.ABC):
    """Abstract base class for configurable Modules."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Define the I/O behavior of Module()."""
        pass

    @classmethod
    def from_config_and_checkpoint(
            cls,
            config_path: Path,
            checkpoint_path: Path,
            in_config_path: str = 'trainer.model',
            in_checkpoint_path: str = 'model',

            map_location='cpu',
            consider_mpi=False,
    ) -> 'Module':
        """Instantiate the module from given config and checkpoint.

        Args:
            config_path: 
            checkpoint_path: 
            in_config_path: 
            in_checkpoint_path: 
            map_location: 
            consider_mpi:
                If True and mpi is used, only read config_path and
                checkpoint_path once and broadcast the content with mpi.
                Reduces the io load.

        Returns:
        
        
        """
        config_path = Path(config_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        assert config_path.is_file(), config_path
        assert checkpoint_path.is_file(), checkpoint_path
        # Load config
        module = cls.from_file(
            config_path,
            in_config_path,
            consider_mpi=False
        )

        # Load weights
        if consider_mpi:
            from paderbox.utils import mpi
            checkpoint_path_content = mpi.call_on_master_and_broadcast(
                Path(checkpoint_path).read_bytes,
            )
            checkpoint = torch.load(
                io.BytesIO(checkpoint_path_content),
                map_location=map_location,
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        for part in in_checkpoint_path.split('.'):
            try:
                checkpoint = deflatten(checkpoint, maxdepth=1)
                checkpoint = checkpoint[part]
            except KeyError:
                raise ValueError(part, in_checkpoint_path, checkpoint)
        module.load_state_dict(checkpoint)

        return module

    @classmethod
    def from_storage_dir(
            cls,
            storage_dir: Path,
            config_name: str = 'config.json',
            checkpoint_name: str = 'ckpt_best_loss.pth',
            in_config_path: str = 'trainer.model',
            in_checkpoint_path: str = 'model',
            consider_mpi=False,
    ) -> 'Module':
        """Instantiate the module from a given storage directory.

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
            config_name: In case you config has a different name.
            checkpoint_name: In case the name is not default.
            in_config_path: In case you want to load an inner module.
            in_checkpoint_path: In case you want to load an inner module.
            consider_mpi: If you use MPI: Only load on master, the distribute.

        Returns:

        """
        storage_dir = Path(storage_dir)
        return cls.from_config_and_checkpoint(
            config_path=storage_dir / config_name,
            checkpoint_path=storage_dir / 'checkpoints' / checkpoint_name,
            in_config_path=in_config_path,
            in_checkpoint_path=in_checkpoint_path,
            consider_mpi=consider_mpi,
        )


class Model(Module, Configurable, abc.ABC):
    """Abstract base class for configurable trainable  models.

    Subclassed models can be trained by padertorch.trainer.Trainer.
    """

    @abc.abstractmethod
    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Define the I/O behavior of Model().

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
        """Produce a review dictionary from given inputs and outputs.

        In particular, this method usually calculates the loss function
        and adds the result to the review dict.
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

    def modify_summary(self, summary):
        """Modify a summary dict.

        This method is primarily used by SummaryHook before dumping a summary.
        Summary contains accumulated values from multiple reviews (lists in
        "scalars" and "histograms", snapshots in "audios" and "images").
        This, e.g., allows to accurately compute and add metrics based on
        other scalars such as F-scores or Error Rates.

        Args:
            summary:
                dict containing nested dicts with keys scalars, histograms,
                audios, images.

        Returns:
            Modified summary dict

        """
        return summary
