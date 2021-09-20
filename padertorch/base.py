"""Provide Module and Model abstract base classes.

This module defines abstract base classes which allow subclassed modules
to be used in models and subclassed models to be configurable and trainable
by instances of pytorch.train.Trainer.
"""
import io
import abc
from pathlib import Path

import numpy as np
import torch
from torch import nn

from paderbox.utils.nested import deflatten
from padertorch.configurable import Configurable
from padertorch.data import example_to_device

__all__ = [
    'Module',
    'Model',
    'torch_dataclass',
]


class Module(nn.Module, Configurable, abc.ABC):
    """Abstract base class for configurable Modules."""

    # Indicate if the module is in training mode.
    #
    # Docstring part from torch.nn.Module.train:
    #  This has any effect only on certain modules. See documentations of
    #  particular modules for details of their behaviors in training/evaluation
    #  mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
    #  etc.
    #
    # Pycharm has problems with autocomplete for this flag.
    # hence add this type annotation
    training: bool

    @abc.abstractmethod
    def forward(self, *args, **kwargs):  # pylint: disable=arguments-differ
        """Define the I/O behavior of Module()."""
        pass

    @classmethod
    def from_config_and_checkpoint(
            cls,
            config_path: (Path, str),
            checkpoint_path: (Path, str),
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
            consider_mpi=consider_mpi,
        )
        return module.load_checkpoint(
            checkpoint_path=checkpoint_path,
            in_checkpoint_path=in_checkpoint_path,
            map_location=map_location,
            consider_mpi=consider_mpi,
        )

    def load_checkpoint(
            self,
            checkpoint_path: (Path, str),
            in_checkpoint_path: str = 'model',

            map_location='cpu',
            consider_mpi=False,
    ) -> 'Module':
        """Update the module parameters from the given checkpoint.

        Args:
            checkpoint_path:
            in_checkpoint_path:
            map_location:
            consider_mpi:
                If True and mpi is used, only read config_path and
                checkpoint_path once and broadcast the content with mpi.
                Reduces the io load.

        Returns:


        """
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()

        assert checkpoint_path.is_file(), checkpoint_path

        # Load weights
        if consider_mpi:
            import dlp_mpi
            if dlp_mpi.IS_MASTER:
                checkpoint_path_content = Path(checkpoint_path).read_bytes()
            else:
                checkpoint_path_content = None
            checkpoint_path_content = dlp_mpi.bcast(checkpoint_path_content)

            checkpoint = torch.load(
                io.BytesIO(checkpoint_path_content),
                map_location=map_location,
            )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        if in_checkpoint_path:
            for part in in_checkpoint_path.split('.'):
                try:
                    checkpoint = deflatten(checkpoint, maxdepth=1)
                    checkpoint = checkpoint[part]
                except KeyError:
                    raise ValueError(part, in_checkpoint_path, checkpoint)
        self.load_state_dict(checkpoint)

        return self

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

    # This flag is True when the model should create a snapshot in the review
    create_snapshot: bool = False

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
                    logged as mean
                histograms: see tensorboardX documentation
                images: see tensorboardX documentation
                    logged as snapshot
                audios: see tensorboardX documentation
                    logged as snapshot
                figures: see tensorboardX documentation
                    logged as snapshot
                texts: see tensorboardX documentation
                    logged as snapshot


        Hints:
         - The contextmanager `torch.no_grad()` disables backpropagation for
            metric computations (i.e. scalars in tensorboard)
         - `self.training` (bool) indicate training or validation mode
         - `self.create_snapshot` (bool) indicates when a snapshot is created
            for reporting.
            Motivation: Only one snapshot per summary interval is reported by
            the summary hook for each snapshot type (e.g., images, audios, ...).
            You can use this flag to avoid unnecessary computations
            by only computing a snapshot if it will be reported instead of
            computing a snapshot in every iteration.

        """
        ...  # calculate loss
        with torch.no_grad():
            ...  # calculate general metrics
            if self.training:
                ...  # calculate training specific metrics
            else:
                ...  # calculate validation specific metrics

            if self.create_snapshot:
                ...  # create snapshots to be displayed in TensorBoard

    def modify_summary(self, summary):
        """Modify a summary dict.

        This method is primarily used by SummaryHook before dumping a summary.
        Summary contains accumulated values from multiple reviews (lists in
        "buffers", "scalars" and "histograms", snapshots in "snapshots",
        "audios" and "images").
        This, e.g., allows to accurately compute and add metrics based on
        other scalars such as F-scores or Error Rates.
        The intermediate formats "buffers" and "snapshots" make no assumption
        on the type of the data saved under this key. Therefore, they can be used
        to agglomerate any data over multiple reviews.


        Args:
            summary:
                dict containing nested dicts with keys scalars, histograms,
                audios, images.

        Returns:
            Modified summary dict

        Hints:
         - For training summary contains a subset of all training values.
           It can happen that the number of values is very low when
           summary_trigger and checkpoint_trigger have different units.
         - For validation the summary contains all values.
         - The intermediate keys "buffers" and "snapshots" must not contain
           any entries by the end of modify_summary.
        """
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)

        assert len(
            summary['buffers']) == 0, "intermediate format buffers has to be converted during modify_summary"
        assert len(
            summary['snapshots']) == 0, "intermediate format snapshots has to be converted during modify summary"

        return summary

    def example_to_device(self, example, device=None, memo=None):
        """
        Transfers `example` to `device` as required by the model. By default,
        the whole example is transferred to `device`, but subclasses can
        override this method to only transfer the required parts of the
        example.

        An example for data that is not required on GPU during training are
        time-domain target signals for an STFT-based model. These are not
        required for loss computation, but are nice to have reported to
        tensorboard.

        Args:
            example: The example to transfer to `device`
            device: The device to transfer `example` to.
            memo: See `copy.deepcopy`

        Returns:
            The `example`, either fully or partially transferred to the device.
        """
        return example_to_device(example, device, memo)

    def extra_repr(self) -> str:
        """Set the extra representation of the module

        The default `extra_repr` prints the model size (number of parameters).

        To print customized extra information, you should re-implement
        this method in your own model. Note that the size information is lost
        if you don't add the output of `super().extra_repr()` to your custom
        `extra_repr`.

        >>> class MyModel(Model):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layer = nn.Linear(10, 10)
        ...     def forward(self, *args, **kwargs): pass
        ...     def review(self, inputs, outputs): pass
        >>> print(repr(MyModel()))
        MyModel(
          size=ModelParameterSize(total_count=110, trainable_count=110, total_bytes=440 Bytes, trainable_bytes=440 Bytes)
          (layer): Linear(in_features=10, out_features=10, bias=True)
        )
        """
        from padertorch.summary import num_parameters
        return f'size={num_parameters(self)}'


def torch_dataclass(cls):
    """
    A decorator that allows using `torch.nn.Module`s as dataclasses.

    The arguments of `dataclass` are set automatically in a way that makes most
    sense with `torch.nn.Module`: init=True, repr=False, eq=False. `repr` and
    `eq` should be handled by `torch`.

    `torch.nn.Module` is implemented in a way that its `__init__` has to be
    called before any torch-specific attributes can be set. `dataclass`
    manipulates `__init__` to set instance attributes without calling
    `super().__init__`, so both can, by default, not be used in conjunction.
    This decorator modifies the `__init__` generated by the `dataclass`
    decorator so that it calls `torch.nn.Module.__init__` before setting
    instance attributes.

    Examples:
        >>> @torch_dataclass
        ... class MyModule(Module):
        ...     some_module: torch.nn.Module
        ...     in_size: int = 42
        ...     out_size: int = 42
        ...
        ...     def __post_init__(self):
        ...         self.layer = torch.nn.Linear(self.in_size, self.out_size)
        ...
        ...     def forward(self): pass
        >>> MyModule(torch.nn.Linear(10, 10))
        MyModule(
          (some_module): Linear(in_features=10, out_features=10, bias=True)
          (layer): Linear(in_features=42, out_features=42, bias=True)
        )

    """
    from dataclasses import dataclass
    import functools

    # If cls doesn't inherit from torch.nn.Module, we don't need to do anything
    if torch.nn.Module not in cls.__mro__:
        return dataclass(cls)

    # repr and eq should be handled by torch. We could wrap torch's
    # extra_repr to include the dataclass args
    cls = dataclass(cls, repr=False, eq=False)

    # We have to modify __init__ so that it calls torch.nn.Module.__init__
    # before the setup code in the __init__ generated by dataclass.
    # torch.nn.Module.__init__ does some setup code before torch-specific
    # attributes (like other modules) can be set.
    old_init = cls.__init__

    @functools.wraps(old_init)
    def _init_fn(self, *args, **kwargs):
        # We have to use torch.nn.Module instead of super().__init__ to
        # support chains of dataclasses. In chains of dataclasses
        # super().__init__ should never be used due to the way the dataclass
        # generates its __init__.
        torch.nn.Module.__init__(self)
        old_init(self, *args, **kwargs)

    setattr(cls, '__init__', _init_fn)

    return cls
