from collections import namedtuple
from distutils.version import LooseVersion
import io
import natsort
import os
from pathlib import Path
import tempfile
import typing as tp

from einops import rearrange
import numpy as np
from paderbox.io import load_yaml
import torch
import yaml
try:
    from parallel_wavegan.utils import download_pretrained_model
except ImportError:
    raise ImportError(
        '`parallel_wavegan` package was not found. '
        'To install it, see here: '
        'https://github.com/kan-bayashi/ParallelWaveGAN'
    )

import padertorch as pt

from ..base import Synthesis


def _pwg_load_model(checkpoint, config=None, stats=None, consider_mpi=False):
    """
    Copy of `parallel_wavegan.utils.load_model` with MPI support

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.
        consider_mpi (bool):

    Returns:

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import parallel_wavegan.models

    # get model and load parameters
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = model_class(**generator_params)
    if consider_mpi:
        checkpoint_content = None
        import dlp_mpi
        if dlp_mpi.IS_MASTER:
            checkpoint_content = Path(checkpoint).read_bytes()
        checkpoint_content = dlp_mpi.bcast(checkpoint_content)
        _checkpoint = torch.load(
            io.BytesIO(checkpoint_content), map_location="cpu")
    else:
        _checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(_checkpoint["model"]["generator"])

    # check stats existence
    if stats is None:
        dirname = os.path.dirname(checkpoint)
        if config["format"] == "hdf5":
            ext = "h5"
        else:
            ext = "npy"
        if os.path.exists(os.path.join(dirname, f"stats.{ext}")):
            stats = os.path.join(dirname, f"stats.{ext}")

    # load stats
    if stats is not None:
        model.register_stats(stats)

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from parallel_wavegan.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion(
                "0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


def load_vocoder_model(
    vocoder_base_path: tp.Union[str, Path], config_name: str = 'config.yml',
    vocoder_stats: str = 'stats.h5', vocoder_checkpoint: str = None,
    consider_mpi=False,
):
    """
    Load a pre-trained vocoder model from
    https://github.com/kan-bayashi/ParallelWaveGAN#results.

    Args:
        vocoder_base_path: Filepath to the vocoder folder containing the
            checkpoint file, config and normalization statistics.
        config_name: Filename of the config file.
        vocoder_stats: Filename of the normalization statistics.
        vocoder_checkpoint: Filename of the checkpoint that should be loaded.
            If None, choose the latest checkpoint in `vocoder_base_path`.
        consider_mpi: If True, reduce IO load on workers

    Returns:
        Loaded model and sampling rate of the synthesized audios
    """

    vocoder_base_path = Path(vocoder_base_path)
    if vocoder_checkpoint is None:
        ckpt_files = natsort.natsorted(list(map(
            lambda p: str(p), vocoder_base_path.glob('*.pkl'))))
        vocoder_checkpoint = ckpt_files[-1]
        print(f'Loading vocoder checkpoint {vocoder_checkpoint}')

    if consider_mpi:
        import dlp_mpi
        config = None
        if dlp_mpi.IS_MASTER:
            config = load_yaml(vocoder_base_path / config_name)
        config = dlp_mpi.bcast(config)
    else:
        config = load_yaml(vocoder_base_path / config_name)

    vocoder = _pwg_load_model(
        vocoder_checkpoint, config,
        stats=str(vocoder_base_path / vocoder_stats),
        consider_mpi=consider_mpi
    )
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval()
    vocoder_params = namedtuple(
        'VocoderParams', [
            'sampling_rate',
            'hop_size',
            'win_size',
            'n_fft',
            'fmin',
            'fmax',
            'num_mels',
        ]
    )
    # window_length = config['win_length']
    # if window_length is None:
    #     window_length = config['fft_size']
    # window_length = window_length / config['sampling_rate'] * 1000
    # shift = config['hop_size'] / config['sampling_rate'] * 1000
    return vocoder, vocoder_params(
        config['sampling_rate'],
        config['hop_size'],
        config['win_length'],
        config['fft_size'],
        config['fmin'],
        config['fmax'],
        config['num_mels'],
    )


class Vocoder(Synthesis):
    """
    Neural vocoder wrapping any models from https://github.com/kan-bayashi/ParallelWaveGAN

    Provides an easy-to-use interface to download vocoders and to perform
    waveform synthesis from log-mel spectrogram. Vocoder models are identified
    by a vocoder tag of the form <database>_<model>.<version_no.>. The __call__
    has to take a log-mel spectrogram (np.ndarray or torch.Tensor), a list of
    sequence lengths (in case of a batched input), the desired output sampling
    rate, and any optional *keyword arguments* to control the synthesis.

    Attributes:
        sampling_rate (int): Sampling rate of the training data the vocoder was
            trained on
    """
    def __init__(
        self,
        database: str = 'libritts',
        base_dir: tp.Optional[tp.Union[str, Path]] = \
            os.environ.get('PWG_BASE_DIR', None),
        vocoder_model: str = 'hifigan',
        vocoder_tag: tp.Optional[str] = None,
        normalize_before: bool = True,
        device: tp.Union[str, int] = 'cpu',
        consider_mpi: bool = False,
        batch_axis: int = 0,
        sequence_axis: int = -1,
        postprocessing: tp.Optional[tp.Callable] = None,
     ):
        """
        Args:
            database: The database the vocoder was trained on (see
                https://github.com/kan-bayashi/ParallelWaveGAN). Used to
                infer the vocoder model. If `vocoder_tag` is not None, this
                argument will be ignored
            base_dir: Path to folder where vocoders will be downloaded to.
                Vocoders will be dumped as directories indexed by their vocoder
                tag. If a vocoder is already downloaded, it will be loaded from
                disk instead of being downloaded again
            vocoder_model: Type of vocoder architecture as specified in the
                vocoder tag, e.g., "parallel_wavegan" or "hifigan". See
                https://github.com/kan-bayashi/ParallelWaveGAN for available
                architectures. If `vocoder_tag` is not None, this
                argument will be ignored
            vocoder_tag: Vocoder tag of the form
                <database>_<vocoder_model>.<version_no.>. If not None, will
                first look under `base_dir` and then try to download it
                from https://github.com/kan-bayashi/ParallelWaveGAN
            normalize_before: If True, perform z-normalization with vocoder
                train statistics. If False, `mel_spec` should be normalized
                with test statistics. Defaults to True
            device: Device (CPU, GPU) used for inference
            consider_mpi: If True, load the weights on the master and distribute
                to all workers
            batch_axis: Axis along which the batches are stacked. If the input
                to __call__ is 2-dimensional, `batch_axis` will be ignored
            sequence_axis: Axis that contains time information
            postprocessing: Optional postprocessing function that is applied to
                the synthesized waveform
        """
        super().__init__(postprocessing=postprocessing)
        self.base_dir = base_dir
        self.vocoder_tag = vocoder_tag
        self.normalize_before = normalize_before
        self.device = device
        self.batch_axis = batch_axis
        self.sequence_axis = sequence_axis

        if self.vocoder_tag is None:
            self.vocoder_tag = str(Path(
                '_'.join((database, vocoder_model))).with_suffix('.v1'))
        if self.base_dir is None:
            self.base_dir = Path(tempfile.gettempdir()) / 'pwg_models'
        else:
            self.base_dir = Path(self.base_dir)
        if not (self.base_dir / self.vocoder_tag).exists():
            # Download vocoder and store it under `self.base_dir`
            if consider_mpi:
                try:
                    import dlp_mpi
                except ImportError as e:
                    raise ImportError(
                        'Could not import dlp_mpi.\n'
                        'Please install it or set consider_mpi=False'
                    ) from e
                if dlp_mpi.IS_MASTER:
                    self._download()
                dlp_mpi.barrier()
            else:
                self._download()
        self.vocoder_model, self.vocoder_params = load_vocoder_model(
            self.base_dir / self.vocoder_tag, consider_mpi=consider_mpi)
        self.sampling_rate = self.vocoder_params.sampling_rate
        self.vocoder_model.to(self.device).eval()

    def _download(self):
        try:
            download_pretrained_model(self.vocoder_tag, str(self.base_dir))
            print(f'Downloaded {self.vocoder_tag} to {self.base_dir}')
        except KeyError as e:
            raise KeyError(
                f'Could not find {self.vocoder_tag} in pretrained models!\n'
                'list(self.base_dir.iterdir()): '
                f'{[p for p in self.base_dir.iterdir() if p.is_dir()]}\n'
                'Check parallel_wavegan.PRETRAINED_MODEL_LIST or '
                'https://github.com/kan-bayashi/ParallelWaveGAN#results for'
                'more pretrained models.\n'
                'You can specify a pretrained model with the vocoder_tag '
                'argument.'
            ) from e

    def __call__(
        self,
        mel_spec: tp.Union[torch.Tensor, np.ndarray],
        sequence_lengths: tp.Optional[tp.List[int]] = None,
        target_sampling_rate: tp.Optional[int] = None,
    ) -> tp.Union[
        torch.Tensor, np.ndarray, tp.List[np.ndarray],
        tp.List[torch.Tensor]
    ]:
        """
        Synthesize waveform from log-mel spectrogram with a neural vocoder

        >>> vocoder = Vocoder()
        >>> mel_spec = torch.zeros((1, 80, 100))
        >>> wav_gen = vocoder(mel_spec)
        >>> wav_gen.shape
        torch.Size([1, 30000])

        Args:
            mel_spec: (Batched) mel-spectrograms where shape must match as
                specified by `self.batch_axis` and `self.sequence_axis`
            sequence_lengths:
            target_sampling_rate: By default, vocoder will produce waveforms
                with sampling rate seen during training. If not None, resample
                to `target_sampling_rate`

        Returns: torch.Tensor or np.ndarray
            Synthesized waveform
        """
        to_numpy = isinstance(mel_spec, np.ndarray)
        if to_numpy:
            mel_spec = torch.from_numpy(mel_spec).to(self.device)
        sequence_axis = self.sequence_axis % mel_spec.ndim
        batch_axis = self.batch_axis % mel_spec.ndim
        if mel_spec.ndim == 2:
            mel_spec = torch.moveaxis(mel_spec, sequence_axis, 0)
            with torch.no_grad():
                y = self.vocoder_model.inference(
                    mel_spec, normalize_before=self.normalize_before
                ).view(-1)
                if to_numpy:
                    y = pt.utils.to_numpy(y, detach=True)
        elif mel_spec.ndim == 3:
            feature_axis = set(range(3)).difference(
                {batch_axis, self.sequence_axis}
            ).pop()
            shape = list(map(
                lambda t: t[1],
                sorted(
                    zip(
                        [batch_axis, sequence_axis, feature_axis],
                        ['b', 't', 'f']
                    ), key=lambda t: t[0]
                )
            ))
            mel_spec = rearrange(mel_spec, f"{' '.join(shape)} -> b t f")
            if sequence_lengths is None:
                sequence_lengths = [mel_spec.shape[1]] * mel_spec.shape[0]
            with torch.no_grad():
                y = []
                for _mel_spec, seq_len in zip(mel_spec, sequence_lengths):
                    y_ = (
                        self.vocoder_model.inference(
                            _mel_spec[:seq_len],
                            normalize_before=self.normalize_before
                        ).view(-1)
                    )
                    if to_numpy:
                        y_ = pt.utils.to_numpy(y, detach=True)
                    y.append(y_)
                try:
                    if to_numpy:
                        y = np.stack(y)
                    else:
                        y = torch.stack(y)
                except RuntimeError:
                    pass
        else:
            raise TypeError(
                'Expected 2- or 3-dim. spectrogram but got '
                f'{mel_spec.ndim}-dim. input with shape {mel_spec.shape}'
            )
        return super().__call__(y, target_sampling_rate)
