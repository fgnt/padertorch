import typing
from pathlib import Path

import numpy as np
import scipy.fftpack
from scipy import signal
import einops
import functools

import torch

import paderbox as pb
import padertorch as pt
import padercontrib as pc

from padertorch.contrib.cb.transform import stft as pt_stft, istft as pt_istft
from padertorch.contrib.cb import complex as pt_complex

import cbj


def kaldi_stft(time_signal, size=512, shift=160, *, axis=-1, window_length=400, pad=False, fading=False):
    # ToDo: window
    return pb.transform.stft(**locals())


def kaldi_istft(stft_signal, size=512, shift=160, *, axis=-1, window_length=400, pad=False, fading=False):
    # ToDo: window
    return pb.transform.istft(**locals())


def stft_to_cepstrum(
        stft_signal,
        norm='ortho',
        eps=None,
):
    """
    Reference implementation to get the cepstrum: dft -> abs -> log -> dft
    >>> signal1 = np.array([1, 2, 3, 4])
    >>> signal2 = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    >>> dft_signal = np.fft.fft(signal1)
    >>> np.fft.fft(np.log(np.abs(dft_signal)), norm='ortho')
    array([2.53758691+0.j, 0.80471896+0.j, 0.45814537+0.j, 0.80471896+0.j])
    >>> dft_signal = np.fft.fft(signal2)
    >>> np.fft.fft(np.log(np.abs(dft_signal)), norm='ortho')
    array([5.67812692+0.j, 1.21752299+0.j, 0.53177166+0.j, 0.33614941+0.j,
           0.28670713+0.j, 0.33614941+0.j, 0.53177166+0.j, 1.21752299+0.j])

    Implementation without redundant operations: rdft -> abs -> log -> dct
    >>> rdft_signal = np.fft.rfft(signal1)
    >>> stft_to_cepstrum(rdft_signal)
    array([2.53758691, 0.80471896, 0.45814537])

    >>> rdft_signal = np.fft.rfft(signal2)
    >>> stft_to_cepstrum(rdft_signal)
    array([5.67812692, 1.21752299, 0.53177166, 0.33614941, 0.28670713])


    Note: a scaling only influences the first value
    >>> rdft_signal = np.fft.rfft(signal1)
    >>> stft_to_cepstrum(rdft_signal * 2)
    array([3.92388127, 0.80471896, 0.45814537])

    >>> stft_to_cepstrum([0., 0, 0])
    array([-1416.79283706,     0.        ,     0.        ])
    >>> stft_to_cepstrum([0., 0, 0], eps=0)
    array([-inf,  nan,  nan])
    >>> stft_to_cepstrum([1., 1, 1])
    array([0., 0., 0.])
    >>> stft_to_cepstrum([1., 1, 1], eps=0)
    array([0., 0., 0.])
    >>> stft_to_cepstrum([0., 1, 1])
    array([-354.19820927, -354.19820927, -354.19820927])
    >>> stft_to_cepstrum([0., 1, 1], eps=0)
    array([-inf, -inf, -inf])
    """
    stft_signal = np.asarray(stft_signal)

    assert stft_signal.shape[-1] % 2 == 1, stft_signal.shape

    stft_signal = np.abs(stft_signal)

    if eps is None:
        eps = np.finfo(stft_signal.dtype).tiny

    if eps != 0:
        stft_signal = np.maximum(stft_signal, eps)

    cepstrum = scipy.fftpack.dct(np.log(stft_signal), type=1)
    if norm == 'ortho':
        cepstrum = cepstrum / np.sqrt(2 * (stft_signal.shape[-1] - 1))
    elif norm is None:
        pass
    else:
        raise ValueError(norm)
    return cepstrum


def sign(signal, *, axis=-1, eps=1e-4, eps_style='where', ord=None):
    """Unit normalization.

    Numpy handles complex numbers in the sign function in a strange way.
    See:  https://github.com/numpy/numpy/issues/13179


    Args:
        signal: STFT signal with shape (..., T, D).
        eps_style: in ['plus', 'max']
    Returns:
        Normalized STFT signal with same shape.

    >>> sign([1+1j, 0,  1e-4])
    array([0.70710678+0.70710678j, 0.        +0.j        ,
           1.        +0.j        ])
    """
    signal = np.asarray(signal)
    norm = np.abs(signal)
    if eps_style == 'plus':
        norm = norm + eps
    elif eps_style == 'max':
        norm = np.maximum(norm, eps)
    elif eps_style == 'where':
        norm = np.where(norm == 0, eps, norm)
    else:
        assert False, eps_style
    return signal / norm


def interchannel_phase_differences_op(a, b):
    return sign(a * b.conj())


def interchannel_phase_differences(signal, second_channel=None, concatenate=False):
    """
    Calculates the inter channel phase differences:

        cos(angle(channel1 * channel2.conj()))
        sin(angle(channel1 * channel2.conj()))

    Args:
        signal: The stft signal.
            Shape: (..., channels, frames, features)
        second_channel:
            The corresponding second channel. When not given, use random
            sampled channels.
        concatenate:


    Returns:
            If concatenate True, return the concaternation of abs, cos and sin
            on the last axis.
            Otherwise return the tuple (cos, sin)


    >>> np.random.seed(0)
    >>> signal = np.ones([6, 4, 5]) * np.exp(1j * np.random.uniform(0, 2*np.pi, [6, 1, 1])) * (np.arange(6)[:, None, None] + 1)
    >>> c, s = interchannel_phase_differences(signal)
    >>> c[0, :, :]
    array([[0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208],
           [0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208]])
    >>> c[:, 0, 0], s[:, 0, 0]
    (array([0.81966208, 0.76070789, 0.93459697, 0.93459697, 0.72366352,
           0.90670355]), array([-0.57284734,  0.64909438,  0.35570844, -0.35570844, -0.69015296,
           -0.42176851]))
    >>> sig = interchannel_phase_differences(signal, concatenate=True)
    >>> sig[-1, 0, :]
    array([6.        , 6.        , 6.        , 6.        , 6.        ,
           0.81966208, 0.81966208, 0.81966208, 0.81966208, 0.81966208,
           0.57284734, 0.57284734, 0.57284734, 0.57284734, 0.57284734])

    >>> sig[:, 0, 0]
    array([1., 2., 3., 4., 5., 6.])

    """
    import itertools, random

    if second_channel is None:
        D = signal.shape[-3]
        assert D == 6, (D, signal.shape)
        l = list(itertools.permutations(range(D), 2))
        np.random.shuffle(l)
        second_channel = np.array(sorted(dict(l).items()))[:, 1]

    sincos = interchannel_phase_differences_op(signal, signal[..., second_channel, :, :])

    if concatenate:
        return np.concatenate([np.abs(signal), sincos.real, sincos.imag], axis=-1)
    else:
        return sincos.real, sincos.imag


def cepstrum(
        time_signal,
        size: int=1024,
        shift: int=256,
        *,
        window: typing.Callable=signal.blackman,
        window_length: int=None,
        fading: bool=True,
        pad: bool=True,
        symmetric_window: bool=False,
        # dft_norm=None,
):
    stft_signal = pb.transform.stft(
        time_signal,
        size=size,
        shift=shift,
        axis=-1,
        window=window,
        window_length=window_length,
        fading=fading,
        pad=pad,
        symmetric_window=symmetric_window,
        # dft_norm=dft_norm,
    )
    return stft_to_cepstrum(stft_signal)



class FeatureExtractor(pt.Configurable):
    """

    >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'

    >>> config = FeatureExtractor.get_config()
    >>> pb.notebook.pprint(config)
    {'factory': 'feature_extractor.FeatureExtractor',
     'type': 'mfcc_high_res',
     'num_mel_bins': 40,
     'num_ceps': 40,
     'low_freq': 20,
     'high_freq': -400,
     'delta_order': 2,
     'output_size': 120}
    >>> fe_mfcc = FeatureExtractor.from_config(config)
    >>> fe_mfcc
    FeatureExtractor('mfcc_high_res', num_mel_bins=40, num_ceps=40, low_freq=20, high_freq=-400, delta_order=2, output_size=120)
    >>> config = FeatureExtractor.get_config({'type': 'mel'})
    >>> fe_mel = FeatureExtractor.from_config(config)
    >>> fe_mel
    FeatureExtractor('mel', delta_order=2, num_mel_bins=23, output_size=69)
    >>> config = FeatureExtractor.get_config({'type': 'mfcc_high_res'})
    >>> fe_mfcc_high_res = FeatureExtractor.from_config(config)
    >>> fe_mfcc_high_res
    FeatureExtractor('mfcc_high_res', num_mel_bins=40, num_ceps=40, low_freq=20, high_freq=-400, delta_order=2, output_size=120)
    >>> config = FeatureExtractor.get_config({'type': 'stft'})
    >>> fe_stft = FeatureExtractor.from_config(config)
    >>> fe_stft
    FeatureExtractor('stft', size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513)

    >>> config = FeatureExtractor.get_config({'type': 'cepstrum'})
    >>> fe_cepstrum = FeatureExtractor.from_config(config)
    >>> fe_cepstrum
    FeatureExtractor('cepstrum', size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513)

    >>> def print_properties(array):
    ...     print(f'array(shape={array.shape}, dtype={array.dtype})')

    >>> print_properties(fe_mfcc([file]))
    array(shape=(290, 1, 120), dtype=float32)
    >>> print_properties(fe_mel([file]))
    array(shape=(290, 1, 69), dtype=float32)
    >>> print_properties(fe_mfcc_high_res([file]))
    array(shape=(290, 1, 120), dtype=float32)
    >>> print_properties(fe_stft([file]))
    array(shape=(1, 186, 513), dtype=complex128)
    >>> print_properties(fe_cepstrum([file]))
    array(shape=(1, 186, 513), dtype=float64)

    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config['type'] in ['stft', 'abs_stft', 'cepstrum']:
            config['size'] = 1024
            config['shift'] = 256
            # config['axis'] = -1
            config['window_length'] = config['size']
            config['pad'] = True
            config['fading'] = True
            config['output_size'] = config['size'] // 2 + 1
            # config['dft_norm'] = 'ortho'
        elif config['type'] in ['kaldi_stft', 'abs_kaldi_stft']:
            config['size'] = 512
            config['shift'] = 160
            # config['axis'] = -1
            config['window_length'] = 400
            config['pad'] = False
            config['fading'] = False
            config['output_size'] = config['size'] // 2 + 1
            # config['dft_norm'] = 'ortho'
        elif config['type'] == 'mfcc':
            config['delta_order'] = 2

            config['num_mel_bins'] = 23
            config['num_ceps'] = 13
            config['low_freq'] = 20
            config['high_freq'] = 0

            config['output_size'] = config['num_ceps'] * (config['delta_order'] + 1)
        elif config['type'] == 'mfcc_high_res':
            for k, v in pb.kaldi.mfcc.mfcc_hires_conf.items():
                config[k] = v
            config['delta_order'] = 2

            config['output_size'] = config['num_ceps'] * (config['delta_order'] + 1)
        elif config['type'] == 'mel':
            config['delta_order'] = 2
            config['num_mel_bins'] = 23

            config['output_size'] = config['num_mel_bins'] * (config['delta_order'] + 1)
        else:
            raise ValueError(config['type'], config)

    def __init__(self, type='mfcc_high_res', **kwargs):
        """
        See above `finalize_dogmatic_config` for valid values for `type` and
        `kwargs`.
        """
        self.type = type

        # output_size is only for documentation
        self.output_size = kwargs.pop('output_size')
        self.kwargs = kwargs
        self._load_audio = None

    def __repr__(self):
        kwargs = ', '.join([f'{k}={v!r}' for k, v in self.kwargs.items()])
        return (
            f'{self.__class__.__name__}'
            f'({self.type!r}, {kwargs}, output_size={self.output_size})'
        )

    def stft(self, time_signal):
        assert 'stft' in self.type or self.type in ['cepstrum'], self.type
        return pb.transform.stft(
            time_signal,
            **self.kwargs,
        )

    def istft(self, stft_signal, samples=None):
        """

        >>> def print_properties(array):
        ...     print(f'array(shape={array.shape}, dtype={array.dtype})')


        >>> config = FeatureExtractor.get_config({'type': 'stft'})
        >>> fe_stft = FeatureExtractor.from_config(config)
        >>> fe_stft
        FeatureExtractor('stft', size=1024, shift=256, window_length=1024, pad=True, fading=True, dft_norm='ortho', output_size=513)


        >>> audio = cbj.io.load('/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav')
        >>> print_properties(audio)
        array(shape=(46797,), dtype=float64)
        >>> samples, = audio.shape

        >>> print_properties(fe_stft([audio]))
        array(shape=(1, 186, 513), dtype=complex128)
        >>> print_properties(fe_stft.istft(fe_stft([audio])))
        array(shape=(1, 46848), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft([audio]), samples=samples))
        array(shape=(1, 46797), dtype=float64)

        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1023]))))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1024]))))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1025]))))
        array(shape=(1280,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1023])), samples=1023))
        array(shape=(1023,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1024])), samples=1024))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1025])), samples=1025))
        array(shape=(1025,), dtype=float64)

        """
        assert 'stft' in self.type or self.type in ['cepstrum'], self.type

        kwargs = {
            k: v
            for k, v in self.kwargs.items()
            if k not in ['pad']
        }

        if torch.is_tensor(stft_signal):
            time_signal = pt_istft(stft_signal, **kwargs, num_samples=samples)
        else:
            time_signal = pb.transform.istft(stft_signal, **kwargs, num_samples=samples)

        # if samples is not None:
        #     if self.kwargs['pad']:
        #         assert time_signal.shape[-1] >= samples, (time_signal.shape, samples)
        #         assert time_signal.shape[-1] < samples + self.kwargs['shift'], (time_signal.shape, samples)
        #         time_signal = time_signal[..., :samples]
        #     else:
        #         raise NotImplementedError(self.kwargs)

        return time_signal

    def __call__(self, channel_file_list):
        """
        Takes a channel file list as input and calculate features with the
        shape: time x channel x feature and dtype float32.
        The size of the feature dim is equal to self.output_size.
        """
        if self._load_audio is None:
            self._load_audio = isinstance(channel_file_list[0], (str, Path))

        if 'stft' in self.type or self.type in ['cepstrum']:
            if self._load_audio:
                channel_file_list = cbj.io.load(channel_file_list, list_to='array')

            if 'stft' in self.type:
                features = pb.transform.stft(
                    channel_file_list,
                    **self.kwargs,
                )

                if 'abs' in self.type:
                    features = np.abs(features)
            else:
                features = cepstrum(
                    channel_file_list,
                    **self.kwargs,
                )

            return features

        elif 'mel' in self.type or 'mfcc' in self.type:
            assert self._load_audio, self._load_audio
            if 'mel' in self.type:
                compute_feats = pb.kaldi.mfcc.compute_fbank_feats
            elif 'mfcc' in self.type:
                compute_feats = pb.kaldi.mfcc.compute_mfcc_feats
            else:
                raise ValueError(self.type)

            features = np.array(compute_feats(
                channel_file_list,
                is_scp=True,
                stacked=True,
                **self.kwargs,
            ))
        else:
            raise ValueError(self.type)

        features = einops.rearrange(
            features,
            'channel time freq -> time channel freq',
        ).astype(np.float32)
        return features

class AbstractFeatureExtractor(pt.Configurable):
    def __repr__(self):
        import inspect
        sig = inspect.signature(self.__class__)
        p: inspect.Parameter
        kwargs = ', '.join([
            f'{name}={getattr(self, name)!r}'
            for name, p in sig.parameters.items()
            if p.kind in [
                # inspect.Parameter.POSITIONAL_ONLY,  # (pos_only, /)
                inspect.Parameter.POSITIONAL_OR_KEYWORD,  # (pos_or_key)
                # inspect.Parameter.VAR_POSITIONAL,  # (*args)
                inspect.Parameter.KEYWORD_ONLY,  # (*, key_only)
                # inspect.Parameter.VAR_KEYWORD,  # (**kwargs)
            ]
        ])

        return f'{self.__class__.__name__}({kwargs})'


class STFT(AbstractFeatureExtractor):
    """
    Feature extractor properties:

        __call__:
            returns the features, allows files and array as input
        property output_size:
            length of the last features axis

    Feature extractor optional properties:

        method stft:
            returns the stft signal, where the number of frames matches with
            the number of frames of the features. The last axis may have a
            different length.

        method istft:
            Inverse of stft. Has argument samples to cut the signal to the
            original length.


    >>> fe = STFT()
    >>> fe
    STFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, dft_norm='ortho', output_size=513, window='blackman')

    >>> def print_properties(array):
    ...     print(f'array(shape={array.shape}, dtype={array.dtype})')
    >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'

    >>> print_properties(STFT()([file]))
    array(shape=(1, 186, 513), dtype=complex128)
    >>> print_properties(AbsSTFT()([file]))
    array(shape=(1, 186, 513), dtype=float64)
    >>> print_properties(Cepstrum()([file]))
    array(shape=(1, 186, 513), dtype=float64)
    >>> print_properties(KaldiSTFT()([file]))
    array(shape=(1, 290, 257), dtype=complex128)
    >>> print_properties(AbsKaldiSTFT()([file]))
    array(shape=(1, 290, 257), dtype=float64)
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config['window_length'] is None:
            config['window_length'] = cls.from_config({**config}).window_length
        if config['output_size'] is None:
            config['output_size'] = cls.from_config({**config}).output_size

    def __init__(
            self,
            size=1024,
            shift=256,
            window_length=None,
            pad=True,
            fading=True,
            # dft_norm='ortho',
            output_size=None,
            window='blackman'
    ):
        self.size = size
        self.shift = shift
        if window_length is None:
            self.window_length = size
        else:
            self.window_length = window_length
        self.pad = pad
        self.fading = fading
        # self.dft_norm = dft_norm
        self.output_size = self._get_output_size(output_size)
        self.window = window

        if callable(window):
            self._window = window
        elif isinstance(window, str):
            self._window = {
                'blackman': signal.windows.blackman,
                'hann': signal.windows.hann,
                'boxcar': signal.windows.boxcar,
                'triang': signal.windows.triang,
                'hamming':  signal.windows.hamming,
                'parzen': signal.windows.parzen,
                'cosine': signal.windows.cosine,
                'blackmanharris': signal.windows.blackmanharris,
                'flattop': signal.windows.flattop,
                'tukey': signal.windows.tukey,
                'bartlett': signal.windows.bartlett,
                'bohman': signal.windows.bohman,
                'kaiser2': functools.partial(signal.windows.kaiser, beta=2),
                'kaiser3': functools.partial(signal.windows.kaiser, beta=2),
            }[window]
        else:
            raise ValueError(window)

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.frequencies
        else:
            assert output_size == self.frequencies, (output_size, self.frequencies)
            return output_size

    @property
    def frequencies(self):
        return self.size // 2 + 1

    def stft(self, signal):
        kwargs = dict(
            size=self.size,
            shift=self.shift,
            axis=-1,
            window_length=self.window_length,
            window=self._window,
            fading=self.fading,
            pad=self.pad,
            # symmetric_window=False,
        )

        if pt_complex.is_torch(signal):
            return pt_stft(signal, **kwargs)
        else:
            return pb.transform.stft(signal, **kwargs)

    def istft(self, signal, num_samples=None):
        """

        Args:
            signal:
            samples:

        Returns:

        >>> def print_properties(array):
        ...     print(f'array(shape={array.shape}, dtype={array.dtype})')


        >>> fe_stft = STFT()
        >>> fe_stft
        STFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, dft_norm='ortho', output_size=513, window='blackman')


        >>> audio = cbj.io.load('/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav')
        >>> print_properties(audio)
        array(shape=(46797,), dtype=float64)
        >>> samples, = audio.shape

        >>> print_properties(fe_stft([audio]))
        array(shape=(1, 186, 513), dtype=complex128)
        >>> print_properties(fe_stft.istft(fe_stft([audio])))
        array(shape=(1, 46848), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft([audio]), samples=samples))
        array(shape=(1, 46797), dtype=float64)

        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1023]))))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1024]))))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1025]))))
        array(shape=(1280,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1023])), samples=1023))
        array(shape=(1023,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1024])), samples=1024))
        array(shape=(1024,), dtype=float64)
        >>> print_properties(fe_stft.istft(fe_stft(np.zeros([1025])), samples=1025))
        array(shape=(1025,), dtype=float64)

        """
        kwargs = dict(
            size=self.size,
            shift=self.shift,
            # axis=axis,
            window=self._window,
            window_length=self.window_length,
            fading=self.fading,
            pad=self.pad,
            # symmetric_window=False,
            # dft_norm=self.dft_norm,
            num_samples=num_samples,
        )

        if pt_complex.is_torch(signal):
            time_signal = pt_istft(signal, **kwargs)
        else:
            time_signal = pb.transform.istft(signal, **kwargs)

        # time_signal = pb.transform.istft(
        #     signal,
        #     size=self.size,
        #     shift=self.shift,
        #     # axis=axis,
        #     window=self._window,
        #     window_length=self.window_length,
        #     fading=self.fading,
        #     pad=self.pad,
        #     # symmetric_window=False,
        #     # dft_norm=self.dft_norm,
        #     num_samples=num_samples,
        # )

        # if num_samples is not None:
        #     if self.pad:
        #         assert time_signal.shape[-1] >= num_samples, (time_signal.shape, num_samples)
        #         assert time_signal.shape[-1] < num_samples + self.shift, (time_signal.shape, num_samples)
        #         time_signal = time_signal[..., :num_samples]
        #     else:
        #         raise NotImplementedError(self)

        return time_signal

    _load_audio = None

    def stft_to_feature(self, stft_signals):
        return stft_signals

    def __call__(self, signals):
        if self._load_audio is None:
            if isinstance(signals, (str, Path)):
                self._load_audio = True
            elif isinstance(signals, (tuple, list)) and isinstance(signals[0], (str, Path)):
                self._load_audio = True
            else:
                self._load_audio = False

        if self._load_audio:
            signals = cbj.io.load(signals, list_to='array')

        return self.stft_to_feature(self.stft(signals))


class AbsSTFT(STFT):
    def stft_to_feature(self, stft_signals):
        # if pt_complex.is_torch(signal):
        # else:
        # Should work with numpy as torch
        return abs(stft_signals)


class AbsIPDSTFT(STFT):

    def _get_output_size(self, output_size):
        if output_size is None:
            return (self.size // 2 + 1) * 3
        else:
            assert output_size == self.frequencies * 3, (output_size, self.frequencies * 3)
            return output_size

    def append_interchannel_phase_differences(self, signals):
        #  (channels, ..., frequencies)
        return interchannel_phase_differences(
            signals, concatenate=True
        )

    def stft_to_feature(self, stft_signals):
        return self.append_interchannel_phase_differences(
            stft_signals
        )


class Log1pAbsSTFT(STFT):
    """
    >>> fe = Log1pAbsSTFT()
    >>> fe
    Log1pAbsSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=513, window='blackman')
    >>> fe.stft_to_feature(np.array([1, 5, 3+4j, -5]))
    array([0.69314718, 1.79175947, 1.79175947, 1.79175947])
    >>> fe(np.ones(10_000)).shape
    (43, 513)
    """
    def stft_to_feature(self, stft_signals):
        if pt_complex.is_torch(stft_signals):
            return torch.log1p(abs(stft_signals))
        else:
            return np.log1p(abs(stft_signals))


class Log1pCosSinAbsSTFT(STFT):
    """
    >>> fe = Log1pCosSinAbsSTFT()
    >>> fe
    Log1pCosSinAbsSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=1539, window='blackman')
    >>> fe.stft_to_feature(np.array([1, 5, 3+4j, -5]))
    array([ 6.93147181e-01,  1.79175947e+00,  1.79175947e+00,  1.79175947e+00,
            1.00000000e+00,  1.00000000e+00,  6.00000000e-01, -1.00000000e+00,
            0.00000000e+00,  0.00000000e+00,  8.00000000e-01,  1.22464680e-16])
    >>> fe(np.ones(10_000)).shape
    (43, 1539)
    """

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.frequencies * 3
        else:
            assert output_size == self.frequencies * 3, (output_size, self.frequencies * 3)
            return output_size

    def stft_to_feature(self, stft_signals):
        if pt_complex.is_torch(stft_signals):
            raise NotImplementedError()
            return torch.log1p(abs(stft_signals))
        else:
            angle = np.angle(stft_signals)
            return np.concatenate([
                np.log1p(abs(stft_signals)),
                np.cos(angle),
                np.sin(angle),
            ], axis=-1)


class AbsRealImagSTFT(STFT):
    """
    >>> fe = AbsRealImagSTFT()
    >>> fe
    AbsRealImagSTFT(size=1024, shift=256, window_length=1024, pad=True, fading=True, output_size=1539, window='blackman')
    >>> fe.stft_to_feature(np.array([1, 5, 3+4j, -5]))
    array([ 1.,  5.,  5.,  5.,  1.,  5.,  3., -5.,  0.,  0.,  4.,  0.])
    >>> fe(np.ones(10_000)).shape
    (43, 1539)
    >>> fe.stft_to_feature(torch.tensor(np.array([1, 5, 3+4j, -5])))
    tensor([ 1.,  5.,  5.,  5.,  1.,  5.,  3., -5.,  0.,  0.,  4.,  0.],
           dtype=torch.float64)
    >>> fe(torch.tensor(np.ones(10_000))).shape
    torch.Size([43, 1539])
    """

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.frequencies * 3
        else:
            assert output_size == self.frequencies * 3, (output_size, self.frequencies * 3)
            return output_size

    def stft_to_feature(self, stft_signals):
        if pt_complex.is_torch(stft_signals):
            concatenate = torch.cat
        else:
            concatenate = np.concatenate
        return concatenate([
            abs(stft_signals),
            stft_signals.real,
            stft_signals.imag,
        ], axis=-1)


class Cepstrum(STFT):
    def stft_to_feature(self, stft_signals):
        return stft_to_cepstrum(stft_signals)


class ScaleIndependentCepstrum(STFT):
    """
    >>> rng = np.random.RandomState(0)
    >>> a = rng.randn(17)
    >>> fe = ScaleIndependentCepstrum()
    >>> fe.stft_to_feature(a)
    array([ 1.53981744,  1.28939946, -0.51793477, -1.71679596, -0.98117105,
            1.32819865, -0.59023165,  0.49065686,  0.0707627 ,  0.04265497,
            1.60944661,  0.20507146, -0.89059183,  1.23656373, -0.00519145,
            0.52410475])
    >>> fe.stft_to_feature(a * 1000)
    array([ 1.53981744,  1.28939946, -0.51793477, -1.71679596, -0.98117105,
            1.32819865, -0.59023165,  0.49065686,  0.0707627 ,  0.04265497,
            1.60944661,  0.20507146, -0.89059183,  1.23656373, -0.00519145,
            0.52410475])
    """

    @property
    def frequencies(self):
        return super().frequencies - 1

    def stft_to_feature(self, stft_signals):
        return stft_to_cepstrum(stft_signals)[..., 1:]


class KaldiSTFT(STFT):
    def __init__(
            self,
            size=512,
            shift=160,
            window_length=400,
            pad=False,
            fading=False,
            output_size=None,
            # dft_norm='ortho',
    ):
        super().__init__(
            size=size,
            shift=shift,
            window_length=window_length,
            pad=pad,
            fading=fading,
            output_size=output_size,
            # dft_norm=dft_norm,
        )


class AbsKaldiSTFT(KaldiSTFT, AbsSTFT):
    pass


class Mel(STFT):
    def __init__(
            self,
            size=1024,
            shift=256,
            window_length=None,
            pad=True,
            fading=True,
            # dft_norm='ortho',
            output_size=None,
            window='blackman',

            num_mel_bins=40,
            low_freq=20,
            high_freq=-400,
            sample_rate=16000,
    ):
        """

        >>> from paderbox.notebook import pprint
        >>> fe = Mel()
        >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'
        >>> import cbj
        >>> audio = cbj.io.load(file)
        >>> pprint(audio)
        array(shape=(46797,), dtype=float64)
        >>> pprint(fe(audio))
        array(shape=(186, 80), dtype=float64)
        >>> pprint(fe([audio]))
        array(shape=(1, 186, 80), dtype=float64)

        >>> pprint(HighResMFCC(stacked=True)([file]))
        array(shape=(290, 1, 120), dtype=float32)
        >>> pprint(HighResMFCC(stacked=False, num_mel_bins=80, num_ceps=80)([file]))
        array(shape=(290, 1, 3, 80), dtype=float32)

        """
        super().__init__(
            size=size, shift=shift, window_length=window_length,
            pad=pad, fading=fading, output_size=output_size,
            window=window,
        )

        from padertorch.contrib.je.modules.features import get_fbanks

        self.num_mel_bins = num_mel_bins

        fbanks = get_fbanks(
            n_mels=num_mel_bins,
            fft_length=size,
            sample_rate=sample_rate,
            fmin=low_freq,
            fmax=high_freq,
        ).astype(np.float32)
        fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
        self.fbanks = np.swapaxes(fbanks, -1, -2)

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.num_mel_bins
        else:
            assert output_size == self.num_mel_bins, (output_size, self.num_mel_bins)
            return output_size

    def stft_to_feature(self, stft_signals):
        return np.abs(stft_signals) @ self.fbanks



class Random_ABC:
    """
    Dummy class for isinstance check
    """
    pass


class RandomMel(STFT, Random_ABC):
    def __init__(
            self,
            size=1024,
            shift=256,
            window_length=None,
            pad=True,
            fading=True,
            # dft_norm='ortho',
            output_size=None,
            window='blackman',

            training=True,

            num_mel_bins=40,
            low_freq=20,
            high_freq=-400,
            sample_rate=16000,
            alpha_max=1.2,
    ):
        """
        >>> from paderbox.notebook import pprint
        >>> np.random.seed(0)
        >>> fe = RandomMel()
        >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'
        >>> import cbj
        >>> audio = cbj.io.load(file)
        >>> pprint(audio)
        array(shape=(46797,), dtype=float64)
        >>> pprint(fe(audio))
        array(shape=(186, 80), dtype=float64)
        >>> pprint(fe([audio, audio]))
        array(shape=(2, 186, 80), dtype=float64)
        >>> pprint(fe([[audio, audio, audio], [audio, audio, audio]]))
        array(shape=(2, 3, 186, 80), dtype=float64)
        >>> feature1 = fe(audio)
        >>> feature2 = fe(audio)
        >>> np.allclose(feature1, feature2)  # the features are random
        False

        """
        self.num_mel_bins = num_mel_bins
        super().__init__(
            size=size, shift=shift, window_length=window_length,
            pad=pad, fading=fading, output_size=output_size,
            window=window,
        )

        self.training = training

        self.fbanks_kwargs = dict(
            n_mels=num_mel_bins,
            fft_length=size,
            sample_rate=sample_rate,
            fmin=low_freq,
            fmax=high_freq,
        )
        self.alpha_max = alpha_max

    def fbanks(self, indipendent_shape, dtype):
        from padertorch.contrib.je.modules.features import get_fbanks
        from padertorch.contrib.je.modules.augment import (
            linear_warping, uniform_sampling_fn, log_uniform_sampling_fn
        )

        if self.training:
            warping_fn = functools.partial(
                linear_warping, n=indipendent_shape,
                alpha_sampling_fn=functools.partial(
                    log_uniform_sampling_fn, scale=2 * np.log(self.alpha_max)
                ),
                fhi_sampling_fn=functools.partial(
                    uniform_sampling_fn, center=.7, scale=.2
                ),
            )
        else:
            warping_fn = None

        fbanks = get_fbanks(
            **self.fbanks_kwargs,
            warping_fn=warping_fn,
        ).astype(dtype)
        fbanks = fbanks / (fbanks.sum(axis=-1, keepdims=True) + 1e-6)
        return np.swapaxes(fbanks, -1, -2)

    def _get_output_size(self, output_size):
        if output_size is None:
            return self.num_mel_bins
        else:
            assert output_size == self.num_mel_bins, (output_size, self.num_mel_bins)
            return output_size
    
    def stft_to_feature(self, stft_signals):
        spectrum = np.abs(stft_signals)
        return spectrum @ self.fbanks(stft_signals.shape[:-2], dtype=spectrum.dtype)#[..., None, :, :]

class IPD_ABC:
    """
    Dummy class for isinstance check
    """
    pass

class RandomIPDMel(RandomMel, IPD_ABC):
    """
    >>> from paderbox.notebook import pprint
    >>> np.random.seed(0)
    >>> fe = RandomIPDMel()
    >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'
    >>> import cbj
    >>> audio = cbj.io.load(file)
    >>> audio = np.array([audio, audio])
    >>> pprint(audio)
    array(shape=(2, 46797), dtype=float64)
    >>> pprint(fe(audio))
    array(shape=(186, 1106), dtype=float64)
    >>> pprint(fe([audio, audio, audio]))
    array(shape=(3, 186, 1106), dtype=float64)
    >>> pprint(fe([[audio, audio, audio]] * 4))
    array(shape=(4, 3, 186, 1106), dtype=float64)
    >>> feature1 = fe(audio)
    >>> feature2 = fe(audio)
    >>> np.allclose(feature1, feature2)  # the features are random
    False

    """
    def _get_output_size(self, output_size):
        expected_output_size = 2 * self.num_mel_bins + 2 * self.frequencies
        if output_size is None:
            return expected_output_size
        else:
            assert output_size == expected_output_size, (output_size, expected_output_size, self.num_mel_bins, self.frequencies)
            return output_size

    def stft_to_feature(self, stft_signals):
        assert stft_signals.shape[-3] == 2, stft_signals.shape
        spectrum = np.abs(stft_signals)
        mel_features = spectrum @ self.fbanks(stft_signals.shape[:-3] + (1,), dtype=spectrum.dtype)
        assert mel_features.shape[-3] == 2, mel_features.shape

        mel_features_1 = np.take(mel_features, 0, axis=-3)
        mel_features_2 = np.take(mel_features, 1, axis=-3)

        ipd = interchannel_phase_differences_op(
            np.take(spectrum, 0, axis=-3),
            np.take(spectrum, 0, axis=-3)
        )

        return np.concatenate([mel_features_1, mel_features_2, ipd.real, ipd.imag], axis=-1)


class HighResMFCC(AbstractFeatureExtractor):
    _is_scp = None
    def __init__(
            self,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20,
            high_freq=-400,
            delta_order=2,
            output_size=None,
            stacked=False,
            channel_first=False,
    ):
        """
        >>> def string_function(array):
        ...     return f'array(shape={array.shape}, dtype={array.dtype})'
        >>> np.set_string_function(string_function)
        >>> config = FeatureExtractor.get_config({'type': 'mfcc_high_res'})
        >>> fe_mfcc_high_res = FeatureExtractor.from_config(config)
        >>> fe_mfcc_high_res
        FeatureExtractor('mfcc_high_res', num_mel_bins=40, num_ceps=40, low_freq=20, high_freq=-400, delta_order=2, output_size=120)
        >>> file = '/net/vol/jenkins/python_unittest_data/timit/data/sample_1.wav'
        >>> fe_mfcc_high_res([file])
        array(shape=(290, 1, 120), dtype=float32)

        >>> HighResMFCC(stacked=True)([file])
        array(shape=(290, 1, 120), dtype=float32)
        >>> HighResMFCC(stacked=False, num_mel_bins=80, num_ceps=80)([file])
        array(shape=(290, 1, 3, 80), dtype=float32)
        """
        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.delta_order = delta_order
        if output_size is None:
            if stacked:
                output_size = num_ceps * (delta_order + 1)
            else:
                output_size = ((delta_order + 1), num_ceps)
        self.output_size = output_size
        self.stacked = stacked
        self.channel_first = channel_first

    def __call__(self, signals):
        if self._is_scp is None:
            if isinstance(signals, (str, Path)):
                self._is_scp = True
            elif isinstance(signals, (tuple, list)) and isinstance(signals[0], (str, Path)):
                self._is_scp = True
            else:
                self._is_scp = False

        if isinstance(signals, np.ndarray) and signals.ndim == 2:
            signals = list(signals)

        features = pc.kaldi.mfcc.compute_mfcc_feats(
            signals,
            num_mel_bins=self.num_mel_bins,
            num_ceps=self.num_ceps,
            low_freq=self.low_freq,
            high_freq=self.high_freq,
            is_scp=self._is_scp,
            delta_order=self.delta_order,
            stacked=self.stacked,

        )
        features = np.array(features)
        if self.channel_first:
            pass
        else:
            features = einops.rearrange(
                features,
                'channel time ... -> time channel ...',
        )#.astype(np.float32)
        return features

    #
    #     pb.kaldi.mfcc.compute_fbank_feats
    #     elif 'mfcc' in self.type:
    #     compute_feats = pb.kaldi.mfcc.compute_mfcc_feats
    #
    # else:
    # raise ValueError(self.type)
    #
#
# features = np.array(compute_feats(
#     channel_file_list,
#     is_scp=True,
#     stacked=True,
#     **self.kwargs,
# ))
# else:
# raise ValueError(self.type)
#
# features = einops.rearrange(
#     features,
#     'channel time freq -> time channel freq',
# ).astype(np.float32)
# return features
