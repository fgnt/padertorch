import typing
import math

from einops import rearrange

import numpy as np

import torch.nn.functional
import torch_complex

from paderbox.array import segment_axis
from paderbox.transform.module_stft import _biorthogonal_window_fastest, _get_window
from padertorch.contrib.cb.array import overlap_add


def stft(
        time_signal,
        size: int = 1024,
        shift: int = 256,
        *,
        # axis=-1,  # I never use this and it complicated the code
        window: [str, typing.Callable] = 'blackman',
        window_length: int = None,
        fading: typing.Optional[typing.Union[bool, str]] = 'full',
        pad: bool = True,
        symmetric_window: bool = False,
):
    """
    >>> import numpy as np
    >>> import random
    >>> from paderbox.transform.module_stft import stft as np_stft, istft as np_istft
    >>> import torch.fft
    >>> kwargs = dict(
    ...     size=np.random.randint(100, 200),
    ...     shift=np.random.randint(40, 100),
    ...     window=random.choice(['blackman', 'hann', 'hamming']),
    ...     fading=random.choice(['full', 'half', False]),
    ... )
    >>> num_samples = np.random.randint(200, 500)
    >>> a = np.random.rand(num_samples)
    >>> A_np = np_stft(a, **kwargs)
    >>> A_pt = stft(torch.tensor(a), **kwargs)
    >>> np.testing.assert_allclose(
    ...     A_np, A_pt.numpy(), err_msg=str(kwargs), atol=1e-10)

    """
    assert isinstance(time_signal, torch.Tensor)
    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    assert fading in [None, True, False, 'full', 'half'], (fading, type(fading))
    if fading not in [False, None]:
        if fading == 'half':
            pad_width = [
                (window_length - shift) // 2,
                math.ceil((window_length - shift) / 2),
            ]
        else:
            pad_width = [
                window_length - shift,
                window_length - shift,
            ]
        time_signal = torch.nn.functional.pad(time_signal, pad_width, mode='constant')

    window = _get_window(
        window=window,
        symmetric_window=symmetric_window,
        window_length=window_length,
    )
    window = torch.from_numpy(window).to(time_signal.device, dtype=time_signal.dtype)

    time_signal_seg = segment_axis(
        time_signal,
        window_length,
        shift=shift,
        axis=-1,
        end='pad' if pad else 'cut'
    )

    # if hasattr(torch, 'rfft'):
    if callable(torch.fft):
        if window_length != size:
            raise NotImplementedError(
                f'Old Torch does not support window_length != size\n'
                f'window_length = {window_length} != {size} = size'
            )

        out = torch.rfft(
            time_signal_seg * window,
            1,
            # size,
        )
        assert out.shape[-1] == 2, out.shape
        return torch_complex.ComplexTensor(out[..., 0], out[..., 1])
    else:
        return torch.fft.rfft(time_signal_seg * window, size)


def _complex_to_real(tensor):
    return torch.stack([tensor.real, tensor.imag], dim=-1)


def istft(
        stft_signal,
        size: int=1024,
        shift: int=256,
        *,
        window: [str, typing.Callable]='blackman',
        fading: typing.Optional[typing.Union[bool, str]] = 'full',
        window_length: int=None,
        symmetric_window: bool=False,
        num_samples: int=None,
        pad: bool=True,
        biorthogonal_window=None,
):
    """
    Why not torch.istft?
     - It failed for me sometimes with a strange error msg, when I used a
       blackman window.
     - It is slower than this function (Approx 2 times slower).
     - This can easily invert our numpy stft (i.e. the parameters match).
     - You can read the source code.
     - Take as input the correct dtype (i.e. complex and not float).
        - Note: In the long term torch_complex.ComplexTensor will be replaced
                with the native type.
    
    >>> import numpy as np
    >>> import random
    >>> import torch.fft
    >>> from paderbox.transform.module_stft import stft as np_stft, istft as np_istft
    >>> kwargs = dict(
    ...     size=np.random.randint(100, 200),
    ...     shift=np.random.randint(40, 70),
    ...     window=random.choice(['blackman', 'hann', 'hamming']),
    ...     fading=random.choice(['full', 'half', False]),
    ... )
    >>> num_samples = np.random.randint(200, 500)
    >>> a = np.random.rand(num_samples)
    >>> A_np = np_stft(a, **kwargs)
    >>> A_pt = stft(torch.tensor(a), **kwargs)
    >>> np.testing.assert_allclose(
    ...     A_np, A_pt.numpy(), err_msg=str(kwargs), atol=1e-10)
    >>> a_np = np_istft(A_np, **kwargs)
    >>> a_pt = istft(A_pt, **kwargs)
    >>> np.testing.assert_allclose(
    ...     a_np, a_pt.numpy(), err_msg=str(kwargs), atol=1e-10)

    >>> kwargs['window_length'] = np.random.randint(70, kwargs['size'] - 1)
    >>> num_samples = np.random.randint(200, 500)
    >>> a = np.random.rand(num_samples)
    >>> A_np = np_stft(a, **kwargs)
    >>> A_pt = stft(torch.tensor(a), **kwargs)
    >>> np.testing.assert_allclose(
    ...     A_np, A_pt.numpy(), err_msg=str(kwargs), atol=1e-10)
    >>> a_np = np_istft(A_np, **kwargs)
    >>> a_pt = istft(A_pt, **kwargs)
    >>> np.testing.assert_allclose(
    ...     a_np, a_pt.numpy(), err_msg=str(kwargs), atol=1e-10)

    >>> kwargs = dict(
    ...     size=4,
    ...     shift=2,
    ...     window='hann',
    ...     fading='full',
    ... )
    >>> num_samples = 8
    >>> a = np.arange(num_samples).astype(np.float32)
    >>> np_stft(a, **kwargs)
    array([[ 0.5+0.j ,  0. +0.5j, -0.5+0.j ],
           [ 4. +0.j , -2. +1.j ,  0. +0.j ],
           [ 8. +0.j , -4. +1.j ,  0. +0.j ],
           [12. +0.j , -6. +1.j ,  0. +0.j ],
           [ 3.5+0.j ,  0. -3.5j, -3.5+0.j ]])
    >>> a_pt_init = torch.tensor(a, requires_grad=True)
    >>> A_pt = stft(a_pt_init, **kwargs)
    >>> a_pt = istft(A_pt, **kwargs)
    >>> A_pt.grad
    >>> rnd_num = torch.randn(a_pt.shape)
    >>> torch.sum(a_pt * rnd_num).backward()
    >>> A_pt.grad
    >>> print(rnd_num, '\\n', a_pt_init.grad)
    """

    if window_length is None:
        window_length = size

    if biorthogonal_window is None:
        window = torch.from_numpy(
            _biorthogonal_window_fastest(
                _get_window(
                    window,
                    symmetric_window,
                    window_length
                ),
                shift)).to(
            device=stft_signal.device,
            dtype=stft_signal.real.dtype,
        )
    else:
        window = biorthogonal_window

    # if hasattr(torch, 'irfft'):
    if callable(torch.fft):
        assert isinstance(stft_signal, torch_complex.ComplexTensor), type(stft_signal)
        stft_signal = rearrange(_complex_to_real(
            stft_signal
        ), '... t f realimag -> ... t f realimag', realimag=2)
        stft_signal = torch.irfft(stft_signal, 1, signal_sizes=(size,))
    else:
        assert isinstance(stft_signal, torch.Tensor) and stft_signal.is_complex(), (type(stft_signal), stft_signal.dtype)
        stft_signal = torch.fft.irfft(stft_signal, size)
        if window_length != size:
            stft_signal = stft_signal[..., :window_length]

    stft_signal = stft_signal * window

    stft_signal = overlap_add(
        stft_signal, shift
    )

    # Remove the padding values from fading in the stft
    assert fading in [None, True, False, 'full', 'half'], fading
    if fading not in [None, False]:
        pad_width = (window_length - shift)
        if fading == 'half':
            pad_width /= 2
        stft_signal = stft_signal[
                      ..., int(pad_width):stft_signal.shape[-1] - math.ceil(
            pad_width)]

    if num_samples is not None:
        if pad:
            assert stft_signal.shape[-1] >= num_samples, (stft_signal.shape, num_samples)
            assert stft_signal.shape[-1] < num_samples + shift, (stft_signal.shape, num_samples)
            stft_signal = stft_signal[..., :num_samples]
        else:
            raise ValueError(
                pad,
                'When padding is False in the stft, the signal is cutted.'
                'This operation can not be inverted.')

    return stft_signal
