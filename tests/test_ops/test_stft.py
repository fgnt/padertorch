import unittest

import numpy as np
import paderbox.testing as tc
import torch
from paderbox.io import load_audio
from paderbox.testing.testfile_fetcher import get_file_path
from paderbox.transform import stft, istft
from padertorch.ops.stft import STFT


class TestSTFTMethods(unittest.TestCase):
    size = 1024
    shift = 256
    window_length = 1024
    window = 'blackman'
    fading = 'full'

    @classmethod
    def setUpClass(self):
        path = get_file_path("sample.wav")

        self.time_signal = load_audio(path)
        # self.time_signal = np.random.randn(5, 3, 5324)
        self.torch_signal = torch.from_numpy(self.time_signal)
        self.stft = STFT(size=self.size, shift=self.shift,
                         window_length=self.window_length, fading=self.fading,
                         complex_representation='concat', window=self.window)
        self.fbins = self.stft.size // 2 + 1

    def test_restore_time_signal_from_stft_and_istft(self):
        x = self.time_signal
        X = self.stft(self.torch_signal)

        tc.assert_almost_equal(
            self.stft.inverse(X)[..., :x.shape[-1]].numpy(), x)
        tc.assert_equal(X.shape, (154, self.fbins * 2))

    def test_stft_frame_count(self):
        stft = self.stft
        stft.fading = False
        x = torch.rand(size=[1023])
        X = stft(x)
        tc.assert_equal(X.shape, (1, self.fbins * 2))

        x = torch.rand(size=[1024])
        X = stft(x)
        tc.assert_equal(X.shape, (1, self.fbins * 2))

        x = torch.rand(size=[1025])
        X = stft(x)
        tc.assert_equal(X.shape, (2, self.fbins * 2))

        stft.fading = True
        x = torch.rand(size=[1023])
        X = stft(x)
        tc.assert_equal(X.shape, (7, self.fbins * 2))

        x = torch.rand(size=[1024])
        X = stft(x)
        tc.assert_equal(X.shape, (7, self.fbins * 2))

        x = torch.rand(size=[1025])
        X = stft(x)
        tc.assert_equal(X.shape, (8, self.fbins * 2))

    def test_compare_stft_to_numpy(self):
        X_numpy = stft(self.time_signal, size=self.size, shift=self.shift,
                       window_length=self.window_length, window=self.window,
                       fading=self.fading)
        X_numpy = np.concatenate([np.real(X_numpy), np.imag(X_numpy)], axis=-1)
        X_torch = self.stft(self.torch_signal).numpy()
        tc.assert_almost_equal(X_torch, X_numpy)

    def test_restore_time_signal_from_numpy_stft_and_torch_istft(self):
        X_numpy = stft(self.time_signal, size=self.size, shift=self.shift,
                       window_length=self.window_length, window=self.window,
                       fading=self.fading)
        X_numpy = np.concatenate([np.real(X_numpy), np.imag(X_numpy)], axis=-1)
        x_torch = self.stft.inverse(torch.from_numpy(X_numpy))
        x_numpy = x_torch.numpy()[..., :self.time_signal.shape[-1]]
        tc.assert_almost_equal(x_numpy, self.time_signal)

    def test_restore_time_signal_from_torch_stft_and_numpy_istft(self):
        X_torch = self.stft(self.torch_signal).numpy()
        X_numpy = X_torch[..., :self.fbins] + 1j * X_torch[..., self.fbins:]
        x_numpy = istft(X_numpy, size=self.size, shift=self.shift,
                        window_length=self.window_length, window=self.window,
                        fading=self.fading)[..., :self.time_signal.shape[-1]]
        tc.assert_almost_equal(x_numpy, self.time_signal)


class TestSmallerSTFTMethods(TestSTFTMethods):
    size = 512
    shift = 20
    window_length = 40
    window = 'hamming'
    fading = 'full'

    def test_restore_time_signal_from_stft_and_istft(self):
        x = self.time_signal
        X = self.stft(self.torch_signal)

        tc.assert_almost_equal(
            self.stft.inverse(X)[..., :x.shape[-1]].numpy(), x)
        tc.assert_equal(X.shape, (1927, self.fbins * 2))

    def test_stft_frame_count(self):
        stft = self.stft
        stft.fading = False
        x = torch.rand(size=[1019])
        X = stft(x)
        tc.assert_equal(X.shape, (50, self.fbins * 2))

        x = torch.rand(size=[1020])
        X = stft(x)
        tc.assert_equal(X.shape, (50, self.fbins * 2))

        x = torch.rand(size=[1021])
        X = stft(x)
        tc.assert_equal(X.shape, (51, self.fbins * 2))

        stft.fading = True
        x = torch.rand(size=[1019])
        X = stft(x)
        tc.assert_equal(X.shape, (52, self.fbins * 2))

        x = torch.rand(size=[1020])
        X = stft(x)
        tc.assert_equal(X.shape, (52, self.fbins * 2))

        x = torch.rand(size=[1021])
        X = stft(x)
        tc.assert_equal(X.shape, (53, self.fbins * 2))
