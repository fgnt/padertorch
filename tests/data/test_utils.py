import string
import unittest

import numpy as np
import torch

from padertorch.data.utils import Padder


class TestPadderBase(unittest.TestCase):
    to_torch = False
    sort_by_key = None
    padding = False
    padding_keys = None

    def setUp(self):

        self.T = 100
        self.B = 4
        self.E = 20
        self.K = 2
        self.F = 257
        self.length_transcription = 7
        self.num_frames = [70, 90, 80, 100]
        self.inputs = [
            {'Y_abs':
                 np.abs(np.random.normal(
                     size=(num_frames_, self.F)
                 )).astype(np.float32),
             'Y_stft': (
                 np.random.normal(size=(num_frames_, self.F)) +
                 1j* np.random.normal(size=(num_frames_, self.F))
                 ).astype(np.complex128),
             'target_mask':
                 np.abs(np.random.choice(
                     [0, 1],
                     size=(num_frames_, self.K, self.F)
                 )).astype(np.float32),
             'noise_mask':
                 np.abs(np.random.choice(
                     [0, 1],
                     size=(self.K, num_frames_, self.F)
                 )).astype(np.float32),
             'num_frames': num_frames_,
             'transcription': np.random.choice(list(string.ascii_letters),
                                               self.length_transcription)
             }
            for num_frames_ in self.num_frames
        ]
        self.padder = Padder(to_torch=self.to_torch,
                             sort_by_key=self.sort_by_key,
                             padding=self.padding,
                             padding_keys=self.padding_keys)
        self.padded = self.padder(self.inputs)
        if self.padding_keys is None:
            self.padding_keys_ = list(self.inputs[0].keys())
        else:
            self.padding_keys_ = self.padding_keys

    def test_length(self):
        if not self.padding:
            return
        if 'Y_abs' in self.padding_keys_:
            np.testing.assert_equal(
                self.padded['Y_abs'].shape,
                (len(self.num_frames), max(self.num_frames), self.F)
            )
        if 'Y_stft' in self.padding_keys_:
            np.testing.assert_equal(
                self.padded['Y_stft'].shape,
                (len(self.num_frames), max(self.num_frames), self.F)
            )
        if 'target_mask' in self.padding_keys_:
            np.testing.assert_equal(
                self.padded['target_mask'].shape,
                (len(self.num_frames), max(self.num_frames), self.K, self.F),
            )
        if 'noise_mask' in self.padding_keys_:
            np.testing.assert_equal(
                self.padded['noise_mask'].shape,
                (len(self.num_frames), self.K, max(self.num_frames), self.F),
            )
        self.assertEqual(len(self.padded['transcription']),
                         len(self.num_frames))
        self.assertEqual(len(self.padded['num_frames']),
                         len(self.num_frames))

    def test_if_torch(self):
        if not self.to_torch:
            return
        for key in self.padding_keys_:
            array = self.inputs[0][key]
            if isinstance(array, np.ndarray) and\
                    not array.dtype.kind in {'U', 'S'} and\
                    not array.dtype in [np.complex64, np.complex128]:
                self.assertTrue([isinstance(self.padded[key], torch.Tensor)])
            elif isinstance(array, np.ndarray) and (
                    array.dtype.kind in {'U', 'S'} or
                            array.dtype in [np.complex64, np.complex128]
            ):
                self.assertTrue([isinstance(self.padded[key], np.ndarray)])

    def test_asserts(self):
        if self.to_torch:
            self.assertTrue(self.padding)

    def test_is_list_if_not_padded(self):
        if self.padding is True:
            return
        self.assertTrue(all(
            [isinstance(value, list) for value in self.padded.values()]
        ))
        self.assertEqual(
            [(len(value)) for value in self.padded.values()],
            [(len(self.num_frames)) for _ in self.padded.values()]
        )


class TestPadderPadding(TestPadderBase):
    to_torch = False
    sort_by_key = None
    padding = True
    padding_keys = None


class TestPadderTorch(TestPadderBase):
    to_torch = True
    sort_by_key = None
    padding = True
    padding_keys = None
