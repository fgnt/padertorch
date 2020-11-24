from padertorch.data.fragmenter import Segmenter
import numpy as np
import torch

def test_simple_case():
    segmenter = Segmenter(length=32000, include_keys=('x', 'y'),
                          shift=16000)
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_copy_keys():
    segmenter = Segmenter(length=32000, include_keys=('x', 'y'),
                          shift=16000, copy_keys='gender')
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    expected_keys = {key: value for key, value in ex.items()
                     if not key == 'num_samples'}.keys()
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in expected_keys])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_include_none():
    segmenter = Segmenter(length=32000, shift=16000)
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_include_none_ignore_torch():
    segmenter = Segmenter(length=32000, shift=16000)
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'z': torch.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))

    segmenter = Segmenter(length=32000, shift=16000,
                          copy_keys=['num_samples', 'gender'])
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    expected_keys = ['x', 'y', 'num_samples', 'gender']
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in expected_keys])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_error_include():
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['x', 'y', 'z'])
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'z': torch.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    error = False
    try:
        segmented = segmenter(ex)
    except ValueError:
        error = True
    assert error, segmented
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['x', 'y', 'z'])
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'z': np.arange(65000).tolist(),
          'num_samples': 65000, 'gender': 'm'}
    error = False
    try:
        segmented = segmenter(ex)
    except ValueError:
        error = True
    assert error, segmented


def test_include_none_ignore_list():
    segmenter = Segmenter(length=32000, shift=16000)
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'z': np.arange(65000).tolist(),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))

    segmenter = Segmenter(length=32000, shift=16000,
                          copy_keys=['num_samples', 'gender'])
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    expected_keys = ['x', 'y', 'num_samples', 'gender']
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in expected_keys])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_include_exclude():
    segmenter = Segmenter(length=32000, shift=16000, exclude_keys='y')
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        assert entry['x'] != entry['y']

def test_axis():
    segmenter = Segmenter(length=32000, shift=16000, include_keys=['x', 'y'],
                          axis=[-1, 0])
    ex = {'x': np.arange(65000), 'y': np.arange(65000)[:, None],
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'][:, 0])

def test_wildcard():
    segmenter = Segmenter(length=32000, shift=16000, include_keys=['audio'])
    ex = {'audio': {'x': np.arange(65000), 'y': np.arange(65000)},
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['audio']['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['audio']['x'], entry['audio']['y'])

    segmenter = Segmenter(length=32000, shift=16000, include_keys=['audio'],
                          axis={'audio.x': -1, 'audio.y': 0})
    ex = {'audio': {'x': np.arange(65000), 'y': np.arange(65000)[:, None],},
          'z': np.arange(65000)[:, None],
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['audio']['x'],
            np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['audio']['x'], entry['audio']['y'][:, 0])
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_equal, entry['audio']['x'], entry['z']
        )
