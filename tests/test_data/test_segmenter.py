from padertorch.data.segment import Segmenter
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


def test_include_to_larger():
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['x', 'y', 'z'])
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
        segmenter(ex)
    except ValueError:
        error = True
    assert error, segmenter
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['x', 'y', 'z'])
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'z': np.arange(65000).tolist(),
          'num_samples': 65000, 'gender': 'm'}
    error = False
    try:
        segmenter(ex)
    except ValueError:
        error = True
    assert error, segmenter


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
        np.testing.assert_equal(entry['y'], np.arange(65000))


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


def test_axis_dict():
    segmenter = Segmenter(length=32000, shift=16000, include_keys=['x', 'y'],
                          axis={'x': -1, 'y': 0})
    ex = {'x': np.arange(65000), 'y': np.arange(65000)[:, None],
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'][:, 0])


def test_axis_dict_wildcard():
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['audio_data'],
                          axis={'audio_data': -1})
    ex = {'audio_data': {'x': np.arange(65000), 'y': np.arange(65000)},
          'z': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['audio_data']['x'],
            np.arange(idx * 16000, 16000 + (idx + 1) * 16000)
        )
        np.testing.assert_equal(entry['audio_data']['x'],
                                entry['audio_data']['y'])
        np.testing.assert_equal(entry['z'],
                                np.arange(65000))


def test_wildcard():
    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['audio_data'])
    ex = {'audio_data': {'x': np.arange(65000), 'y': np.arange(65000)},
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['audio_data']['x'], np.arange(
                idx * 16000, 16000 + (idx + 1) * 16000)
        )
        np.testing.assert_equal(entry['audio_data']['x'],
                                entry['audio_data']['y'])


def test_wildcard_exclude():
    ex = {
        'audio_data': {'x': np.arange(65000), 'y': np.arange(65000)[:, None]},
        'z': np.arange(65000)[:, None],
        'num_samples': 65000, 'gender': 'm'
    }

    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['audio_data'],
                          exclude_keys=['audio_data.y'],
                          axis={'audio_data': -1})
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['audio_data']['x'],
            np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['audio_data']['y'],
                                np.arange(65000)[:, None])
