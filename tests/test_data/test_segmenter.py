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


def test_fixed_anchor():
    segmenter = Segmenter(length=32000, include_keys=('x', 'y'),
                          shift=16000, anchor=10)
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(
            entry['x'], 10 + np.arange(idx * 16000, 16000 + (idx + 1) * 16000))
        np.testing.assert_equal(entry['x'], entry['y'])


def test_copy_keys():
    segmenter = Segmenter(length=32000, include_keys=('x', 'y'),
                          shift=16000, copy_keys='gender')
    ex = {'x': np.arange(65000), 'y': np.arange(65000),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    expected_keys = [key for key in ex.keys() if not key == 'num_samples']
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
    error = False
    try:
        segmenter(ex)
    except AssertionError:
        error = True
    assert error, segmenter


def test_include_none_with_torch():
    segmenter = Segmenter(length=32000, shift=16000)
    array = np.random.randn(5,10,64000)
    ex = {'x': array.copy(), 'y': array.copy(),
          'z': torch.tensor(array),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(entry['x'], entry['z'].numpy())
        np.testing.assert_equal(entry['x'], entry['y'])


def test_error_include_list():
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

    segmenter = Segmenter(length=32000, shift=16000,
                          include_keys=['x', 'y', 'z'],
                          axis={'x': 0, 'y': 1, 'z': -1})
    array = np.random.randn(65000, 5, 10)
    ex = {'x': array.copy(), 'y': array.copy().transpose(1,0,2),
          'z': torch.tensor(array.transpose(1,2,0)),
          'num_samples': 65000, 'gender': 'm'}
    segmented = segmenter(ex)
    assert type(segmented) == list, segmented
    for idx, entry in enumerate(segmented):
        assert all([key in entry.keys() for key in ex.keys()])
        np.testing.assert_equal(entry['x'], entry['z'].numpy().transpose(2,0,1))
        np.testing.assert_equal(entry['x'], entry['y'].transpose(1,0,2))


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


def test_length_mode():
    examples = [{'x': np.arange(16000), 'y': np.arange(16000),
                 'num_samples': 16000, 'gender': 'm'},
                {'x': np.arange(15900), 'y': np.arange(15900),
                 'num_samples': 15900, 'gender': 'm'}]
    new_length = [{'constant': 950, 'max': 942, 'min': 1000},
                  {'constant': 950, 'max': 936, 'min': 994}]
    for mode in ['constant', 'max', 'min']:
        for idx, ex in enumerate(examples):
            segmenter = Segmenter(length=950, include_keys=('x'),
                                  mode=mode, padding=True)
            segmented = segmenter(ex)
            np.testing.assert_equal(segmented[0]['x'],
                                    np.arange(0, new_length[idx][mode]))
    new_length = [{'constant': 950, 'max': 947, 'min': 950},
                  {'constant': 950, 'max': 950, 'min': 954}]
    for mode in ['constant', 'max', 'min']:
        for idx, ex in enumerate(examples):
            segmenter = Segmenter(length=950, shift=250, include_keys=('x'),
                                  mode=mode, padding=True)
            segmented = segmenter(ex)
            np.testing.assert_equal(segmented[0]['x'],
                                    np.arange(0, new_length[idx][mode]))