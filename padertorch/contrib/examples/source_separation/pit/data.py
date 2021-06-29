from functools import partial

import einops
import numpy as np

import paderbox as pb
import padertorch as pt
from paderbox.transform import stft


def prepare_iterable(
        db, dataset: str, batch_size, return_keys=None, prefetch=True, shuffle=True
):
    audio_keys = ['observation', 'speech_source']
    iterable = db.get_dataset(dataset)

    iterable = (
        iterable
        .map(partial(read_audio, audio_keys=audio_keys))
        .map(partial(pre_batch_transform, return_keys=return_keys))
    )
    if shuffle:
        iterable = iterable.shuffle(reshuffle=True)
    iterable = (
        iterable
        .batch(batch_size)
        .map(pt.data.batch.Sorter('num_samples'))
        .map(pt.data.utils.collate_fn)
        .map(post_batch_transform)
    )

    if prefetch:
        iterable = iterable.prefetch(4, 8)

    return iterable


def read_audio(example, src_key="audio_path", audio_keys=None):
    data = {
        audio_key: pb.io.audioread.recursive_load_audio(
            example[src_key][audio_key],
        )
        for audio_key in audio_keys
    }
    example["audio_data"] = data
    return example


def pre_batch_transform(inputs, return_keys=None):
    s = inputs['audio_data']['speech_source']
    y = inputs['audio_data']['observation']
    S = stft(s, 512, 128)
    Y = stft(y, 512, 128)
    Y = einops.rearrange(Y, 't f -> t f')
    S = einops.rearrange(S, 'k t f -> t k f')
    X = S  # Same for WSJ0_2MIX database
    num_frames = Y.shape[0]

    return_dict = dict()

    def maybe_add(key, value):
        if return_keys is None or key in return_keys:
            return_dict[key] = value

    maybe_add('example_id', inputs['example_id'])
    maybe_add('s', np.ascontiguousarray(s, np.float32))
    maybe_add('S', np.ascontiguousarray(S, np.float32))
    maybe_add('y', np.ascontiguousarray(y, np.float32))
    maybe_add('Y', np.ascontiguousarray(Y, np.complex64))
    maybe_add('X_abs', np.ascontiguousarray(np.abs(X), np.float32))
    maybe_add('Y_abs', np.ascontiguousarray(np.abs(Y), np.float32))
    maybe_add('num_frames', num_frames)
    maybe_add('cos_phase_difference', np.ascontiguousarray(
        np.cos(np.angle(Y[:, None, :]) - np.angle(X)), np.float32)
    )

    return return_dict


def post_batch_transform(batch):
    return batch


