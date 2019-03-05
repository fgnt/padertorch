from functools import partial

import einops
import numpy as np
import padertorch as pt
from paderbox.database.iterator import AudioReader
from paderbox.database.keys import *
from paderbox.transform import stft
from paderbox.speech_enhancement import ideal_binary_mask


def pre_batch_transform(inputs, return_keys=None):
    s = inputs['audio_data']['speech_source']
    y = inputs['audio_data']['observation']
    S = stft(s, 512, 128)
    Y = stft(y, 512, 128)
    Y = einops.rearrange(Y, 't f -> t f')
    S = einops.rearrange(S, 'k t f -> t k f')
    X = S  # Same for MERL database
    num_frames = Y.shape[0]

    return_dict = dict()

    def maybe_add(key, value):
        if return_keys is None or key in return_keys:
            return_dict[key] = value

    maybe_add('example_id', inputs['example_id'])
    maybe_add('s', np.ascontiguousarray(s, np.float32))
    maybe_add('y', np.ascontiguousarray(y, np.float32))
    maybe_add('Y', np.ascontiguousarray(Y, np.complex64))
    maybe_add('X_abs', np.ascontiguousarray(np.abs(X), np.float32))
    maybe_add('Y_abs', np.ascontiguousarray(np.abs(Y), np.float32))
    maybe_add('num_frames', num_frames)
    maybe_add('cos_phase_difference', np.ascontiguousarray(
        np.cos(np.angle(Y[:, None, :]) - np.angle(X)), np.float32)
    )

    if return_keys is None or 'target_mask' in return_keys:
        return_dict['target_mask'] = np.ascontiguousarray(
            ideal_binary_mask(S, source_axis=-2), np.float32
        )

    return return_dict


def post_batch_transform(batch):
    return batch


def prepare_iterable(
        db, dataset: str, batch_size, return_keys=None, prefetch=True,
        iterator_slice=None
):
    audio_keys = [OBSERVATION, SPEECH_SOURCE]
    audio_reader = AudioReader(audio_keys=audio_keys, read_fn=db.read_fn)
    iterator = db.get_iterator_by_names(dataset)

    if iterator_slice is not None:
        iterator = iterator[iterator_slice]

    iterator = (
        iterator
        .map(audio_reader)
        .map(partial(pre_batch_transform, return_keys=return_keys))
        .shuffle(reshuffle=False)
        .batch(batch_size)
        .map(lambda batch: sorted(
            batch,
            key=lambda example: example["num_frames"],
            reverse=True,
        ))
        .map(pt.data.utils.collate_fn)
        .map(post_batch_transform)
        .tile(reps=50, shuffle=True)  # Simulates reshuffle to some degree
    )

    if prefetch:
        iterator = iterator.prefetch(4, 8)

    return iterator
