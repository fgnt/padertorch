import einops
import numpy as np

import paderbox as pb
import padertorch as pt
from paderbox.transform import stft


def prepare_dataset(db, dataset_name: str, batch_size, prefetch=True, shuffle=True):
    """
    Prepares the dataset for the training process (loading audio data, STFT)

    Args:
        db: database object
        dataset_name: name of the dataset that should be used
        batch_size: batch size for the training
        prefetch: should the data be prefetched
        shuffle: should the data be shuffeled

    Returns:
        desired dataset of the database, the dataset is prepared for training
    """
    dataset = db.get_dataset(dataset_name)

    #Loading of the dataset and preparation of the data (STFT), the lambda function fixes the audio keys for read_audio
    dataset = (dataset.map(read_audio).map(pre_batch_transform))

    if shuffle:
        dataset = dataset.shuffle(reshuffle=True)

    #Splitting the dataset in batches and sorts examples in a batch w.r.t. their duration
    dataset = (
        dataset
        .batch(batch_size)
        .map(pt.data.batch.Sorter('num_frames'))
        .map(pt.data.utils.collate_fn)
    )

    if prefetch:
        dataset = dataset.prefetch(4, 8)

    return dataset



def read_audio(example):
    """
    Loading of the audio data for an element of the dataset.

    Args:
        example: for which the audio is loaded

    Returns:
        example with loaded audio data
    """

    audio_keys = ['observation', 'speech_source']

    data = {
        audio_key: pb.io.audioread.recursive_load_audio(
            example['audio_path'][audio_key],
        )
        for audio_key in audio_keys
    }
    example['audio_data'] = data
    return example


def pre_batch_transform(inputs):
    """
    Prepares the data by creating a dictionary with all data that is necessary for the model (e.g. STFT of observation).
    Explanation of some keys:
        s: speech source
        x: speech image
        y: observation
        Capital letters: STFT of signal
        
    Args:
        inputs: element of the database
    Returns:
        dictionary with necessary data
    """
    s = inputs['audio_data']['speech_source']
    y = inputs['audio_data']['observation']
    Y = einops.rearrange(stft(y, 512, 128), 't f -> t f')
    S = einops.rearrange(stft(s, 512, 128), 'k t f -> t k f')
    X = S  # Same for WSJ0_2MIX database
    num_frames = Y.shape[0]

    return_dict = dict()

    return_dict['example_id'] = inputs['example_id']
    return_dict['s'] = np.ascontiguousarray(s, np.float32)
    return_dict['S'] = np.ascontiguousarray(S, np.float32)
    return_dict['y'] = np.ascontiguousarray(y, np.float32)
    return_dict['Y'] = np.ascontiguousarray(Y, np.complex64)
    return_dict['X_abs'] = np.ascontiguousarray(np.abs(X), np.float32)
    return_dict['Y_abs'] = np.ascontiguousarray(np.abs(Y), np.float32)
    return_dict['num_frames'] = num_frames
    return_dict['cos_phase_difference'] = np.ascontiguousarray(np.cos(np.angle(Y[:, None, :]) - np.angle(X)), np.float32)

    return return_dict


