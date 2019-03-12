"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.wavenet.train print_config
python -m padertorch.contrib.examples.wavenet.train
"""
import os
from collections import OrderedDict
from pathlib import Path

from paderbox.utils.nested import deflatten
from paderbox.utils.timer import timeStamped
from padertorch.contrib.je.data import DataProvider
from padertorch.contrib.je.transforms import ReadAudio, STFT, Spectrogram, \
    MelTransform, SegmentAxis, Fragmenter
from padertorch.models.wavenet import WaveNet
from padertorch.train.optimizer import Adam
from padertorch.train.trainer import Trainer
from sacred import Experiment as Exp
from sacred.commands import print_config
from padertorch import Module
from paderbox.io.json_module import load_json
from scipy.io import wavfile
import torch

nickname = 'wavenet-inference'
ex = Exp(nickname)


@ex.config
def config():
    model_dir = ''
    assert len(model_dir) > 0, 'Set the model path on the command line.'
    config_name = str(Path('1') / 'config.json')
    checkpoint_name = 'ckpt_best_loss.pth'
    data_config = load_json(str(Path(model_dir) / config_name))['data_config']
    num_examples = 3
    storage_dir = str(Path(model_dir) / 'inferred')


@ex.capture
def get_model(model_dir, config_name, checkpoint_name):
    model_dir = Path(model_dir)
    model = Module.from_storage_dir(
        model_dir, config_name=config_name, checkpoint_name=checkpoint_name,
        in_config_path='train_config.model'
    )
    return model.cuda()


@ex.capture
def get_dataset(data_config):
    data_config['transforms'] = OrderedDict(
        reader=data_config['transforms']['reader'],
        stft=data_config['transforms']['stft'],
        spectrogram=data_config['transforms']['spectrogram'],
        mel_transform=data_config['transforms']['mel_transform'],
    )
    data_provider = DataProvider.from_config(data_config)

    def get_spec(example):
        return example["spectrogram"]

    test_iter = data_provider.get_test_iterator().map(get_spec)

    return test_iter


@ex.automain
def main(_run, num_examples, storage_dir):
    print_config(_run)
    model = get_model()
    dataset = get_dataset()
    os.makedirs(storage_dir, exist_ok=True)
    i = 0
    for x in dataset:
        x = model.wavenet.infer_gpu(torch.Tensor(x).transpose(1, 2))
        for audio in x.cpu().data.numpy():
            if i >= num_examples:
                break
            wavfile.write(
                str(Path(storage_dir) / f'{i}.wav'), model.sample_rate, audio
            )
            i += 1
        if i >= num_examples:
            break
