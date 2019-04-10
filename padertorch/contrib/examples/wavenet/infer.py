"""
Example call:

python -m padertorch.contrib.examples.wavenet.infer with model_dir=/path/to/storage_root
"""
import os
from pathlib import Path

import torch
from paderbox.io.json_module import load_json
from padertorch import Module
from padertorch.contrib.je.data.data_provider import DataProvider
from sacred import Experiment as Exp
from sacred.commands import print_config
from scipy.io import wavfile

nickname = 'wavenet-inference'
ex = Exp(nickname)


@ex.config
def config():
    model_dir = ''
    assert len(model_dir) > 0, 'Set the model path on the command line.'
    config_name = str(Path('1') / 'config.json')
    checkpoint_name = 'ckpt_best_loss.pth'
    data_config = load_json(str(Path(model_dir) / config_name))['data_config']
    dataset_names = 'test'
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
def get_dataset(data_config, dataset_names):
    data_provider = DataProvider.from_config(data_config)
    dataset = data_provider.get_iterator(dataset_names, shuffle=True)

    def get_spec(example):
        return example["spectrogram"]

    return dataset.map(get_spec)


@ex.automain
def main(_run, num_examples, storage_dir):
    print_config(_run)
    model = get_model()
    dataset = get_dataset()
    os.makedirs(storage_dir, exist_ok=True)
    i = 0
    for x in dataset:
        x = model.wavenet.infer_gpu(torch.Tensor(x))
        for audio in x.cpu().data.numpy():
            if i >= num_examples:
                break
            wavfile.write(
                str(Path(storage_dir) / f'{i}.wav'), model.sample_rate, audio
            )
            i += 1
        if i >= num_examples:
            break
