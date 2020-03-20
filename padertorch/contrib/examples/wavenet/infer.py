"""
Example call:

python -m padertorch.contrib.examples.wavenet.infer with exp_dir=/path/to/exp_dir
"""
import os
from pathlib import Path

import torch
from padertorch.contrib.examples.wavenet.train import get_datasets, get_model
from sacred import Experiment as Exp
from scipy.io import wavfile

nickname = 'wavenet-inference'
ex = Exp(nickname)


@ex.config
def config():
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the model path on the command line.'
    num_examples = 10


@ex.capture
def load_model(exp_dir):
    model = get_model()
    ckpt = torch.load(
        Path(exp_dir) / 'checkpoints' / 'ckpt_best_loss.pth',
        map_location='cpu'
    )
    model.load_state_dict(ckpt['model'])
    return model.cuda()


@ex.automain
def main(exp_dir, num_examples):
    model = load_model()
    _, _, test_set = get_datasets(exp_dir, max_length=10., batch_size=1)
    storage_dir = Path(exp_dir) / 'inferred'
    os.makedirs(str(storage_dir), exist_ok=True)
    i = 0
    for example in test_set:
        x = model.wavenet.infer_gpu(torch.Tensor(example['features']))
        for audio in x.cpu().data.numpy():
            if i >= num_examples:
                break
            wavfile.write(
                str(storage_dir / f'{i}.wav'), model.sample_rate, audio
            )
            i += 1
        if i >= num_examples:
            break
