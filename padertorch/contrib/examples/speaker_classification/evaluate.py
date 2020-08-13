"""
Example call:

export STORAGE_ROOT=<your desired storage root>
python -m padertorch.contrib.examples.speaker_classification.eval with model_path=<path/to/trained/model>
"""
import os
from pathlib import Path

from sacred import Experiment, commands
import numpy as np
import torch

import paderbox as pb
from paderbox.utils.timer import timeStamped

import padertorch as pt
from padertorch.contrib.examples.speaker_classification.data import get_datasets
from padertorch.contrib.examples.speaker_classification.train import get_model

ex = Experiment('speaker_clf')


@ex.config
def defaults():
    model_path = None
    load_ckpt = 'ckpt_best_loss.pth'
    batch_size = 1
    device = 'cpu'


@ex.automain
def main(model_path, load_ckpt, batch_size, device, _run):
    commands.print_config(_run)

    model_path = Path(model_path)
    eval_dir = Path(model_path) / 'eval' / timeStamped('')[1:]
    # eval_dir.mkdir(parents=True, exist_ok=False)

    model = get_model()
    model = model.load_checkpoint(model_path / 'checkpoints' / load_ckpt)
    model.to(device)
    # Turn on evaluation mode for, e.g., BatchNorm and Dropout modules
    model.eval()

    _, _, test_set = get_datasets(model_path, batch_size)

    misclassified_examples = dict()
    correct_classified_examples = dict()
    accuracies = list()
    for batch in test_set:
        output = model(pt.data.example_to_device(batch, device=device))
        prediction = torch.argmax(output, dim=-1).cpu().detach().numpy()
        confidence = torch.softmax(output, dim=-1).max(dim=-1).values.cpu()\
            .detach().numpy()
        label = np.array(batch['speaker_id'])
        hits = (label == prediction).astype('bool')
        accuracies.extend(hits.tolist())
        misclassified_examples.update({
            k: {
                'label': v1,
                'predicted_label': v2,
                'audio_path': v3,
                'confidence': f'{v4:.2%}',
            }
            for k, v1, v2, v3, v4 in zip(
                np.array(batch['example_id'])[~hits], label[~hits],
                prediction[~hits], np.array(batch['audio_path'])[~hits],
                confidence[~hits]
            )
        })
        # for each correct predicted label, collect the audio paths
        correct_classified_examples.update({
            k: correct_classified_examples[k] + [v]
            if k in correct_classified_examples.keys() else [v]
            for k, v in zip(
                prediction[hits], np.array(batch['audio_path'])[hits]
            )
        })

    accuracy = np.array(accuracies).astype('float').mean()
    misclassified_dir = eval_dir / 'misclassified_examples'
    for example_id, v in misclassified_examples.items():
        label, prediction_label, audio_path, _ = v.values()
        predicted_speaker_audio_path = \
            correct_classified_examples[prediction_label][0]
        example_dir = \
            misclassified_dir / f'{example_id}_{label}_{prediction_label}'
        example_dir.mkdir(parents=True)
        os.symlink(audio_path, example_dir / 'example.wav')
        os.symlink(
            predicted_speaker_audio_path,
            example_dir / 'predicted_speaker_example.wav'
        )
    outputs = dict(
        accuracy=f'{accuracy:.2%} ({np.sum(accuracies)}/{len(accuracies)})',
        misclassifications=misclassified_examples,
    )
    pb.io.dump_json(outputs, eval_dir / 'results.json')
