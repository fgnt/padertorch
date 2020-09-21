"""
Example calls:

On single CPU (slow):
python -m padertorch.contrib.examples.speaker_classification.supervised.evaluate with model_path=<path/to/trained/model> device=cpu

On GPU:
python -m padertorch.contrib.examples.speaker_classification.supervised.evaluate with model_path=<path/to/trained/model> device=0 batch_size=16

On multiple CPUs:
mpiexec -np 8 python -m padertorch.contrib.examples.speaker_classification.supervised.evaluate with model_path=<path/to/trained/model> device=cpu
"""
import os
from pathlib import Path
import warnings

from sacred import Experiment, commands
import numpy as np
import torch

from dlp_mpi import COMM, IS_MASTER, MASTER, split_managed

from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json

import padertorch as pt

from .data import get_datasets

ex = Experiment('speaker_clf')


@ex.config
def defaults():
    model_path = None
    assert model_path is not None, (
        'model_path cannot be None.\n'
        'Start the evaluation with "python -m padertorch.contrib.examples.'
        'speaker_classification.evaluate with model_path=<path/to/trained/model>"'
    )
    load_ckpt = 'ckpt_best_loss.pth'
    batch_size = 1
    device = 0 if torch.cuda.is_available() else 'cpu'
    store_misclassified = True


@ex.automain
def main(_run, model_path, load_ckpt, batch_size, device, store_misclassified):
    if IS_MASTER:
        commands.print_config(_run)

    model_path = Path(model_path)
    eval_dir = get_new_subdir(
        model_path / 'eval', id_naming='time', consider_mpi=True
    )
    # perform evaluation on a sub-set (10%) of the dataset used for training
    config = load_json(model_path / 'config.json')
    database_json = config['database_json']
    dataset = config['dataset']

    model = pt.Model.from_config(config['model'])
    model = model.load_checkpoint(
        model_path / 'checkpoints' / load_ckpt, consider_mpi=True
    )
    model.to(device)
    # Turn on evaluation mode for, e.g., BatchNorm and Dropout modules
    model.eval()

    _, _, test_set = get_datasets(
        model_path, database_json, dataset, batch_size,
        return_indexable=device == 'cpu'
    )
    with torch.no_grad():
        summary = dict(
            misclassified_examples=dict(),
            correct_classified_examples=dict(),
            hits=list()
        )
        for batch in split_managed(
            test_set, is_indexable=device == 'cpu',
            progress_bar=True, allow_single_worker=True
        ):
            output = model(pt.data.example_to_device(batch, device))
            prediction = torch.argmax(output, dim=-1).cpu().numpy()
            confidence = torch.softmax(output, dim=-1).max(dim=-1).values.cpu()\
                .numpy()
            label = np.array(batch['speaker_id'])
            hits = (label == prediction).astype('bool')
            summary['hits'].extend(hits.tolist())
            summary['misclassified_examples'].update({
                k: {
                    'true_label': v1,
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
            correct_classified = summary['correct_classified_examples']
            summary['correct_classified_examples'].update({
                k: correct_classified[k] + [v]
                if k in correct_classified.keys() else [v]
                for k, v in zip(
                    prediction[hits], np.array(batch['audio_path'])[hits]
                )
            })

    summary_list = COMM.gather(summary, root=MASTER)

    if IS_MASTER:
        print(f'\nlen(summary_list): {len(summary_list)}')
        if len(summary_list) > 1:
            summary = dict(
                misclassified_examples=dict(),
                correct_classified_examples=dict(),
                hits=list(),
            )
            for partial_summary in summary_list:
                summary['hits'].extend(partial_summary['hits'])
                summary['misclassified_examples'].update(
                    partial_summary['misclassified_examples']
                )
                for label, audio_path_list in \
                        partial_summary['correct_classified_examples'].items():
                    summary['correct_classified_examples'].update({
                        label: summary['correct_classified_examples'][label]
                        + audio_path_list if label in
                        summary['correct_classified_examples'].keys()
                        else audio_path_list
                    })
        hits = summary['hits']
        misclassified_examples = summary['misclassified_examples']
        correct_classified_examples = summary['correct_classified_examples']
        accuracy = np.array(hits).astype('float').mean()
        if store_misclassified:
            misclassified_dir = eval_dir / 'misclassified_examples'
            for example_id, v in misclassified_examples.items():
                label, prediction_label, audio_path, _ = v.values()
                try:
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
                except KeyError:
                    warnings.warn(
                        'There were no correctly predicted inputs from speaker '
                        f'with speaker label {prediction_label}'
                    )
        outputs = dict(
            accuracy=f'{accuracy:.2%} ({np.sum(hits)}/{len(hits)})',
            misclassifications=misclassified_examples,
        )
        print(f'Speaker classification accuracy on test set: {accuracy:.2%}')
        print(f'Wrote results to {eval_dir / "results.json"}')
        dump_json(outputs, eval_dir / 'results.json')
