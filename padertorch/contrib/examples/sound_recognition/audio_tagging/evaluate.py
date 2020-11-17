"""
Example call:

python -m padertorch.contrib.examples.audio_synthesis.wavenet.evaluate with exp_dir=/path/to/exp_dir
"""
from pathlib import Path
import numpy as np
import torch
from padertorch import Model
from padertorch.contrib.examples.sound_recognition.audio_tagging.data import get_datasets
from padertorch.contrib.je.modules.reduce import Mean
from sacred import Experiment, commands
from sklearn import metrics
from tqdm import tqdm

from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json
from pb_sed.evaluation import instance_based
from pprint import pprint


ex = Experiment('audio-tagging-eval')


@ex.config
def config():
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the model path on the command line.'
    storage_dir = str(get_new_subdir(
        Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
    ))
    database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    num_workers = 8
    batch_size = 32
    max_padding_rate = .05
    device = 0
    ckpt_name = 'ckpt_best_map.pth'


@ex.automain
def main(
        _run, exp_dir, storage_dir, database_json, ckpt_name,
        num_workers, batch_size, max_padding_rate,
        device
):
    commands.print_config(_run)

    exp_dir = Path(exp_dir)
    storage_dir = Path(storage_dir)

    config = load_json(exp_dir / 'config.json')

    model = Model.from_storage_dir(
        exp_dir, consider_mpi=True, checkpoint_name=ckpt_name
    )
    model.to(device)
    model.eval()

    _, validation_data, test_data = get_datasets(
        database_json=database_json, min_signal_length=1.5,
        audio_reader=config['audio_reader'], stft=config['stft'],
        num_workers=num_workers,
        batch_size=batch_size, max_padding_rate=max_padding_rate,
        storage_dir=exp_dir,
    )

    outputs = []
    with torch.no_grad():
        for example in tqdm(validation_data):
            example = model.example_to_device(example, device)
            (y, seq_len), _ = model(example)
            y = Mean(axis=-1)(y, seq_len)
            outputs.append((
                y.cpu().detach().numpy(),
                example['events'].cpu().detach().numpy(),
            ))

    scores, targets = list(zip(*outputs))
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    thresholds, f1 = instance_based.get_optimal_thresholds(
        targets, scores, metric='f1'
    )
    decisions = scores > thresholds
    f1, p, r = instance_based.fscore(targets, decisions, event_wise=True)
    ap = metrics.average_precision_score(targets, scores, None)
    auc = metrics.roc_auc_score(targets, scores, None)
    pos_class_indices, precision_at_hits = instance_based.positive_class_precisions(
        targets, scores
    )
    lwlrap, per_class_lwlrap, weight_per_class = instance_based.lwlrap_from_precisions(
        precision_at_hits, pos_class_indices, num_classes=targets.shape[1]
    )
    overall_results = {
        'validation': {
            'mF1': np.mean(f1),
            'mP': np.mean(p),
            'mR': np.mean(r),
            'mAP': np.mean(ap),
            'mAUC': np.mean(auc),
            'lwlrap': lwlrap,
        }
    }
    event_validation_results = {}
    labels = load_json(exp_dir / 'events.json')
    for i, label in enumerate(labels):
        event_validation_results[label] = {
            'F1': f1[i],
            'P': p[i],
            'R': r[i],
            'AP': ap[i],
            'AUC': auc[i],
            'lwlrap': per_class_lwlrap[i],
        }

    outputs = []
    with torch.no_grad():
        for example in tqdm(test_data):
            example = model.example_to_device(example, device)
            (y, seq_len), _ = model(example)
            y = Mean(axis=-1)(y, seq_len)
            outputs.append((
                example['example_id'],
                y.cpu().detach().numpy(),
                example['events'].cpu().detach().numpy(),
            ))

    example_ids, scores, targets = list(zip(*outputs))
    example_ids = np.concatenate(example_ids).tolist()
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    decisions = scores > thresholds
    f1, p, r = instance_based.fscore(targets, decisions, event_wise=True)
    ap = metrics.average_precision_score(targets, scores, None)
    auc = metrics.roc_auc_score(targets, scores, None)
    pos_class_indices, precision_at_hits = instance_based.positive_class_precisions(
        targets, scores
    )
    lwlrap, per_class_lwlrap, weight_per_class = instance_based.lwlrap_from_precisions(
        precision_at_hits, pos_class_indices, num_classes=targets.shape[1]
    )
    overall_results['test'] = {
        'mF1': np.mean(f1),
        'mP': np.mean(p),
        'mR': np.mean(r),
        'mAP': np.mean(ap),
        'mAUC': np.mean(auc),
        'lwlrap': lwlrap,
    }
    dump_json(
        overall_results, storage_dir/'overall.json', indent=4, sort_keys=False
    )
    event_results = {}
    for i, label in sorted(enumerate(labels), key=lambda x: ap[x[0]], reverse=True):
        event_results[label] = {
            'validation': event_validation_results[label],
            'test': {
                'F1': f1[i],
                'P': p[i],
                'R': r[i],
                'AP': ap[i],
                'AUC': auc[i],
                'lwlrap': per_class_lwlrap[i],
            },
        }
    dump_json(
        event_results, storage_dir / 'event_wise.json', indent=4, sort_keys=False
    )
    fp = np.argwhere(decisions*(1-targets))
    dump_json(
        sorted([(example_ids[n], labels[i]) for n, i in fp]), storage_dir / 'fp.json', indent=4, sort_keys=False
    )
    fn = np.argwhere((1-decisions)*targets)
    dump_json(
        sorted([(example_ids[n], labels[i]) for n, i in fn]), storage_dir / 'fn.json', indent=4, sort_keys=False
    )
    pprint(overall_results)
