"""
Evaluation script for the last mask estimator trained in $STORAGE_ROOT.
Saves results to $STORAGE_ROOT/speech_enhancement/simple_mask_estimator_{id}/evaluate_{eval_id}
mpiexec -np 8 python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.evaluate with database_json=/path/to/json

If you want to evaluate a specific checkpoint, specify the path
as an additional argument to the call.
mpiexec -np 8 python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.evaluate with checkpoint_path=/path/to/checkpoint database_json=/path/to/json
"""
import os
from pathlib import Path

import dlp_mpi
import numpy as np
import paderbox as pb
import padertorch as pt
import pb_bss
import torch
from einops import rearrange
from lazy_dataset.database import JsonDatabase
from pb_bss.extraction import get_bf_vector
from sacred import Experiment, observers

from .model import SimpleMaskEstimator

ex = Experiment('Evaluate Simple Mask Estimator')


@ex.config
def config():
    checkpoint_path = get_checkpoint_path()
    eval_dir = pt.io.get_new_subdir(
        checkpoint_path.parent / '..', prefix='evaluate', consider_mpi=True)
    database_json = None
    assert database_json is not None, (
        'You have to specify a path to a json describing your database,'
        'use "with database_json=/Path/To/Json" as suffix to your call'
    )
    assert Path(database_json).exists(), database_json
    assert Path(checkpoint_path).exists(), checkpoint_path
    ex.observers.append(observers.FileStorageObserver(
        Path(eval_dir).expanduser().resolve() / 'sacred')
    )


def get_checkpoint_path():
    storage_root = os.environ.get('STORAGE_ROOT')
    if storage_root is None:
        raise EnvironmentError(
            'You have to specify an STORAGE_ROOT '
            'environmental variable see getting_started'
        )
    elif not Path(storage_root).exists():
        raise FileNotFoundError(
            'You have to specify an existing STORAGE_ROOT '
            'environmental variable see getting_started.\n'
            f'Got: {storage_root}'
        )
    else:
        storage_root = Path(storage_root).expanduser().resolve()
    task_dir = storage_root / 'speech_enhancement'
    dirs = list(task_dir.glob('simple_mask_estimator_*'))
    latest_id = sorted(
        [int(path.name.split('_')[-1]) for path in dirs])[-1]
    model_dir = task_dir / f'simple_mask_estimator_{latest_id}'
    return model_dir / 'checkpoints' / 'ckpt_best_loss.pth'


def prepare_data(example):
    stft = pb.transform.STFT(shift=256, size=1024)
    audio_data = dict()
    for key in ['observation', 'speech_source']:
        audio_data[key] = np.array([
            pb.io.load_audio(audio) for audio in example['audio_path'][key]])
    net_input = audio_data.copy()
    net_input['observation_abs'] = np.abs(
        stft(audio_data['observation'])).astype(np.float32)
    net_input['observation_stft'] = stft(audio_data['observation'])
    net_input['example_id'] = example['example_id']
    return net_input


def get_test_dataset(database: JsonDatabase):
    val_iterator = database.get_dataset('et05_simu')
    return val_iterator.map(prepare_data)


@ex.automain
def evaluate(checkpoint_path, eval_dir, database_json):
    model = SimpleMaskEstimator(513)

    model.load_checkpoint(
        checkpoint_path=checkpoint_path,
        in_checkpoint_path='model',
        consider_mpi=True
    )
    model.eval()
    if dlp_mpi.IS_MASTER:
        print(f'Start to evaluate the checkpoint {checkpoint_path.resolve()} '
              f'and will write the evaluation result to'
              f' {eval_dir / "result.json"}')
    database = JsonDatabase(database_json)
    test_dataset = get_test_dataset(database)
    with torch.no_grad():
        summary = dict(masked=dict(), beamformed=dict(), observed=dict())
        for batch in dlp_mpi.split_managed(
                test_dataset, is_indexable=True,
                progress_bar=True,
                allow_single_worker=True
        ):
            model_output = model(pt.data.example_to_device(batch))

            example_id = batch['example_id']
            s = batch['speech_source'][0][None]

            speech_mask = model_output['speech_mask_prediction'].numpy()
            Y = batch['observation_stft']
            Z_mask = speech_mask[0] * Y[0]
            z_mask = pb.transform.istft(Z_mask)[None]

            speech_mask = np.median(speech_mask, axis=0).T
            noise_mask = model_output['noise_mask_prediction'].numpy()
            noise_mask = np.median(noise_mask, axis=0).T
            Y = rearrange(Y, 'c t f -> f c t')
            target_psd = pb_bss.extraction.get_power_spectral_density_matrix(
                Y, speech_mask,
            )
            noise_psd = pb_bss.extraction.get_power_spectral_density_matrix(
                Y, noise_mask,
            )
            beamformer = pb_bss.extraction.get_bf_vector(
                'mvdr_souden',
                target_psd_matrix=target_psd,
                noise_psd_matrix=noise_psd

            )
            Z_bf = pb_bss.extraction.apply_beamforming_vector(beamformer, Y).T
            z_bf = pb.transform.istft(Z_bf)[None]

            y = batch['observation'][0][None]

            s = s[:, :z_bf.shape[1]]
            for key, signal in zip(summary.keys(), [z_mask, z_bf, y]):
                signal = signal[:, :s.shape[1]]
                entry = pb_bss.evaluation.OutputMetrics(
                    speech_prediction=signal, speech_source=s,
                    sample_rate=16000
                ).as_dict()
                entry.pop('mir_eval_selection')
                summary[key][example_id] = entry

    summary_list = dlp_mpi.COMM.gather(summary, root=dlp_mpi.MASTER)

    if dlp_mpi.IS_MASTER:
        print(f'\n len(summary_list): {len(summary_list)}')
        summary = dict(masked=dict(), beamformed=dict(), observed=dict())
        for partial_summary in summary_list:
            for signal_type, metric in partial_summary.items():
                summary[signal_type].update(metric)
        for signal_type, values in summary.items():
            print(signal_type)
            for metric in next(iter(values.values())).keys():
                mean = np.mean([value[metric] for key, value in values.items()
                                if '_mean' not in key])
                values[metric + '_mean'] = mean
                print(f'{metric}: {mean}')

        result_json_path = eval_dir / 'result.json'
        print(f"Exporting result: {result_json_path}")
        pb.io.dump_json(summary, result_json_path)
