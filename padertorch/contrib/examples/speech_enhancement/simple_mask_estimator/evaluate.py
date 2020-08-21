"""
Evaluation script for the last mask estimator trained in $STORAGE_ROOT.
Saves results to $STORAGE_ROOT/speech_enhancement/simple_mask_estimator_{id}/evaluate_{eval_id}
mpiexec -np 8 python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.evaluate

"""
import os
import sys
from pathlib import Path

import dlp_mpi
import numpy as np
import paderbox as pb
import padertorch as pt
import pb_bss
import torch
from einops import rearrange
from padercontrib.database import JsonAudioDatabase
from padercontrib.database.chime import Chime3
from padercontrib.database.iterator import AudioReader
from pb_bss.extraction import get_bf_vector

from . import SimpleMaskEstimator


def change_example_structure(example):
    stft = pb.transform.stft
    audio_data = example['audio_data']
    net_input = dict()
    net_input['observation'] = audio_data['observation']
    net_input['observation_stft'] = stft(
        audio_data['observation']).astype(np.complex64)
    net_input['observation_abs'] = np.abs(
        net_input['observation_stft']).astype(np.float32)
    net_input['speech_source'] = audio_data['speech_source']
    net_input['example_id'] = example['example_id']
    return net_input


def get_test_dataset(database: JsonAudioDatabase):
    # AudioReader is a specialized function to read audio organized
    # in a json as described in pb.database.database
    audio_reader = AudioReader(audio_keys=[
        'observation', 'speech_source'
    ])
    val_iterator = database.get_dataset_test()
    return val_iterator.map(audio_reader) \
        .map(change_example_structure)


def evaluate(checkpoint_path):
    model = SimpleMaskEstimator(513)
    model_dir = checkpoint_path.parent / '..'
    eval_dir = pt.io.get_new_subdir(model_dir, prefix='evaluate',
                                    consider_mpi=True)

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
    database = Chime3()
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


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        STORAGE_ROOT = os.environ.get('STORAGE_ROOT')
        if STORAGE_ROOT is None:
            raise EnvironmentError(
                'You have to specify an STORAGE_ROOT '
                'environmental variable see getting_started'
            )
        elif not Path(STORAGE_ROOT).exists():
            raise FileNotFoundError(
                'You have to specify an existing STORAGE_ROOT '
                'environmental variable see getting_started.\n'
                f'Got: {STORAGE_ROOT}'
            )
        else:
            STORAGE_ROOT = Path(STORAGE_ROOT).expanduser().resolve()
        task_dir = STORAGE_ROOT / 'speech_enhancement'
        dirs = list(task_dir.glob('simple_mask_estimator_*'))
        latest_id = sorted(
            [int(path.name.split('_')[-1]) for path in dirs])[-1]
        model_dir = task_dir / f'simple_mask_estimator_{latest_id}'
        checkpoint_path = model_dir / 'checkpoints' / 'ckpt_best_loss.pth'
    elif len(args) == 2:
        checkpoint_path = Path(args[1]).expanduser().resolve()
    else:
        raise ValueError('Not more than one argument allowed, the one'
                         'argument describes the checkpoint path', args)
    evaluate(checkpoint_path)
