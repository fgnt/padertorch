"""
Example call on NT infrastructure:

mpiexec -np 8 python -m padertorch.contrib.examples.source_separation.tasnet.evaluate with model_path=<model_path> database_json=<database_json>
"""
import itertools

import os
import warnings
from collections import defaultdict
import operator

import dlp_mpi
import numpy as np
import paderbox as pb
import pb_bss
import sacred.commands
import torch
from lazy_dataset.database import JsonDatabase
from pathlib import Path
from pprint import pprint
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import InvalidConfigError, MissingConfigError
from tqdm import tqdm

import padertorch as pt
from padertorch.contrib.neumann.evaluation import compute_means
from .train import prepare_iterable

# Unfortunately need to disable this since conda scipy needs update
warnings.simplefilter(action='ignore', category=FutureWarning)

experiment_name = 'tasnet'
ex = Experiment(experiment_name)

JSON_BASE = os.environ.get('NT_DATABASE_JSONS_DIR', None)

@ex.config
def config():
    debug = False
    dump_audio = False

    # Model config
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    checkpoint_name = 'ckpt_best_loss.pth'
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_subdir(
            Path(model_path) / 'evaluation', consider_mpi=True)

    # Database config
    database_json = None
    if database_json is None and JSON_BASE:
        database_json = Path(JSON_BASE) / 'wsj0_2mix_8k.json'

    sample_rate = 8000
    datasets = ['mix_2_spk_min_cv', 'mix_2_spk_min_tt']
    target = 'speech_source'

    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')

    locals()  # Fix highlighting

    ex.observers.append(FileStorageObserver(
        Path(Path(experiment_dir) / 'sacred')
    ))


@ex.capture
def get_model(_run, model_path, checkpoint_name):
    model_path = Path(model_path)
    model = pt.Module.from_storage_dir(
        model_path,
        checkpoint_name=checkpoint_name,
        consider_mpi=True  # Loads the weights only on master
    )

    # TODO: Can this run info be stored more elegantly?
    checkpoint_path = model_path / 'checkpoints' / checkpoint_name
    _run.info['checkpoint_path'] = str(checkpoint_path.expanduser().resolve())

    return model


@ex.capture
def dump_config_and_makefile(_config):
    """
    Dumps the configuration into the experiment dir and creates a Makefile
    next to it. If a Makefile already exists, it does not do anything.
    """

    experiment_dir = Path(_config['experiment_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"

    if not makefile_path.exists():
        config_path = experiment_dir / "config.json"
        pb.io.dump_json(_config, config_path)

        from padertorch.contrib.examples.source_separation.tasnet.templates import \
            MAKEFILE_TEMPLATE_EVAL
        makefile_path.write_text(
            MAKEFILE_TEMPLATE_EVAL.format(
                main_python_path=pt.configurable.resolve_main_python_path(),
                experiment_name=experiment_name,
            )
        )


@ex.command
def init(_config, _run):
    """Create a storage dir, write Makefile. Do not start any evaluation."""
    sacred.commands.print_config(_run)
    dump_config_and_makefile()

    print()
    print('Initialized storage dir. Now run these commands:')
    print(f"cd {_config['experiment_dir']}")
    print(f"make evaluate")
    print()
    print('or')
    print()
    print('make ccsalloc')


@ex.main
def main(_run, datasets, debug, experiment_dir, dump_audio,
         sample_rate, _log, database_json):
    experiment_dir = Path(experiment_dir)

    if dlp_mpi.IS_MASTER:
        sacred.commands.print_config(_run)
        dump_config_and_makefile()

    model = get_model()
    db = JsonDatabase(database_json)

    model.eval()
    results = defaultdict(dict)
    with torch.no_grad():
        for dataset in datasets:
            iterable = prepare_iterable(
                db, dataset, 1,
                chunk_size=-1,
                prefetch=False,
                iterator_slice=slice(dlp_mpi.RANK, 20 if debug else None,
                                     dlp_mpi.SIZE),
            )

            if dump_audio:
                (experiment_dir / 'audio' / dataset).mkdir(
                    parents=True, exist_ok=True)

            for batch in tqdm(iterable, total=len(iterable),
                              disable=not dlp_mpi.IS_MASTER):
                example_id = batch['example_id'][0]
                results[dataset][example_id] = entry = dict()

                try:
                    model_output = model(model.example_to_device(batch))

                    # Bring to numpy float64 for evaluation metrics computation
                    observation = batch['y'][0].astype(np.float64)[None, ]
                    speech_prediction = (
                        model_output['out'][0].cpu().numpy().astype(np.float64)
                    )
                    speech_source = batch['s'][0].astype(np.float64)

                    input_metrics = pb_bss.evaluation.InputMetrics(
                        observation=observation,
                        speech_source=speech_source,
                        sample_rate=sample_rate,
                        enable_si_sdr=True,
                    )

                    output_metrics = pb_bss.evaluation.OutputMetrics(
                        speech_prediction=speech_prediction,
                        speech_source=speech_source,
                        sample_rate=sample_rate,
                        enable_si_sdr=True,
                    )

                    # Select the metrics to compute
                    entry['input'] = dict(
                        mir_eval=input_metrics.mir_eval,
                        si_sdr=input_metrics.si_sdr,
                        # TODO: stoi fails with short speech segments (https://github.com/mpariente/pystoi/issues/21)
                        # stoi=input_metrics.stoi,
                        # TODO: pesq creates "Processing error" messages
                        # pesq=input_metrics.pesq,
                    )

                    # Remove selection from mir_eval dict to enable
                    # recursive calculation of improvement
                    entry['output'] = dict(
                        mir_eval={
                            k: v for k, v in output_metrics.mir_eval.items()
                            if k != 'selection'
                        },
                        si_sdr=output_metrics.si_sdr,
                        # stoi=output_metrics.stoi,
                        # pesq=output_metrics.pesq,
                    )

                    entry['improvement'] = pb.utils.nested.nested_op(
                        operator.sub, entry['output'], entry['input'],
                    )
                    entry['selection'] = output_metrics.mir_eval['selection']

                    if dump_audio:
                        entry['audio_path'] = batch['audio_path']
                        entry['audio_path']['estimated'] = []
                        for k, audio in enumerate(speech_prediction):
                            audio_path = (
                                experiment_dir / 'audio' / dataset / 
                                f'{example_id}_{k}.wav'
                            )
                            pb.io.dump_audio(
                                audio, audio_path, sample_rate=sample_rate)
                            entry['audio_path']['estimated'].append(audio_path)
                except:
                    _log.error(
                        f'Exception was raised in example with ID '
                        f'"{example_id}"'
                    )
                    raise

    results = dlp_mpi.gather(results, root=dlp_mpi.MASTER)

    if dlp_mpi.IS_MASTER:
        # Combine all results to one. This function raises an exception if it
        # finds duplicate keys
        results = pb.utils.nested.nested_merge(*results)

        for dataset, values in results.items():
            _log.info(f'{dataset}: {len(values)}')

        # Write results to JSON
        result_json_path = experiment_dir / 'result.json'
        _log.info(f"Exporting result: {result_json_path}")
        pb.io.dump_json(results, result_json_path)

        # Compute means for some metrics
        means = compute_means(results)
        mean_json_path = experiment_dir / 'means.json'
        _log.info(f'Exporting means: {mean_json_path}')
        pb.io.dump_json(means, mean_json_path)

        _log.info('Resulting means:')

        pprint(means)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
