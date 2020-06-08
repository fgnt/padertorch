import os
import warnings
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib as mpl
import numpy as np
import paderbox as pb
from lazy_dataset.database import JsonDatabase

import padertorch as pt
import sacred.commands
from sacred.utils import InvalidConfigError, MissingConfigError
import torch
import dlp_mpi as mpi
from padertorch.contrib.ldrude.utils import (
    get_new_folder
)
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
import pb_bss

from .train import prepare_iterable

# Unfortunately need to disable this since conda scipy needs update
warnings.simplefilter(action='ignore', category=FutureWarning)

mpl.use("Agg")
nickname = "or-pit"
ex = Experiment(nickname)


def get_storage_dir():
    # Sacred should not add path_template to the config
    # -> move this few lines to a function
    path_template = Path(os.environ["STORAGE"]) / "pth_evaluate" / nickname
    return str(get_new_folder(
        path_template, mkdir=False, consider_mpi=True,
    ))


@ex.config
def config():
    debug = False
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    checkpoint_name = 'ckpt_best_loss.pth'
    experiment_dir = get_storage_dir()
    datasets = ["mix_2_spk_min_cv", "mix_2_spk_min_tt"]
    export_audio = False
    sample_rate = 8000
    target = 'speech_source'
    database_json = None
    oracle_num_spk = False  # If true, the model is forced to perform the correct (oracle) number of iterations
    max_iterations = 4  # The number of iterations is limited to this number

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

    "SHELL := /bin/bash"
    ""
    "evaluate:"
    "\tpython -m {main_python_path} with config.json"
    ""
    "ccsalloc:"
    "\tccsalloc \\"
    "\t\t--notifyuser=awe \\"
    "\t\t--res=rset=200:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\"
    "\t\t--time=1h \\"
    "\t\t--join \\"
    "\t\t--stdout=stdout \\"
    "\t\t--tracefile=trace_%reqid.trace \\"
    "\t\t-N evaluate_{nickname} \\"
    "\t\tompi \\"
    "\t\t-x STORAGE \\"
    "\t\t-x NT_MERL_MIXTURES_DIR \\"
    "\t\t-x NT_DATABASE_JSONS_DIR \\"
    "\t\t-x KALDI_ROOT \\"
    "\t\t-x LD_PRELOAD \\"
    "\t\t-x CONDA_EXE \\"
    "\t\t-x CONDA_PREFIX \\"
    "\t\t-x CONDA_PYTHON_EXE \\"
    "\t\t-x CONDA_DEFAULT_ENV \\"
    "\t\t-x PATH \\"
    "\t\t-- \\"
    "\t\tpython -m {main_python_path} with config.json"

    experiment_dir = Path(_config['experiment_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"

    model_path = Path(_config['model_path'])

    # Create symlinks between model and evaluation
    eval_symlink = model_path / 'eval' / experiment_dir.name
    if not eval_symlink.is_symlink():
        eval_symlink.parent.mkdir(exist_ok=True)
        eval_symlink.symlink_to(experiment_dir, target_is_directory=True)

    model_symlink = experiment_dir / 'model'
    if not model_symlink.is_symlink():
        model_symlink.symlink_to(model_path, target_is_directory=True)

    if not makefile_path.exists():
        config_path = experiment_dir / "config.json"

        pb.io.dump_json(_config, config_path)
        main_python_path = pt.configurable.resolve_main_python_path()
        makefile_path.write_text(
            f"SHELL := /bin/bash\n"
            f"\n"
            f"evaluate:\n"
            f"\tpython -m {main_python_path} with config.json\n"
            f"\n"
            f"ccsalloc:\n"
            f"\tccsalloc \\\n"
            f"\t\t--notifyuser=awe \\\n"
            f"\t\t--res=rset=200:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\\n"
            f"\t\t--time=1h \\\n"
            f"\t\t--join \\\n"
            f"\t\t--stdout=stdout \\\n"
            f"\t\t--tracefile=trace_%reqid.trace \\\n"
            f"\t\t-N evaluate_{nickname} \\\n"
            f"\t\tompi \\\n"
            f"\t\t-x STORAGE \\\n"
            f"\t\t-x NT_MERL_MIXTURES_DIR \\\n"
            f"\t\t-x NT_DATABASE_JSONS_DIR \\\n"
            f"\t\t-x KALDI_ROOT \\\n"
            f"\t\t-x LD_PRELOAD \\\n"
            f"\t\t-x CONDA_EXE \\\n"
            f"\t\t-x CONDA_PREFIX \\\n"
            f"\t\t-x CONDA_PYTHON_EXE \\\n"
            f"\t\t-x CONDA_DEFAULT_ENV \\\n"
            f"\t\t-x PATH \\\n"
            f"\t\t-- \\\n"
            f"\t\tpython -m {main_python_path} with config.json\n"
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
def main(_run, datasets, debug, experiment_dir, export_audio,
         sample_rate, _log, database_json, oracle_num_spk, max_iterations):
    experiment_dir = Path(experiment_dir)

    if mpi.IS_MASTER:
        sacred.commands.print_config(_run)
        dump_config_and_makefile()

    model = get_model()
    db = JsonDatabase(database_json)

    model.eval()
    with torch.no_grad():
        summary = defaultdict(dict)
        for dataset in datasets:
            iterable = prepare_iterable(
                db, dataset, 1,
                chunk_size=-1,
                prefetch=False,
                shuffle=False,
                iterator_slice=slice(mpi.RANK, 20 if debug else None,
                                     mpi.SIZE),
            )

            if export_audio:
                (experiment_dir / 'audio' / dataset).mkdir(
                    parents=True, exist_ok=True)

            for batch in tqdm(
                    iterable, total=len(iterable), disable=not mpi.IS_MASTER,
                    desc=dataset,
            ):
                example_id = batch['example_id'][0]
                summary[dataset][example_id] = entry = dict()
                oracle_speaker_count = \
                    entry['oracle_speaker_count'] = batch['s'][0].shape[0]

                try:
                    model_output = model.decode(
                        pt.data.example_to_device(batch),
                        max_iterations=max_iterations,
                        oracle_num_speakers=oracle_speaker_count if
                        oracle_num_spk else None
                    )

                    # Bring to numpy float64 for evaluation metrics computation
                    s = batch['s'][0].astype(np.float64)
                    z = model_output['out'][0].cpu().numpy().astype(np.float64)

                    estimated_speaker_count = \
                        entry['estimated_speaker_count'] = z.shape[0]
                    entry['source_counting_accuracy'] = \
                        estimated_speaker_count == oracle_speaker_count

                    if oracle_speaker_count == estimated_speaker_count:
                        # These evaluations don't work if the number of
                        # speakers in s and z don't match
                        entry['mir_eval'] = pb_bss.evaluation.mir_eval_sources(
                            s, z, return_dict=True)

                        # Get the correct order for si_sdr and saving
                        z = z[entry['mir_eval']['selection']]

                        entry['si_sdr'] = pb_bss.evaluation.si_sdr(s, z)
                    else:
                        warnings.warn(
                            'The number of speakers is estimated incorrectly '
                            'for some examples! The calculated SDR values '
                            'might not be representative!'
                        )

                    if export_audio:
                        entry['audio_path'] = batch['audio_path']
                        entry['audio_path'].setdefault('estimated', [])

                        for k, audio in enumerate(z):
                            audio_path = (
                                    experiment_dir / 'audio' / dataset /
                                    f'{example_id}_{k}.wav')
                            pb.io.dump_audio(
                                audio, audio_path, sample_rate=sample_rate)
                            entry['audio_path']['estimated'].append(audio_path)
                except:
                    _log.error(
                        f'Exception was raised in example with ID '
                        f'"{example_id}"'
                    )
                    raise

    summary_list = mpi.gather(summary, root=mpi.MASTER)

    if mpi.IS_MASTER:
        # Combine all summaries to one
        for partial_summary in summary_list:
            for dataset, values in partial_summary.items():
                summary[dataset].update(values)

        for dataset, values in summary.items():
            _log.info(f'{dataset}: {len(values)}')

        # Write summary to JSON
        result_json_path = experiment_dir / 'result.json'
        _log.info(f"Exporting result: {result_json_path}")
        pb.io.dump_json(summary, result_json_path)

        # Compute means for some metrics
        mean_keys = ['mir_eval.sdr', 'mir_eval.sar', 'mir_eval.sir', 'si_sdr',
                     'source_counting_accuracy']
        means = {}
        for dataset, dataset_results in summary.items():
            means[dataset] = {}
            flattened = {
                k: pb.utils.nested.flatten(v) for k, v in
                dataset_results.items()
            }
            for mean_key in mean_keys:
                try:
                    means[dataset][mean_key] = np.mean(np.array([
                        v[mean_key] for v in flattened.values()
                    ]))
                except KeyError:
                    warnings.warn(f'Couldn\'t compute mean for {mean_key}.')
            means[dataset] = pb.utils.nested.deflatten(means[dataset])

        mean_json_path = experiment_dir / 'means.json'
        _log.info(f'Exporting means: {mean_json_path}')
        pb.io.dump_json(means, mean_json_path)

        _log.info('Resulting means:')

        pprint(means)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
