"""
Example call on NT infrastructure:

mpiexec -np 8 python -m padertorch.contrib.examples.source_separation.pit.evaluate with model_path=<model_path> database_json=<path/to/database.json>


For usage with a single cpu:

python -m padertorch.contrib.examples.source_separation.pit.evaluate with model_path=<model_path> database_json=<path/to/database.json> debug=True


Example call on PC2 infrastructure (only neccessary in Paderborn University infrastructure):

python -m padertorch.contrib.examples.source_separation.pit.evaluate init with model_path=<model_path>


TODO: Add input mir_sdr result to be able to calculate gains.
TODO: Add pesq, stoi, invasive_sxr.
"""
import os
from collections import defaultdict
from pathlib import Path

import einops
import sacred.commands
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import InvalidConfigError, MissingConfigError
import torch

import dlp_mpi
import paderbox as pb
import padertorch as pt
import pb_bss
from paderbox.transform import istft
from lazy_dataset.database import JsonDatabase
from padertorch.contrib.examples.source_separation.pit.data import prepare_iterable
from padertorch.contrib.examples.source_separation.pit.templates import MAKEFILE_TEMPLATE_EVAL as MAKEFILE_TEMPLATE

experiment_name = "pit"
ex = Experiment(experiment_name)


@ex.config
def config():
    debug = False

    database_json = None
    if "WSJ0_2MIX" in os.environ:
        database_json = os.environ.get("WSJ0_2MIX")
    assert len(database_json) > 0, 'Set path to database Json on the command line or set environment variable WSJ0_2MIX'
    model_path = ''
    checkpoint_name = 'ckpt_best_loss.pth'
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_subdir(
            Path(model_path) / 'evaluation', consider_mpi=True)
    batch_size = 1
    datasets = ["mix_2_spk_min_cv", "mix_2_spk_min_tt"]
    locals()  # Fix highlighting

    ex.observers.append(FileStorageObserver(
        Path(Path(experiment_dir) / 'sacred')
    ))
    if database_json is None:
        raise MissingConfigError(
            'You have to set the path to the database JSON!', 'database_json')
    if not Path(database_json).exists():
        raise InvalidConfigError('The database JSON does not exist!',
                                 'database_json')

@ex.capture
def get_model(_run, model_path, checkpoint_name):
    model_path = Path(model_path)
    model = pt.Module.from_storage_dir(
        model_path,
        checkpoint_name=checkpoint_name,
        consider_mpi=True  # Loads the weights only on master
    )

    checkpoint_path = model_path / 'checkpoints' / checkpoint_name
    _run.info['checkpoint_path'] = str(checkpoint_path.expanduser().resolve())

    return model


@ex.command
def init(_config, _run):
    """Creates a storage dir, writes Makefile. Does not start any evaluation."""
    experiment_dir = Path(_config['experiment_dir'])

    config_path = Path(experiment_dir) / "config.json"
    pb.io.dump_json(_config, config_path)

    makefile_path = Path(experiment_dir) / "Makefile"
    makefile_path.write_text(MAKEFILE_TEMPLATE.format(
        main_python_path=pt.configurable.resolve_main_python_path(),
        experiment_dir=experiment_dir,
        experiment_name=experiment_name
    ))

    sacred.commands.print_config(_run)
    print()
    print('Initialized storage dir. Now run these commands:')
    print(f"cd {experiment_dir}")
    print(f"make evaluate")
    print()
    print('or')
    print()
    print('make ccsalloc')


@ex.main
def main(_run, batch_size, datasets, debug, experiment_dir, database_json):
    experiment_dir = Path(experiment_dir)

    if dlp_mpi.IS_MASTER:
        sacred.commands.print_config(_run)

    model = get_model()
    db = JsonDatabase(json_path=database_json)

    model.eval()
    with torch.no_grad():
        summary = defaultdict(dict)
        for dataset in datasets:
            iterable = prepare_iterable(
                db, dataset, batch_size,
                return_keys=None,
                prefetch=False,
            )

            for batch in dlp_mpi.split_managed(iterable, is_indexable=False,
                                           progress_bar=True,
                                           allow_single_worker=debug
                                           ):
                entry = dict()
                model_output = model(pt.data.example_to_device(batch))

                example_id = batch['example_id'][0]
                s = batch['s'][0]
                Y = batch['Y'][0]
                mask = model_output[0].numpy()

                Z = mask * Y[:, None, :]
                z = istft(
                    einops.rearrange(Z, "t k f -> k t f"),
                    size=512, shift=128
                )

                s = s[:, :z.shape[1]]
                z = z[:, :s.shape[1]]
                entry['metrics'] \
                    = pb_bss.evaluation.OutputMetrics(speech_prediction=z, speech_source=s).as_dict()

        summary[dataset][example_id] = entry

    summary_list = dlp_mpi.gather(summary, root=dlp_mpi.MASTER)

    if dlp_mpi.IS_MASTER:
        print(f'len(summary_list): {len(summary_list)}')
        for partial_summary in summary_list:
            for dataset, values in partial_summary.items():
                summary[dataset].update(values)

        for dataset, values in summary.items():
            print(f'{dataset}: {len(values)}')

        result_json_path = experiment_dir / 'result.json'
        print(f"Exporting result: {result_json_path}")
        pb.io.dump_json(summary, result_json_path)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()

