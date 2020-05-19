"""
Example call on NT infrastructure:

export STORAGE=<your/desired/storage/root>
export WSJ0_2MIX=<path/to/wsj0_2mix/json>
mkdir -p $STORAGE/pth_evaluate/evaluate
mpiexec -np 8 python -m padertorch.contrib.examples.pit.evaluate with model_path=<model_path>

For usage with a single cpu:
export STORAGE=<your/desired/storage/root>
export WSJ0_2MIX=<path/to/wsj0_2mix/json>
mkdir -p $STORAGE/pth_evaluate/evaluate
python -m padertorch.contrib.examples.pit.evaluate with model_path=<model_path> debug=True

Example call on PC2 infrastructure (only neccessary in Paderborn University infrastructure):

export STORAGE=<your/desired/storage/root>
mkdir -p $STORAGE/pth_evaluate/pit
python -m padertorch.contrib.examples.pit.evaluate init with model_path=<model_path>


TODO: Add input mir_sdr result to be able to calculate gains.
TODO: Add pesq, stoi, invasive_sxr.
TODO: mpi: For mpi I need to know the experiment dir before.
"""
import os
from collections import defaultdict
from pathlib import Path

import einops
import sacred.commands
from sacred import Experiment
import torch

import paderbox as pb
import padertorch as pt
import pb_bss
from paderbox.transform import istft
from lazy_dataset.database import JsonDatabase
from dlp_mpi import COMM, IS_MASTER, MASTER, split_managed
from padertorch.contrib.examples.pit.data import prepare_iterable

from padertorch.contrib.ldrude.utils import (
    decorator_append_file_storage_observer_with_lazy_basedir,
    get_new_folder
)
from . import MAKEFILE_TEMPLATE_EVAL as MAKEFILE_TEMPLATE

nickname = "pit"
ex = Experiment(nickname)
path_template = Path(os.environ["STORAGE"]) / "pth_evaluate" / nickname


@ex.config
def config():
    debug = False
    database_json = ''
    if "WSJ0_2MIX" in os.environ:
        database_json = os.environ.get("WSJ0_2MIX")
    assert len(database_json) > 0, 'Set path to database Json on the command line or set environment variable WSJ0_2MIX'
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    checkpoint_name = 'ckpt_best_loss.pth'
    experiment_dir = str(get_new_folder(
        path_template, mkdir=False, consider_mpi=True,
    ))
    batch_size = 1
    datasets = ["mix_2_spk_min_cv", "mix_2_spk_min_tt"]
    locals()  # Fix highlighting


@decorator_append_file_storage_observer_with_lazy_basedir(
    ex,
    consider_mpi=True
)
def basedir(_config):
    return _config['experiment_dir']


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
        nickname=nickname
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

    if IS_MASTER:
        sacred.commands.print_config(_run)

    # TODO: Substantially faster to load the model once and distribute via MPI
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

            for batch in split_managed(iterable, is_indexable=False,
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
                entry['mir_eval'] \
                    = pb_bss.evaluation.mir_eval_sources(s, z, return_dict=True)

                summary[dataset][example_id] = entry

    summary_list = COMM.gather(summary, root=MASTER)

    if IS_MASTER:
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

