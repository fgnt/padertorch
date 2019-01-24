"""
Example calls:
mpiexec -np 8 python -m padertorch.contrib.ldrude.evaluate_pit with model_path=/net/vol/ldrude/projects/2017/project_dc_storage/pth_models/pit/93

TODO: Add input mir_sdr result to be able to calculate gains.
TODO: Add pesq, stoi, invasive_sxr.
TODO: mpi: For mpi I need to know the experiment dir before.
"""
import os
import warnings
from collections import defaultdict
from pathlib import Path
import itertools

import einops
import matplotlib as mpl
import sacred.commands
from sacred import Experiment
from tqdm import tqdm

import paderbox as pb
import padertorch as pt
from paderbox.database.merl_mixtures import MerlMixtures
from paderbox.transform import istft
from paderbox.utils.mpi import COMM
from paderbox.utils.mpi import RANK
from paderbox.utils.mpi import SIZE
from paderbox.utils.mpi import IS_MASTER
from paderbox.utils.mpi import MASTER
from padertorch.contrib.ldrude.data import prepare_iterable
from padertorch.contrib.ldrude.utils import (
    decorator_append_file_storage_observer_with_lazy_basedir,
    get_new_folder
)


# Unfortunately need to disable this since conda scipy needs update
warnings.simplefilter(action='ignore', category=FutureWarning)


mpl.use("Agg")
nickname = Path(__file__).stem
ex = Experiment(nickname)
path_template = Path(os.environ["STORAGE"]) / "pth_evaluate" / nickname


@ex.config
def config():
    debug = False
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    checkpoint_name = 'ckpt_best_loss.pth'
    experiment_dir = get_new_folder(path_template, mkdir=False)
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
    model = pt.Module.from_storage_dir(model_path, checkpoint_name)

    # TODO: Can this run info be stored more elegantly?
    checkpoint_path = model_path / 'checkpoints' / checkpoint_name
    _run.info['checkpoint_path'] = str(checkpoint_path.expanduser().resolve())

    return model


@ex.automain
def main(_run, batch_size, datasets, debug, experiment_dir):
    if IS_MASTER:
        sacred.commands.print_config(_run)

    # Not necessary yet, but needed once we export more files.
    experiment_dir = COMM.bcast(experiment_dir, root=MASTER)
    print(f'RANK={RANK}, experiment_dir={experiment_dir}')

    model = get_model()
    db = MerlMixtures()

    summary = defaultdict(dict)
    for dataset in datasets:
        iterable = prepare_iterable(
            db, dataset, batch_size,
            return_keys=None,
            prefetch=False,
            iterator_slice=slice(RANK, 20 if debug else None, SIZE)
        )
        iterable = tqdm(iterable, total=len(iterable), disable=not IS_MASTER)
        for batch in iterable:
            entry = dict()
            model_output = model(pt.data.batch_to_device(batch))

            example_id = batch['example_id'][0]
            s = batch['s'][0]
            Y = batch['Y'][0]
            mask = model_output[0].detach().numpy()

            Z = mask * Y[:, None, :]
            z = istft(
                einops.rearrange(Z, "t k f -> k t f"),
                size=512, shift=128
            )

            s = s[:, :z.shape[1]]
            z = z[:, :s.shape[1]]
            entry['mir_eval'] \
                = pb.evaluation.mir_eval_sources(s, z, return_dict=True)

            summary[dataset][example_id] = entry

    summary_list = COMM.gather(summary, root=MASTER)

    if IS_MASTER:
        print(f'len(summary_list): {len(summary_list)}')
        summary = list(itertools.chain.from_iterable(summary_list))
        print(f'len(summary): {len(summary)}')
        result_json_path = experiment_dir / 'result.json'
        print(f"Exporting result: {result_json_path}")
        pb.io.dump_json(summary, result_json_path)
