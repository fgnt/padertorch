# TODO: Add input mir_sdr result to be able to calculate gains.
# TODO: Add pesq, stoi, invasive_sxr.
# TODO: mpi: For mpi I need to know the experiment dir before.

from sacred import Experiment
import sacred.commands
from paderbox.database.merl_mixtures import MerlMixtures
import os
from pathlib import Path
from collections import defaultdict

import padertorch as pt
import sys

from padertorch.contrib.ldrude.utils import get_new_folder
import paderbox as pb
from tqdm import tqdm
from paderbox.transform import istft
import einops
from padertorch.contrib.ldrude.data import prepare_iterable

from paderbox.utils.mpi import COMM
from paderbox.utils.mpi import IS_MASTER
from paderbox.utils.mpi import MASTER

# Unfortunately need to disable this since conda scipy needs update
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


nickname = "pit"
ex = Experiment(nickname)

path_template = Path(os.environ["STORAGE"]) / "pth_evaluate" / nickname


if __name__ == "__main__":
    if "print_config" in sys.argv:
        storage_dir = Path(path_template / "tmp")
    else:
        # generate_path(nickname, family='models')
        storage_dir = get_new_folder(path_template)

        ex.observers.append(
            sacred.observers.FileStorageObserver.create(str(storage_dir))
        )


@ex.config
def config():
    debug = False
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    checkpoint_name = 'ckpt_best_loss.pth'

    batch_size = 1
    datasets = ["mix_2_spk_min_cv", "mix_2_spk_min_tt"]


@ex.automain
def main(
        _config, _run, model_path, batch_size, datasets, checkpoint_name,
        debug
):
    sacred.commands.print_config(_run)

    model_path = Path(model_path)
    model = pt.Module.from_storage_dir(model_path, checkpoint_name)

    # TODO: Can this run info be stored more elegantly?
    checkpoint_path = model_path / 'checkpoints' / checkpoint_name
    _run.info['checkpoint_path'] = checkpoint_path.expanduser().resolve()

    db = MerlMixtures()

    summary = defaultdict(dict)
    for dataset in datasets:
        iterable = prepare_iterable(
            db, dataset, batch_size,
            return_keys=None,
            prefetch=False,
            iterator_slice=slice(0, 100, 1) if debug else None
        )

        # TODO Do not hardcode the number here.
        for batch in tqdm(iterable, total=len(iterable)):
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

    result_json_path = storage_dir / 'result.json'
    print(f"Exporting result: {result_json_path}")
    pb.io.dump_json(summary, result_json_path)
