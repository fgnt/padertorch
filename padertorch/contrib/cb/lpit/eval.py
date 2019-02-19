"""


mpiexec -np 8 python -m cbj.pytorch.eval_am with model_path=..

Other options:
# checkpoint=ckpt_8905


"""

import sys
import os
import io
from pathlib import Path

from tqdm import tqdm
from sacred import Experiment
import sacred.commands

import torch
import einops
import yaml

import numpy as np

# from paderbox.database.merl_mixtures import MerlMixtures
# from paderbox.database.keys import *
import paderbox as pb
import padertorch as pt
from paderbox.utils import mpi
from paderbox.database.iterator import AudioReader
from paderbox.transform import istft
# from tf_bss.sacred_helper import generate_path

from cbj.sacred_helper import (
    decorator_append_file_storage_observer_with_lazy_basedir,
)

# from pth_bss.utils import get_new_folder

# Unfortunately need to disable this since conda scipy needs update
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


print(sys.argv)


from cbj_lib.run import get_new_folder


ex = Experiment("lpit")
import functools


@ex.capture
@functools.lru_cache()
def get_storage_dir(model_path, checkpoint_path):
    model_path = Path(model_path)
    (model_path / 'eval').mkdir(exist_ok=True)

    storage_dir = get_new_folder(
        model_path / 'eval',
        try_id=Path(checkpoint_path).with_suffix('').name,
        consider_mpi=True,
        chdir=False,
        mkdir=False,
    )
    return Path(storage_dir).expanduser().resolve()


@decorator_append_file_storage_observer_with_lazy_basedir(
    ex, consider_mpi=True
)
def basedir():
    return get_storage_dir() / 'sacred'


def get_checkpoint_path(model_path, checkpoint):
    p1 = (Path(model_path) / 'checkpoints' / checkpoint).expanduser().resolve()
    p2 = (Path(model_path) / checkpoint).expanduser().resolve()
    p3 = Path(checkpoint).expanduser().resolve()

    if p1.exists():
        return p1
    elif p2.exists():
        return p2
    elif p3.exists():
        return p3
    else:
        raise FileNotFoundError(f"No such file or directory: '{checkpoint}'")


@ex.config
def config():
    model_path = ''
    assert len(model_path) > 0, 'Set the model path on the command line.'
    # batch_size = 1
    # test_dataset = "mix_2_spk_min_tt"

    checkpoint = 'ckpt_best_loss.pth'
    checkpoint_path = get_checkpoint_path(model_path, checkpoint)
    config_file = 'config.yaml'


from .train import Model
# from .model import levenshtein_distance


@ex.main
def main(
        _config,
        _run,
        model_path,
        # batch_size,
        # test_dataset,
        # checkpoint,
        config_file,
        checkpoint_path
):

    storage_dir = get_storage_dir()
    storage_dir.mkdir(exist_ok=True)

    print('model_path:', model_path)
    print('storage_dir:', storage_dir)
    # os.chdir(os.path.expanduser(storage_dir))

    if pb.utils.mpi.IS_MASTER:
        sacred.commands.print_config(_run)

    model = Model.from_config_and_checkpoint(
        config_path=Path(model_path) / config_file,
        checkpoint_path=checkpoint_path,
        consider_mpi=True,
    )

    db: pb.database.wsj_bss.WsjBss = model.db

    it = model.get_iterable('cv_dev93')

    summary = {}
    model.eval()

    from cbj.scheduler.mpi import share_master

    with torch.no_grad():
        for it_example in share_master(
                it,
                allow_single_worker=True,
        ):
            example_id = it_example['example_id']

            try:
                example: Model.NNInput = pt.data.example_to_device(
                    model.transform(it_example)
                )
            except pb.database.iterator.FilterException:
                continue

            predict = model(example)

            estimate = einops.rearrange(
                example.Feature[..., None] * predict,
                'D T F K -> D K T F'.lower()
            )[0]
            target = einops.rearrange(
                example.Target, 'K D T F -> D K T F'.lower()
            )[0]

            istft_kwargs = model.feature_extractor.kwargs.copy()
            del istft_kwargs['pad']

            estimate = pb.transform.istft(
                estimate,
                **istft_kwargs,
            )
            target = pb.transform.istft(
                target,
                **istft_kwargs,
            )

            mir_eval = pb.evaluation.mir_eval_sources(
                reference=target,
                estimation=estimate,
                return_dict=True,
            )

            summary[example_id] = {
                'mir_eval': mir_eval,
            }

    summaries = pb.utils.mpi.gather(list(summary.items()))
    if pb.utils.mpi.IS_MASTER:
        import itertools

        summary = dict(itertools.chain(*summaries))

        pb.io.dump_json(
            {
                'details': summary
            },
            storage_dir / 'result.json'
        )


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
