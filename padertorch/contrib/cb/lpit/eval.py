"""


mpiexec -np 8 python -m cbj.pytorch.eval_am with model_path=..

Other options:
# checkpoint=ckpt_8905


"""

if not __package__:
    from cbj_lib import set_package
    set_package()

import sys
import os
import io
from pathlib import Path

from tqdm import tqdm
from sacred import Experiment
import sacred.commands

import torch
import einops
from einops import rearrange
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

from paderbox.utils import nested

import cbj
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


def beamform(Observation, masks):
    """

    Args:
        Observation: shape: ... D T F
        masks: shape: T F Ktarget

    Returns:
        shape: ... Ktaget T F


    >>> k_source, k_target, d, t, f = (2, 3, 6, 4, 5)
    >>> Clean = np.ones([k_source, d, t, f])
    >>> Observation = np.ones([d, t, f])
    >>> masks = np.ones([t, f, k_target])
    >>> beamform(Clean, masks).shape
    (2, 3, 4, 5)
    >>> beamform(Observation, masks).shape
    (3, 4, 5)
    """
    from paderbox.speech_enhancement.beamformer import (
        get_power_spectral_density_matrix,
        get_mvdr_vector_souden,
        apply_beamforming_vector,
        blind_analytic_normalization,
        get_gev_vector,
    )

    # Observation.shape: KSource D T F
    # masks.shape: T F Ktarget
    # should return: Ktaget T F

    Observation = rearrange(Observation, '... d t f -> ... f d t')
    masks = rearrange(masks, 't f ktarget -> ktarget f t')

    # Remove zeros.
    masks = np.clip(masks, 1e-10, None)

    # Add broadcasting dimensions. Needed for get_power_spectral_density_matrix.
    while masks.ndim < Observation.ndim:
        masks = masks[None, ...]

    psds = get_power_spectral_density_matrix(
        Observation[..., None, :, :, :],
        masks,
    )  # shape: ..., ktarget, f, d, d

    iter_shape = psds.shape[:-4]
    ktarget, f, _, _ = psds.shape[-4:]
    t = Observation.shape[-1]

    out = np.empty([*iter_shape, ktarget, f, t], dtype=Observation.dtype)

    for idx in np.ndindex(iter_shape):
        sub_psds = psds[idx]

        for target in range(len(sub_psds)):
            target_pds = sub_psds[target]
            distortion_pds = np.sum(np.delete(sub_psds, target, axis=0), axis=0)
            # w = get_mvdr_vector_souden(target_pds, distortion_pds)
            w = get_gev_vector(target_pds, distortion_pds)
            w = blind_analytic_normalization(w, noise_psd_matrix=distortion_pds)

            out[(*idx, target, slice(None), slice(None))] = apply_beamforming_vector(w, Observation[idx])

    return rearrange(out, '... ktarget f t -> ... ktarget t f')


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

    if mpi.IS_MASTER:
        os.symlink(checkpoint_path, storage_dir / 'checkpoint')

    db: pb.database.wsj_bss.WsjBss = model.db

    it = model.get_iterable('cv_dev93', snr_range=(20, 30))
    it = it.copy(freeze=True).sort()

    summary = {}
    model.eval()

    from cbj.scheduler.mpi import share_master
    # from cbj.scheduler.mpi import mpi_lazy_parallel_unordered_map

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
                return None, 'exclude'

            predict = pt.utils.to_numpy(model(example).predict)

            # example['audio_data']['speech_image']
            # example['audio_data']['noise_image']

            observation = it_example['audio_data']['observation']
            speech_image = it_example['audio_data']['speech_image']
            speech_source = it_example['audio_data']['speech_source']
            noise_image = it_example['audio_data']['noise_image']

            Observation = model.feature_extractor(observation)
            Speech_image = model.feature_extractor(speech_image)
            Noise_image = model.feature_extractor(noise_image)

            # target = np.array([*speech_image])
            # target = speech_image

            Estimate = rearrange(Observation[..., None] * predict,
                                 'D T F K -> D K T F'.lower())[0]
            Estimate_clean = rearrange(Speech_image[..., None] * predict,
                                       'KSource D T F Ktaget -> D KSource Ktaget T F'.lower())[0]
            Estimate_noise = rearrange(Noise_image[..., None] * predict,
                                       'D T F Ktaget -> D Ktaget T F'.lower())[0]
            speech_source = rearrange(speech_source, 'Kplus1 N -> Kplus1 N'.lower())

            pooled_predict = np.mean(predict, axis=0)

            Beamformed = beamform(Observation, pooled_predict)
            Beamformed_clean = beamform(Speech_image, pooled_predict)
            Beamformed_noise = beamform(Noise_image, pooled_predict)

            istft_kwargs = model.feature_extractor.kwargs.copy()
            del istft_kwargs['pad']

            def istft(STFTSignal):
                return pb.transform.istft(
                    STFTSignal, **istft_kwargs,
                )[..., :observation.shape[-1]]

            estimate = istft(Estimate)
            estimate_clean = istft(Estimate_clean)
            estimate_noise = istft(Estimate_noise)

            beamformed = istft(Beamformed)
            beamformed_clean = istft(Beamformed_clean)
            beamformed_noise = istft(Beamformed_noise)

            mir_eval = pb.evaluation.mir_eval_sources(
                reference=speech_source, estimation=estimate, return_dict=True)
            bf_mir_eval = pb.evaluation.mir_eval_sources(
                reference=speech_source, estimation=beamformed, return_dict=True)

            selection = bf_mir_eval['permutation']

            invasive_sxr = pb.evaluation.output_sxr(
                # rearrange(estimate_clean, 'ksource ktaget samples -> ktaget ksource samples'),
                rearrange(estimate_clean, 'ksource ktaget samples -> ksource ktaget samples')[:, selection, :],
                rearrange(estimate_noise, 'ktaget samples -> ktaget samples')[selection, :],
                return_dict=True)

            bf_invasive_sxr = pb.evaluation.output_sxr(
                # rearrange(beamformed_clean, 'ksource ktaget samples -> ktaget ksource samples'),
                rearrange(beamformed_clean, 'ksource ktaget samples -> ksource ktaget samples')[:, selection, :],
                rearrange(beamformed_noise, 'ktaget samples -> ktaget samples')[selection, :],
                return_dict=True)

            del mir_eval['permutation']
            del bf_mir_eval['permutation']

            summary[example_id] = nested.flatten({
                'mir_eval': mir_eval,
                'invasive': invasive_sxr,
                'bf_mir_eval': bf_mir_eval,
                'bf_invasive_sxr': bf_invasive_sxr,
            }, sep=None)
            # return example_id, nested.flatten({
            #     'mir_eval': mir_eval,
            #     'invasive': invasive_sxr,
            #     'bf_mir_eval': bf_mir_eval,
            #     'bf_invasive_sxr': bf_invasive_sxr,
            # }, sep=None)

        # for it_example in share_master(
        #         it,
        #         allow_single_worker=True,
        # ):
        #     calculate_scores()
        #     # 'exclude'

        # summary = {
        #     example_id: scores
        #     for example_id, scores in map_unordered(
        #         calculate_scores,
        #         it,
        #     )
        #     if scores != 'exclude'
        # }

    summaries = pb.utils.mpi.gather(list(summary.items()))
    if pb.utils.mpi.IS_MASTER:
        import itertools

        summary = dict(itertools.chain(*summaries))

        score_keys = next(iter(summary.values())).keys()

        scores = {
            # score_key: {
            #     metric: np.mean([e[score_key][metric] for e in summary.values()])
            #     for metric in ['sdr', 'sir', 'sar']
            # }
            score_key: np.mean([e[score_key] for e in summary.values()])
            for score_key in score_keys
            # 'invasive': {
            #     metric: np.mean([e['invasive'][metric] for e in summary.values()])
            #     for metric in ['sdr', 'sir', 'snr']
            # }
        }
        scores = nested.deflatten(scores, sep=None)

        summary = {
            k: nested.deflatten(v, sep=None)
            for k, v in summary.items()
        }

        pb.io.dump_json(
            {
                'details': summary,
                'scores': scores,
            },
            storage_dir / 'result.json'
        )
        print('scores', scores)


if __name__ == '__main__':
    with pb.utils.debug_utils.debug_on(Exception):
        ex.run_commandline()
