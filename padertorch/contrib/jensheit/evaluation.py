import numpy as np

import paderbox as pb
from padercontrib.database import keys as DB_K
from paderbox.utils.numpy_utils import morph
from padertorch.data import example_to_device
from padertorch.modules.mask_estimator import MaskKeys as M_K

__all__ = [
    'beamforming'
]


def beamforming(observation, speech_mask, noise_mask,
                speech_image=None, noise_image=None,
                get_bf_fn=pb.speech_enhancement.get_mvdr_vector_souden):
    """

    :param observation: ...xCxTxF
    :param speech_mask: ...xCxTxF
    :param noise_mask: ...xCxTxF
    :param speech_image: ...xCxTxF
    :param noise_image: ...xCxTxF
    :return: predicted speech signal: ...xTxF
    """
    speech_mask = np.median(speech_mask, axis=-3).swapaxes(-2, -1)
    noise_mask = np.median(noise_mask, axis=-3).swapaxes(-2, -1)
    obs = morph('...ctf->...fct', observation)
    covariance = pb.speech_enhancement.get_power_spectral_density_matrix
    speech_psd = covariance(obs, speech_mask)
    noise_psd = covariance(obs, noise_mask)
    bf_vec = get_bf_fn(speech_psd, noise_psd)
    speech_pred = pb.speech_enhancement.apply_beamforming_vector(
        bf_vec, obs).swapaxes(-2, -1)
    if speech_image is not None:
        image_contribution = pb.speech_enhancement.apply_beamforming_vector(
            bf_vec, morph('...ctf->...fct', speech_image)).swapaxes(-2, -1)
    else:
        image_contribution = None
    if noise_image is not None:
        noise_contribution = pb.speech_enhancement.apply_beamforming_vector(
            bf_vec, morph('...ctf->...fct', noise_image)).swapaxes(-2, -1)
    else:
        noise_contribution = None
    return speech_pred, image_contribution, noise_contribution


def evaluate_masks(example, model, stft):
    model_out = model(example_to_device(example))
    speech_image = example[DB_K.SPEECH_IMAGE][0]
    speech_pred, image_cont, noise_cont = beamforming(
        example[M_K.OBSERVATION_STFT][0],
        model_out[M_K.SPEECH_MASK_PRED][0].detach().numpy(),
        model_out[M_K.NOISE_MASK_PRED][0].detach().numpy(),
        stft(speech_image),
        stft(example[DB_K.NOISE_IMAGE][0])
    )
    ex_id = example[DB_K.EXAMPLE_ID][0]
    pesq = pb.evaluation.pesq(example[DB_K.SPEECH_IMAGE][0][0],
                              stft.inverse(speech_pred))[0]
    snr = np.mean(-10 * np.log10(np.abs(image_cont) ** 2
                                 / np.abs(noise_cont) ** 2))
    print(ex_id, snr, pesq)
    return ex_id, snr, pesq
