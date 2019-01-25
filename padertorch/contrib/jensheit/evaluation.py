import paderbox as pb
import numpy as np
from paderbox.utils.numpy_utils import morph


__all__ = [
    'beamforming'
]

def beamforming(observation, speech_mask, noise_mask,
                get_bf_fn=pb.speech_enhancement.get_mvdr_vector_souden):
    """

    :param observation: ...xCxTxF
    :param speech_mask: ...xCxTxF
    :param noise_mask: ...xCxTxF
    :return: predicted speech signal: ...xTxF
    """
    speech_mask = np.median(speech_mask, axis=-3).swapaxes(-2, -1)
    noise_mask = np.median(noise_mask, axis=-3).swapaxes(-2, -1)
    obs = morph('...CTF->...FCT', observation)
    covariance = pb.speech_enhancement.get_power_spectral_density_matrix
    speech_psd = covariance(obs, speech_mask)
    noise_psd = covariance(obs, noise_mask)
    mvdr = get_bf_fn(speech_psd, noise_psd)
    return pb.speech_enhancement.apply_beamforming_vector(
        mvdr, obs
    ).swap