import copy

import cached_property

import numpy as np
from einops import rearrange, reduce

import paderbox as pb


# from padertorch.contrib.cb.lpit.model import FeatureExtractor


class Metrics:

    def __init__(
            self,
            enhanced,
            speech_source,
            enhanced_speech_image=None,
            enhanced_noise_image=None,
    ):
        samples = enhanced.shape[-1]

        if enhanced_speech_image is not None:
            def get_msg(msg):
                msg = f'{msg}'
                msg += f'\nShapes:'
                msg += f'\n\tenh: {enhanced.shape} (N)'
                msg += f'\n\tenh_speech: {enhanced_speech_image.shape} (K_source, K_target, N)'
                msg += f'\n\tenh_noise: {enhanced_noise_image.shape} (K_target, N)'
                msg += f'\n\tspeech_source: {speech_source.shape} (K_source, N)'
                return msg

            ksource, ktaget, samples_ = enhanced_speech_image.shape
            assert samples == samples_, get_msg((samples, samples_))
            ktaget_, samples_ = enhanced_noise_image.shape
            assert samples == samples_, get_msg((samples, samples_))
            assert ktaget == ktaget_, get_msg((ktaget, ktaget_))

            assert ksource < 5, get_msg(ksource)
            assert ktaget < 5, get_msg(ktaget)

            ksource_, samples_ = speech_source.shape
            assert samples == samples_, get_msg((samples, samples_))
            assert ksource == ksource_, get_msg((ksource, ksource_))

        self.enhanced = enhanced
        self.enhanced_speech_image = enhanced_speech_image
        self.enhanced_noise_image = enhanced_noise_image
        self.speech_source = speech_source

    @cached_property.cached_property
    def enhanced_speech(self):
        assert self.enhanced.ndim == 2, self.enhanced.shape
        assert self.enhanced.shape[0] < 10, self.enhanced.shape
        assert self.enhanced.shape[0] == len(self.selection) + 1, self.enhanced.shape
        return self.enhanced[self.selection]

    @cached_property.cached_property
    def mir_eval(self):
        return pb.evaluation.mir_eval_sources(
            reference=self.speech_source,
            estimation=self.enhanced,
            return_dict=True,
        )

    @cached_property.cached_property
    def pesq(self):
        try:
            return pb.evaluation.pesq(
                reference=self.speech_source,
                degraded=self.enhanced_speech,
                rate=8000,
                mode='nb',
            )
        except OSError:
            pass

        try:
            return pb.evaluation.pesq(
                reference=self.speech_source,
                degraded=self.enhanced_speech,
                rate=8000,
                mode='nb',
            )
        except OSError:
            return np.nan

    @cached_property.cached_property
    def pypesq(self):
        import pypesq

        assert self.speech_source.shape == self.enhanced_speech.shape, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.ndim == 2, (self.speech_source.shape, self.enhanced_speech.shape)
        assert self.speech_source.shape[0] < 5, (self.speech_source.shape, self.enhanced_speech.shape)

        return [
            pypesq.pypesq(ref=ref, deg=deg, fs=8000, mode='nb')
            for ref, deg in zip(self.speech_source, self.enhanced_speech)
        ]

    @cached_property.cached_property
    def selection(self):
        return self.mir_eval['permutation']

    @cached_property.cached_property
    def sxr(self):
        invasive_sxr = pb.evaluation.output_sxr(
            # rearrange(beamformed_clean, 'ksource ktaget samples -> ktaget ksource samples'),
            rearrange(
                self.enhanced_speech_image,
                'ksource ktaget samples -> ksource ktaget samples'
            )[:, self.selection, :],
            rearrange(
                self.enhanced_noise_image, 'ktaget samples -> ktaget samples'
            )[self.selection, :],
            return_dict=True,
        )
        return invasive_sxr

    # def asasd(self):
    #     import tf_bss.predict
    #     tf_bss.predict.run_evaluation(
    #         speech_image,
    #         noise_image,
    #         speech_prediction,
    #         speech_contribution,
    #         noise_contribution,
    #         reference_channel=0,
    #         sample_rate=8000,
    #     )

    @classmethod
    def from_beamformed(
            cls,
            mask,  # T Ktarget F
            Observation,  # D T F (stft signal)
            Speech_image,  # Ksource D T F (stft signal)
            Noise_image,  # D T F (stft signal)
            speech_source,  # K N (time signal)
            istft,  # callable(signal, samples=samples)
            Observation_image=None,  # D T F (stft signal)
            bf_algorithm='gev_rank1_mvdr_souden_ban',
    ):
        """
        ToDo: consider WPE that is applied on Observation

        Args:
            mask:
            Observation:
            Speech_image:
            Noise_image:
            speech_source:
            istft:
            Observation_image
            bf_algorithm:

        Returns:

        >>> np.random.seed(0)
        >>> from cbj.pytorch.feature_extractor import STFT
        >>> K_source, K_target, D, N = (2, 3, 6, 2000)

        >>> feature_extractor = STFT()

        >>> source = np.random.randn(*[K_source, N])
        >>> clean = source[:, None, :] + 0.1 * np.random.randn(*[K_source, D, N])
        >>> noise = 0.1 * np.random.randn(*[D, N])
        >>> observation = np.sum(clean, axis=0) + noise

        >>> Clean = feature_extractor(clean)
        >>> Noise = feature_extractor(noise)
        >>> Observation = feature_extractor(observation)

        >>> *_, T, F = Clean.shape

        >>> masks = np.ones([T, K_target, F])

        >>> metrics = Metrics.from_beamformed(masks, Observation, Clean, Noise, source, feature_extractor.istft)
        >>> metrics.mir_eval
        {'sdr': array([-9.16698329, -3.36861713]), 'sir': array([-4.89098757,  3.98217684]), 'sar': array([-1.02479974, -1.02479974]), 'permutation': array([0, 1])}

        >>> metrics.pesq
        array([2.436, 2.174])
        >>> metrics.sxr
        {'sdr': -2.872081633804722, 'sir': 5.551115123125783e-17, 'snr': 0.2845825847275161}

        >>> masks = np.abs(Clean) / np.abs(Observation)
        >>> masks = reduce(masks, 'k d t f -> t k f', 'mean')

        >>> metrics = Metrics.from_beamformed(masks, Observation, Clean, Noise, source, feature_extractor.istft)
        >>> metrics.mir_eval
        {'sdr': array([2.06123106, 1.99629643]), 'sir': array([7.65268272, 7.32708473]), 'sar': array([4.1517464 , 4.23969479]), 'permutation': array([0, 1])}
        >>> metrics.pesq
        array([3.219, 2.678])
        >>> metrics.sxr
        {'sdr': -1.770731059827935, 'sir': 0.8418365252948881, 'snr': 1.6776824700103203}

        >>> masks = np.abs(Clean) / np.abs(Observation)
        >>> masks = reduce(masks, 'k d t f -> t k f', 'mean')

        >>> metrics = Metrics.from_beamformed(masks, Observation, Clean, Noise, source, feature_extractor.istft, Observation_image=Observation)
        >>> metrics.mir_eval
        {'sdr': array([2.06123106, 1.99629643]), 'sir': array([7.65268272, 7.32708473]), 'sar': array([4.1517464 , 4.23969479]), 'permutation': array([0, 1])}
        >>> metrics.pesq
        array([3.219, 2.678])
        >>> metrics.sxr
        {'sdr': -1.770731059827935, 'sir': 0.8418365252948881, 'snr': 1.6776824700103203}

        >>> masks = np.abs(Clean) / np.abs(Observation)
        >>> masks = reduce(masks, 'k d t f -> t k f', 'mean')

        >>> metrics = Metrics.from_masked(masks, Observation, Clean, Noise, source, feature_extractor.istft)
        >>> metrics.mir_eval
        {'sdr': array([4.18588544, 4.33405695]), 'sir': array([5.29488686, 5.51248187]), 'sar': array([11.78138385, 11.6503694 ]), 'permutation': array([0, 1])}

        >>> metrics.pesq
        array([3.948, 3.247])
        >>> metrics.sxr
        {'sdr': 1.5880883634117893, 'sir': 1.6399432777052907, 'snr': 20.8442268520095}
        """
        assert mask.ndim == 3, mask.shape

        if Observation_image is None:
            Observation_image = Observation
            Observation_dereverbated = None
        else:
            Observation_dereverbated = Observation

        Beamformed = None
        Beamformed_clean = None
        Beamformed_noise = None
        if Speech_image is not None and Noise_image is not None:
            Beamformed, Beamformed_clean, Beamformed_noise = beamform(
                # np.sum(Speech_image, axis=0) + Noise_image,  # drop WPE
                Observation_image,
                mask,
                Speech_image=Speech_image,
                Noise_image=Noise_image,
                algorithm=bf_algorithm,
            )

        if Beamformed is None or Observation_dereverbated is not None:
            if Beamformed is None and Observation_dereverbated is None:
                Observation_dereverbated = Observation
            # mir_eval and pesq can handle a dereverbated signal
            # SXR is not well defined, when the signal is dereverbated
            # -> recalculated beamformed signal from dereverbated signal.
            Beamformed = beamform(
                Observation_dereverbated,
                mask,
                # Speech_image=Speech_image,
                # Noise_image=Noise_image,
                algorithm=bf_algorithm,
            )

        K, N = speech_source.shape

        beamformed = istft(Beamformed, samples=N)
        if Beamformed_clean is None:
            beamformed_clean = None
        else:
            beamformed_clean = istft(Beamformed_clean, samples=N)
        if Beamformed_noise is None:
            beamformed_noise = None
        else:
            beamformed_noise = istft(Beamformed_noise, samples=N)

        return cls(
            enhanced=beamformed,
            speech_source=speech_source,
            enhanced_speech_image=beamformed_clean,
            enhanced_noise_image=beamformed_noise,
        )

    @classmethod
    def from_masked(
            cls,
            mask,  # T Ktarget F
            Observation,  # D T F (stft signal)
            Speech_image,  # Ksource D T F (stft signal)
            Noise_image,  # D T F (stft signal)
            speech_source,  # D N (time signal)
            istft,  # callable(signal, samples=samples)
            ref_channel=0,
    ):
        assert mask.ndim == 3, mask.shape

        Estimate = rearrange(
            Observation[ref_channel, ..., None, :] * mask,
            'T K F -> K T F'.lower()
        )

        K, N = speech_source.shape

        estimate = istft(Estimate, samples=N)

        if Speech_image is not None and Noise_image is not None:
            Estimate_clean = rearrange(
                Speech_image[:, ref_channel, ..., None, :] * mask,
                'ksource t ktarget f -> ksource ktarget t f'
            )
            Estimate_noise = rearrange(
                Noise_image[ref_channel, ..., None, :] * mask,
                't ktarget F -> ktarget t f'.lower()
            )
            estimate_clean = istft(Estimate_clean, samples=N)
            estimate_noise = istft(Estimate_noise, samples=N)
        else:
            assert Speech_image is None, Speech_image
            assert Noise_image is None, Noise_image
            estimate_clean = None
            estimate_noise = None

        return cls(
            enhanced=estimate,
            speech_source=speech_source,
            enhanced_speech_image=estimate_clean,
            enhanced_noise_image=estimate_noise,
        )

# def get_sxr(
#         pooled_predict,
#         Observation,
#         Speech_image,
#         Noise_image,
#         selection,
#         istft,
#         bf_algorithm='gev_rank1_mvdr_souden_ban',
# ):
#     Beamformed, Beamformed_clean, Beamformed_noise = beamform(
#         Observation,
#         pooled_predict,
#         Speech_image=Speech_image,
#         Noise_image=Noise_image,
#         algorithm=bf_algorithm,
#     )
#
#     beamformed = istft(Beamformed)
#     beamformed_clean = istft(Beamformed_clean)
#     beamformed_noise = istft(Beamformed_noise)
#
#     bf_invasive_sxr = pb.evaluation.output_sxr(
#         # rearrange(beamformed_clean, 'ksource ktaget samples -> ktaget ksource samples'),
#         rearrange(beamformed_clean,
#                   'ksource ktaget samples -> ksource ktaget samples')[:, selection, :],
#         rearrange(beamformed_noise, 'ktaget samples -> ktaget samples')[
#         selection, :],
#         return_dict=True,
#     )
#
#     return bf_invasive_sxr


def beamform(
        Observation,
        masks,
        algorithm='gev_rank1_mvdr_souden_ban',
        Speech_image=None,
        Noise_image=None,
):
    """

    Args:
        Observation: shape: ... D T F
                            e.g.: (D T F) for Observation
                                  or (Ksource D T F) for Speech_image
        masks: shape: T Ktarget F

    Returns:
        shape: Ktaget T F
        or
        tuple(
            shape: Ktaget T F           # Observation enhanced
            shape: Ksource Ktaget T F   # beamformed Speech_image
            shape: Ktaget T F           # beamformed Noise_image
        )



    >>> k_source, k_target, d, t, f = (2, 3, 6, 4, 5)
    >>> Clean = np.ones([k_source, d, t, f])
    >>> Noise = np.ones([d, t, f])
    >>> Observation = np.ones([d, t, f])
    >>> masks = np.ones([t, k_target, f])
    >>> beamform(Clean, masks).shape
    Traceback (most recent call last):
    ...
    ValueError: too many values to unpack (expected 3)
    >>> beamform(Observation, masks).shape
    (3, 4, 5)
    >>> O, S, N = beamform(Observation, masks, Speech_image=Clean, Noise_image=Noise)
    >>> O.shape, S.shape, N.shape
    ((3, 4, 5), (2, 3, 4, 5), (3, 4, 5))

    >>> _ = beamform(Observation, masks, 'gev_rank1_mvdr_souden_ban')
    >>> _ = beamform(Observation, masks, 'mvdr_souden_ban')
    >>> _ = beamform(Observation, masks, 'mvdr_souden')
    >>> _ = beamform(Observation, masks, 'gev_rank1_mvdr_souden')
    >>> _ = beamform(Observation, masks, 'gev_rank1_gev_ban')
    >>> _ = beamform(Observation, masks, 'gev_ban')
    >>> _ = beamform(Observation, masks, 'gev')
    >>> _ = beamform(Observation, masks, 'gev_rank1_gev')

    >>> _ = beamform(Observation, masks, 'mvdr_souden_ban_ban')
    Traceback (most recent call last):
    ...
    AssertionError: ('ban', 'mvdr_souden_ban_ban')
    >>> _ = beamform(Observation, masks, 'mvdr')
    Traceback (most recent call last):
    ...
    ValueError: ('mvdr', 'mvdr')
    """
    from paderbox.speech_enhancement.beamformer import (
        get_power_spectral_density_matrix,
        get_mvdr_vector_souden,
        apply_beamforming_vector,
        blind_analytic_normalization,
        get_gev_vector,
        _gevd_rank_one_estimate,
    )

    # Observation.shape: KSource D T F
    # masks.shape: T Ktarget F
    # should return: Ktaget T F

    assert (Speech_image is None) == (Noise_image is None), (Speech_image, Noise_image)
    have_image = not(Speech_image is None)

    D, T, F = Observation.shape
    _, Ktarget, _ = masks.shape

    assert Ktarget < 30, (Ktarget, Observation.shape, masks.shape)
    assert D < 30, (D, Observation.shape, masks.shape)

    Observation = rearrange(Observation, 'd t f -> f d t')
    if have_image:
        Ksource, _, _, _ = Speech_image.shape
        assert Ksource < 30, (Ksource, Observation.shape, masks.shape, Speech_image.shape)

        Speech_image = rearrange(Speech_image, 'ksource d t f -> ksource f d t', d=D, f=F, t=T)
        Noise_image = rearrange(Noise_image, 'd t f -> f d t', d=D, f=F, t=T)

    masks = rearrange(masks, 't ktarget f -> ktarget f t', t=T, f=F)

    # Remove zeros.
    masks = np.clip(masks, 1e-10, None)

    # Add broadcasting dimensions. Needed for get_power_spectral_density_matrix.
    # while masks.ndim < Observation.ndim:
    #     masks = masks[None, ...]

    psds = get_power_spectral_density_matrix(
        Observation[None, :, :, :],
        masks,
    )  # shape: ktarget, f, d, d

    Enhanced = []
    Speech_image_enhanced = []
    Noise_image_enhanced = []

    for target in range(len(psds)):
        algorithm_copy = copy.copy(algorithm)

        def algorithm_contains(part):
            nonlocal algorithm_copy
            if algorithm_copy.startswith(part):
                algorithm_copy = algorithm_copy[len(part):].lstrip('_')
                return True
            else:
                return False

        target_pds = psds[target]
        distortion_pds = np.sum(np.delete(psds, target, axis=0), axis=0)

        if algorithm_contains('gev_rank1_'):
            target_pds = _gevd_rank_one_estimate(target_pds, distortion_pds)

        if algorithm_contains('gev'):
            w = get_gev_vector(target_pds, distortion_pds)
        elif algorithm_contains('mvdr_souden'):
            w = get_mvdr_vector_souden(target_pds, distortion_pds)
        else:
            raise ValueError(algorithm_copy, algorithm)

        if algorithm_contains('ban'):
            w = blind_analytic_normalization(w, noise_psd_matrix=distortion_pds)

        assert algorithm_copy == '', (algorithm_copy, algorithm)

        Enhanced.append(apply_beamforming_vector(w, Observation))
        if have_image:
            Speech_image_enhanced.append(apply_beamforming_vector(w, Speech_image))
            Noise_image_enhanced.append(apply_beamforming_vector(w, Noise_image))

    Enhanced = np.array(Enhanced)
    Speech_image_enhanced = np.array(Speech_image_enhanced)
    Noise_image_enhanced = np.array(Noise_image_enhanced)

    if have_image:
        return (
            rearrange(Enhanced, 'ktarget f t -> ktarget t f'),
            rearrange(Speech_image_enhanced, 'ktarget ksource f t -> ksource ktarget t f'),
            rearrange(Noise_image_enhanced, 'ktarget f t -> ktarget t f'),
        )
    else:
        return rearrange(Enhanced, 'ktarget f t -> ktarget t f')