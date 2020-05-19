import numpy as np
import padertorch as pt
import torch
import torch.nn.functional as F
from padertorch.modules.mask_estimator import MaskEstimator
from padertorch.modules.mask_estimator import MaskKeys as K
from padertorch.ops.mappings import TORCH_POOLING_FN_MAP

from padertorch.summary import mask_to_image, stft_to_image

class MaskLossKeys:
    NOISE_MASK = 'noise_mask_loss'
    SPEECH_MASK = 'speech_mask_loss'
    WEIGHTED_NOISE_MASK = 'power_weighted_noise_mask_loss'
    WEIGHTED_SPEECH_MASK = 'power_weighted_speech_mask_loss'
    MASK = 'mask_loss'
    WEIGHTED_MASK = 'power_weighted_mask_loss'
    TOTAL_MASK = 'total_mask_loss'
    VAD = 'VAD_loss'
    REC = 'reconstruction_loss'


class MaskEstimatorModel(pt.Model):
    """
    Implements a mask estimator [1].

    Example usage may be found in padertorch.contrib.jensheit:

    [1] Heymann 2016, https://groups.uni-paderborn.de/nt/pubs/2016/icassp_2016_heymann_paper.pdf


    ToDo: normalization
    ToDo: input transform/ at least a logarithm or spectrogram
    ToDo: Add vad estimation?

    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['estimator'] = dict(factory=MaskEstimator)

    def __init__(self, estimator, transformer=None, reduction: str = 'average',
                 sample_rate: int = 16000):
        super().__init__()
        self.estimator = estimator
        self.transformer = transformer
        self.reduction = reduction
        self.sample_rate = sample_rate

    def forward(self, batch):
        """
        :param batch: dict of lists with key observation_abs
                observation_abs is a list of tensors with shape C,T,F
        :return:
        """
        obs = batch[K.OBSERVATION_ABS]
        num_frames = batch[K.NUM_FRAMES]
        out = {key: [v[:, :frames] for v, frames in zip(value, num_frames)]
               for key, value in self.estimator(obs).items()}
        assert isinstance(out, dict)
        return out

    def review(self, batch, output):
        """
        :param batch: dict of lists
        :param output: output of the forward step
        :return:
        """
        losses = self.add_losses(batch, output)
        return dict(loss= losses.pop(MaskLossKeys.MASK),
                    scalars=losses,
                    audios=self.add_audios(batch, output),
                    images=self.add_images(batch, output)
                    )

    def add_images(self, batch, output):
        images = dict()
        if K.SPEECH_PRED in output:
            speech_pred = output[K.SPEECH_PRED][0]
            images['speech_pred'] = mask_to_image(speech_pred, True)
        if K.SPEECH_MASK_PRED in output:
            speech_mask = output[K.SPEECH_MASK_PRED][0]
            images['speech_mask'] = mask_to_image(speech_mask, True)
        observation = batch[K.OBSERVATION_ABS][0]
        images['observed_stft'] = stft_to_image(observation, True)
        if K.NOISE_MASK_PRED in output:
            noise_mask = output[K.NOISE_MASK_PRED][0]
            images['noise_mask'] = mask_to_image(noise_mask, True)
        if batch is not None and K.SPEECH_MASK_TARGET in batch:
            images['speech_mask_target'] = mask_to_image(
                batch[K.SPEECH_MASK_TARGET][0], True)
            if K.NOISE_MASK_TARGET in batch:
                images['noise_mask_target'] = mask_to_image(
                    batch[K.NOISE_MASK_TARGET][0], True)
        return images

    # ToDo: add scalar review

    def add_audios(self, batch, output):
        audio_dict = dict()
        if K.OBSERVATION in batch:
            audio_dict.update({K.OBSERVATION: (maybe_remove_channel(
                batch[K.OBSERVATION][0]), self.sample_rate)})
        if K.SPEECH_IMAGE in batch:
            audio_dict.update({
            K.SPEECH_IMAGE: (maybe_remove_channel(batch[K.SPEECH_IMAGE][0]),
                             self.sample_rate)})
        if K.SPEECH_PRED in output and self.transformer is not None:
            phase = np.exp(1j*np.angle(batch[K.OBSERVATION_STFT][0]))
            enh = output[K.SPEECH_PRED][0].detach().cpu().numpy() * phase
            enhanced_time = self.transformer.inverse(enh)
            audio_dict.update({K.SPEECH_PRED: (maybe_remove_channel(
                enhanced_time), self.sample_rate)})
        if K.SPEECH_MASK_PRED in output and self.transformer is not None:
            obs = batch[K.OBSERVATION_STFT][0]
            enh = output[K.SPEECH_MASK_PRED][0].detach().cpu().numpy() * obs
            enhanced_time = self.transformer.inverse(enh)
            audio_dict.update({K.SPEECH_MASK_PRED: (maybe_remove_channel(
                enhanced_time), self.sample_rate)})
        return audio_dict

    def add_losses(self, batch, output):
        noise_loss = list()
        speech_loss = list()
        speech_reconstruction_loss = list()
        vad_loss = list()
        weighted_noise_loss = list()
        weighted_speech_loss = list()
        power_weights = None
        for idx, observation_abs in enumerate(batch[K.OBSERVATION_ABS]):
            if K.OBSERVATION_STFT in batch:
                power_weights = np.abs(batch[K.OBSERVATION_STFT][idx]) ** 2
                power_weights = observation_abs.new(power_weights)
            if K.SPEECH_MASK_TARGET in batch:
                speech_mask_target = batch[K.SPEECH_MASK_TARGET][idx]
            else:
                speech_mask_target = None
            if K.SPEECH_TARGET in batch:
                speech_target = batch[K.SPEECH_TARGET][idx]
            else:
                speech_target = None
            if K.NOISE_MASK_TARGET in batch:
                noise_mask_target = batch[K.NOISE_MASK_TARGET][idx]
            else:
                noise_mask_target = None
            if K.NOISE_MASK_LOGITS in output:
                noise_mask_logits = output[K.NOISE_MASK_LOGITS][idx]
            else:
                noise_mask_logits = None

            if K.SPEECH_MASK_LOGITS in output:
                speech_mask_logits = output[K.SPEECH_MASK_LOGITS][idx]
            else:
                speech_mask_logits = None

            if K.SPEECH_PRED in output:
                speech_pred = output[K.SPEECH_PRED][idx]
            else:
                speech_pred = None

            def get_loss(target, logits):
                return F.binary_cross_entropy_with_logits(
                    input=logits, target=target,
                    reduction='none')

            def weight_loss(sample_loss, power_weights):
                return torch.sum(sample_loss * power_weights) / torch.sum(
                    power_weights)

            # Noise mask
            if noise_mask_target is not None and noise_mask_logits is not None:
                sample_loss = get_loss(noise_mask_target, noise_mask_logits)
                noise_loss.append(
                    TORCH_POOLING_FN_MAP[self.reduction](sample_loss))
                if power_weights is not None:
                    weighted_noise_loss.append(
                        weight_loss(sample_loss, power_weights))
            # Speech mask
            if speech_mask_target is not None\
                    and speech_mask_logits is not None:
                sample_loss = get_loss(speech_mask_target, speech_mask_logits)
                speech_loss.append(
                    TORCH_POOLING_FN_MAP[self.reduction](sample_loss))
                if power_weights is not None:
                    weighted_speech_loss.append(
                        weight_loss(sample_loss, power_weights))

            if speech_target is not None and speech_pred is not None:
                sample_loss = F.mse_loss(
                    speech_target, speech_pred)
                speech_reconstruction_loss.append(
                    TORCH_POOLING_FN_MAP[self.reduction](sample_loss))

            # VAD
            if K.VAD in batch and K.VAD_LOGITS in output:
                vad_target = batch[K.VAD][idx]
                vad_logits = output[K.VAD_LOGITS][idx]
                vad_loss.append(get_loss(
                    vad_target, vad_logits))

        loss = []
        weighted_loss = []
        losses = dict()

        if len(speech_reconstruction_loss) > 0:
            speech_reconstruction_loss = sum(speech_reconstruction_loss)
            loss.append(speech_reconstruction_loss)
            losses[MaskLossKeys.REC] = speech_reconstruction_loss
        if len(noise_loss) > 0:
            noise_loss = sum(noise_loss)
            loss.append(noise_loss)
            losses[MaskLossKeys.NOISE_MASK] = noise_loss
            if power_weights is not None:
                weighted_loss.append(sum(weighted_noise_loss))
        if len(speech_loss) > 0:
            speech_loss = sum(speech_loss)
            loss.append(speech_loss)
            losses[MaskLossKeys.SPEECH_MASK] = speech_loss
            if power_weights is not None:
                weighted_loss.append(sum(weighted_speech_loss))
        if len(vad_loss) > 0:
            vad_loss = sum(vad_loss)
            loss.append(vad_loss)
            losses[MaskLossKeys.VAD] = vad_loss
        if len(loss) > 0:
            loss = sum(loss)
            if power_weights is not None:
                weighted_loss = sum(weighted_loss)
                losses[MaskLossKeys.WEIGHTED_MASK] = weighted_loss
            losses[MaskLossKeys.MASK] = loss
        return losses

def maybe_remove_channel(signal, exp_dim=1, ref_channel=0):
    if isinstance(signal, torch.Tensor):
        dim = signal.dim()
    elif isinstance(signal, np.ndarray):
        dim = signal.ndim
    else:
        raise ValueError
    if dim == exp_dim + 1:
        assert signal.shape[0] < 20, f'The first dim is supposed to be the ' \
            f'channel dimension, however the shape is {signal.shape}'
        return signal[ref_channel]
    elif dim == exp_dim:
        return signal
    else:
        raise ValueError(f'Either the signal has ndim {exp_dim} or'
                         f' {exp_dim +1}', signal.shape)
