import numpy as np
import padertorch as pt
import torch
import torch.nn.functional as F
from padertorch.data.utils import collate_fn
from padertorch.modules.mask_estimator import MaskEstimator
from padertorch.modules.mask_estimator import MaskKeys as M_K
from padertorch.ops.mappings import POOLING_FN_MAP
from paderbox.database import keys as K

class MaskLossKeys:
    NOISE_MASK = 'noise_mask_loss'
    SPEECH_MASK = 'speech_mask_loss'
    WEIGHTED_NOISE_MASK = 'power_weighted_noise_mask_loss'
    WEIGHTED_SPEECH_MASK = 'power_weighted_speech_mask_loss'
    MASK = 'mask_loss'
    WEIGHTED_MASK = 'power_weighted_mask_loss'
    TOTAL_MASK = 'total_mask_loss'
    VAD = 'VAD_loss'


class MaskEstimatorModel(pt.Model):
    """
    Implements a mask estimator [1].

    Check out this repository to see example code:
    git clone git@ntgit.upb.de:scratch/jensheit/padertorch_mask_example

    [1] Heymann 2016, https://groups.uni-paderborn.de/nt/pubs/2016/icassp_2016_heymann_paper.pdf


    ToDo: normalization
    ToDo: input transform/ at least a logarithm or spectrogram
    ToDo: Add vad estimation?

    """

    @classmethod
    def get_signature(cls):
        default_dict = super().get_signature()
        default_dict['estimator'] = {
            'cls': MaskEstimator,
            'kwargs': dict()
        }
        return default_dict

    def __init__(self, estimator, reduction: str = 'mean'):
        super().__init__()
        self.estimator = estimator
        self.reduction = reduction

    def forward(self, batch):
        """
        :param batch: dict of lists with key observation_abs
                observation_abs is a list of tensors with shape C,T,F
        :return:
        """
        obs = batch[M_K.OBSERVATION_ABS]
        num_channels = obs[0].shape[0]
        if num_channels == 1:
            obs = pt.pack_sequence([tensor[0] for tensor in obs])
            out = {key: [v.unsqueeze(0) for v in value]
                   for key, value in self.estimator(obs).items()}
        else:
            out = collate_fn([self.estimator(x) for x in obs])
        assert isinstance(out, dict)
        return out

    def review(self, batch, output):
        """
        :param batch: dict of lists
        :param output: output of the forward step
        :return:
        """
        losses = self.add_losses(batch, output)
        return dict(losses={'loss': losses[MaskLossKeys.MASK]},
                    scalars=losses,
                    audios=self.add_audios(batch, output),
                    images=self.add_images(batch, output)
                    )

    def add_images(self, batch, output):
        speech_mask = output[M_K.SPEECH_MASK_PRED][0]
        observation = batch[M_K.OBSERVATION_ABS][0]
        images = dict()
        images['speech_mask'] = masks_to_images(speech_mask)
        images['observed_stft'] = masks_to_images(
            torch.abs(observation) / torch.max(torch.abs(observation))
        )
        if M_K.NOISE_MASK_PRED in output:
            noise_mask = output[M_K.NOISE_MASK_PRED][0]
            images['noise_mask'] = masks_to_images(noise_mask)
        if batch is not None and M_K.SPEECH_MASK_TARGET in batch:
            images['speech_mask_target'] = masks_to_images(
                batch[M_K.SPEECH_MASK_TARGET][0])
            if M_K.NOISE_MASK_TARGET in batch:
                images['noise_mask_target'] = masks_to_images(
                    batch[M_K.NOISE_MASK_TARGET][0])
        return images

    # ToDo: add scalar review

    def add_audios(self, batch, output):
        audio_dict = {
            K.OBSERVATION: batch[K.OBSERVATION][0][0],
            K.SPEECH_IMAGE: batch[K.SPEECH_IMAGE][0][0]
        }
        return audio_dict

    def add_losses(self, batch, output):
        noise_loss = list()
        speech_loss = list()
        vad_loss = list()
        weighted_noise_loss = list()
        weighted_speech_loss = list()
        for idx, observation_stft in enumerate(batch[M_K.OBSERVATION_STFT]):
            power_weights = np.abs(observation_stft) ** 2
            if M_K.SPEECH_MASK_TARGET in batch:
                speech_mask_target = batch[M_K.SPEECH_MASK_TARGET][idx]
            else:
                speech_mask_target = None
            if M_K.NOISE_MASK_TARGET in batch:
                noise_mask_target = batch[M_K.NOISE_MASK_TARGET][idx]
            else:
                noise_mask_target = None
            if M_K.NOISE_MASK_LOGITS in output:
                noise_mask_logits = output[M_K.NOISE_MASK_LOGITS][idx]
            else:
                noise_mask_logits = None

            if M_K.SPEECH_MASK_LOGITS in output:
                speech_mask_logits = output[M_K.SPEECH_MASK_LOGITS][idx]
            else:
                speech_mask_logits = None

            power_weights = batch[K.OBSERVATION][0].new(power_weights)

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
                    POOLING_FN_MAP[self.reduction](sample_loss))
                weighted_noise_loss.append(
                    weight_loss(sample_loss, power_weights))
            # Speech mask
            if speech_mask_target is not None\
                    and speech_mask_logits is not None:
                sample_loss = get_loss(speech_mask_target, speech_mask_logits)
                speech_loss.append(
                    POOLING_FN_MAP[self.reduction](sample_loss))
                weighted_speech_loss.append(
                    weight_loss(sample_loss, power_weights))
            # VAD
            if M_K.VAD in batch and M_K.VAD in output:
                vad_target = batch[M_K.VAD][idx]
                vad_logits = output[M_K.VAD_LOGITS][idx]
                vad_loss.append(get_loss(
                    vad_target, vad_logits))

        loss = []
        weighted_loss = []
        losses = dict()

        if len(noise_loss) > 0:
            noise_loss = sum(loss)
            loss.append(noise_loss)
            losses[MaskLossKeys.NOISE_MASK] = noise_loss
            weighted_loss.append(sum(weighted_noise_loss))
        if len(speech_loss) > 0:
            speech_loss = sum(speech_loss)
            loss.append(speech_loss)
            losses[MaskLossKeys.SPEECH_MASK] = speech_loss
            weighted_loss.append(sum(weighted_speech_loss))
        if len(vad_loss) > 0:
            losses[MaskLossKeys.VAD] = sum(vad_loss)
        if len(loss) > 0:
            loss = sum(loss)
            weighted_loss = sum(weighted_loss)
            losses[MaskLossKeys.MASK] = loss
            losses[MaskLossKeys.WEIGHTED_MASK] = weighted_loss
        return losses


def masks_to_images(masks):
    """
    For more details of the output shape, see the tensorboardx docs.

    :param masks: Shape (frames, batch, features)
    :param format: Defines the shape of masks, normally 'tbf'.
    :return: Shape(batch, features, frames, 1)
    """
    images = torch.clamp(masks * 255, 0, 255)
    images = images.type(torch.ByteTensor)
    return images[0].numpy().transpose(1, 0)[::-1]
