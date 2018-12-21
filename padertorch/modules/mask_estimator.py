import padertorch as pt
import torch
from paderbox.database.keys import *
from padertorch.modules.normalization import Normalization
from padertorch.ops import pack_sequence, pad_packed_sequence
from padertorch.ops.mappings import ACTIVATION_FN_MAP
from padertorch.modules.dense import DenseStack
from padertorch.modules.recurrent import LSTM
from torch.nn.utils.rnn import PackedSequence

__all__ = [
    "MaskKeys",
    "OfflineMaskEstimator",
    "OnlineMaskEstimator",
]


class MaskKeys:
    SPEECH_MASK_PRED = 'speech_mask_prediction'
    NOISE_MASK_PRED = 'noise_mask_prediction'
    SPEECH_MASK_LOGITS = 'speech_mask_logits'
    NOISE_MASK_LOGITS = 'noise_mask_logits'
    SPEECH_MASK_TARGET = 'speech_mask_target'
    NOISE_MASK_TARGET = 'noise_mask_target'
    OBSERVATION_STFT = 'observation_stft'
    OBSERVATION_ABS = 'observation_abs'
    NUM_FRAMES = NUM_FRAMES
    MASK_ESTIMATOR_STATE = 'mask_estimator_state'
    SPEECH_PRED = 'speech_prediction'
    VAD = 'vad'
    VAD_LOGITS = 'vad_logits'


# ToDo: Add vad estimation?

M_K = MaskKeys


class MaskEstimator(pt.Module):
    @classmethod
    def get_signature(cls):
        default_dict = super().get_signature()
        default_dict['recurrent'] = {
            'kwargs': dict(input_size=513,
                           output_states=True),
            'cls': LSTM,
            LSTM: dict()
        }
        default_dict['dense'] = {
            'kwargs': dict(input_size=512),
            'cls': DenseStack,
            DenseStack: dict()
        }
        default_dict['normalization'] = {
            'cls': Normalization,
            'kwargs': dict(num_features=513,
                           order='l2',
                           statistics_axis=1)
        }
        return default_dict

    def __init__(self, dense, normalization, recurrent,
                 num_features: int = 513,
                 use_log: bool = False,
                 use_powerspectrum: bool = False,
                 separate_masks: bool = True,
                 output_activation: str = 'sigmoid'
                 ):
        super().__init__()
        self.dense = dense
        self.normalization = normalization
        self.recurrent = recurrent

        self.classifier = torch.nn.Linear(
            self.dense.opts.num_units[-1],
            num_features
        )
        if self.separate_masks:
            self.classifier_noise = torch.nn.Linear(
                self.dense.num_units[-1],
                num_features
            )
        # ToDo implement log and powerspectrum
        if not use_log or not use_powerspectrum:
            raise NotImplementedError
        self.use_log = use_log
        self.use_powerspectrum = use_powerspectrum
        self.separate_masks = separate_masks
        self.output_activation = output_activation

    def forward(self, x):
        '''
        :param x: Tensor of shape(C,T,F)
        :return:
        '''
        if not isinstance(x, PackedSequence):
            x = self.normalization(x)  # ToDo allow PackedSequence
            x = pack_sequence(x)
        h, states = self.recurrent(x)
        h = PackedSequence(self.dense(h.data), h.batch_sizes)
        target_logits = PackedSequence(self.classifier(h.data), h.batch_sizes)
        target_logits = pad_packed_sequence(target_logits)[0].permute(1, 0, 2)
        if self.opts.separate_masks:
            noise_logits = PackedSequence(self.classifier_noise(h.data),
                                          h.batch_sizes)
            noise_logits = pad_packed_sequence(noise_logits)[0].permute(1, 0,
                                                                        2)

            return {
                M_K.SPEECH_MASK_PRED: ACTIVATION_FN_MAP[
                    self.opts.output_activation](target_logits),
                M_K.SPEECH_MASK_LOGITS: target_logits,
                M_K.NOISE_MASK_PRED: ACTIVATION_FN_MAP[
                    self.opts.output_activation](noise_logits),
                M_K.NOISE_MASK_LOGITS: noise_logits,
                M_K.MASK_ESTIMATOR_STATE: states
            }
        else:
            return {
                M_K.SPEECH_MASK_PRED: ACTIVATION_FN_MAP[
                    self.opts.output_activation](target_logits),
                M_K.SPEECH_MASK_LOGITS: target_logits,
                M_K.MASK_ESTIMATOR_STATE: states
            }
