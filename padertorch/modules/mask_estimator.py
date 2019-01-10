from torch.nn.utils.rnn import PackedSequence

import padertorch as pt
from padertorch.modules import fully_connected_stack
from padertorch.modules.normalization import Normalization
from padertorch.modules.recurrent import LSTM
from padertorch.ops import pack_sequence, pad_packed_sequence
from padertorch.ops.mappings import ACTIVATION_FN_MAP

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
        num_features = 513
        default_dict['recurrent'] = {
            'kwargs': dict(input_size=num_features),
            'cls': LSTM,
            LSTM: dict()
        }
        default_dict['fully_connected'] = {
            'kwargs': dict(input_size=512, hidden_size=[1024]*3,
                           output_size=num_features*2),
            'cls': fully_connected_stack,
            fully_connected_stack: dict()
        }
        default_dict['normalization'] = {
            'cls': Normalization,
            'kwargs': dict(num_features=num_features,
                           order='l2',
                           statistics_axis=1)
        }
        return default_dict

    @classmethod
    def get_config(
            cls,
            updates=None,
            out_config=None,
    ):
        super().get_config(updates=updates, out_config=out_config)
        num_features = out_config['kwargs']['num_features']
        recu_in = out_config['kwargs']['recurrent']['kwargs']['input_size']
        assert recu_in == num_features, (recu_in, num_features)
        fc = out_config['kwargs']['fully_connected']['kwargs']['output_size']
        assert fc == 2*num_features, (fc, num_features)
        n_in = out_config['kwargs']['normalization']['kwargs']['num_features']
        assert n_in == num_features, (n_in, num_features)
        return out_config

    def __init__(self, fully_connected, normalization, recurrent:LSTM,
                 num_features=513,
                 use_log: bool = False,
                 use_powerspectrum: bool = False,
                 separate_masks: bool = True,
                 output_activation: str = 'sigmoid',
                 fix_states: bool = False
                 ):
        super().__init__()
        self.fully_connected = fully_connected
        self.normalization = normalization
        self.recurrent = recurrent
        # ToDo implement log and powerspectrum
        if use_log or use_powerspectrum:
            raise NotImplementedError
        self.num_features = num_features
        self.use_log = use_log
        self.use_powerspectrum = use_powerspectrum
        self.separate_masks = separate_masks
        self.output_activation = output_activation
        self.fix_states = fix_states

    def forward(self, x):
        """
        :param x: Tensor of shape(C,T,F)
        :return:
        """
        if not isinstance(x, PackedSequence):
            # x = self.normalization(x)  # ToDo allow PackedSequence
            x = pack_sequence(x)
        if not self.fix_states:
            self.recurrent.reset_states(x)
        h = self.recurrent(x)
        h = PackedSequence(self.fully_connected(h.data), h.batch_sizes)
        out = pad_packed_sequence(h)[0].permute(1, 0, 2)
        target_logits = out[..., :self.num_features]
        if self.separate_masks:
            noise_logits = out[..., self.num_features:]
            return {
                M_K.SPEECH_MASK_PRED: ACTIVATION_FN_MAP[
                    self.output_activation]()(target_logits),
                M_K.SPEECH_MASK_LOGITS: target_logits,
                M_K.NOISE_MASK_PRED: ACTIVATION_FN_MAP[
                    self.output_activation]()(noise_logits),
                M_K.NOISE_MASK_LOGITS: noise_logits
            }
        else:
            return {
                M_K.SPEECH_MASK_PRED: ACTIVATION_FN_MAP[
                    self.output_activation]()(target_logits),
                M_K.SPEECH_MASK_LOGITS: target_logits
            }
