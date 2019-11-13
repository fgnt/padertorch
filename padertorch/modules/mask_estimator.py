import torch
from einops import rearrange
from torch.nn.utils.rnn import PackedSequence

import padertorch as pt
from padertorch.modules import fully_connected_stack
from padertorch.modules.normalization import Normalization
from padertorch.modules.recurrent import StatefulLSTM
from padertorch.ops import pack_sequence, pad_packed_sequence
from padertorch.ops.mappings import ACTIVATION_FN_MAP

__all__ = [
    "MaskKeys",
    "MaskEstimator",
]


class MaskKeys:
    SPEECH_MASK_PRED = 'speech_mask_prediction'
    SPEECH_TARGET = 'speech_target'
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
    def finalize_dogmatic_config(cls, config):
        num_features = config['num_features']
        config['recurrent'] = dict(
            factory=StatefulLSTM,
            input_size=num_features,
            hidden_size=256,
            bidirectional=True,
        )
        if config['recurrent']['bidirectional']:
            recurrent_output = 2 * config['recurrent']['hidden_size']
        else:
            recurrent_output = config['recurrent']['hidden_size']
        config['fully_connected'] = dict(
            factory=fully_connected_stack,
            input_size=recurrent_output,
            hidden_size=[1024] * 3,
            output_size=num_features * 2,
        )
        recu_in = config['recurrent']['input_size']
        assert recu_in == num_features, (recu_in, num_features)
        fc = config['fully_connected']['output_size']
        assert fc == 2 * num_features, (fc, num_features)
        config['normalization'] = dict(
            factory=Normalization,
            num_features=num_features,
            order='l2',
            statistics_axis=0,
        )
        n_in = config['normalization']['num_features']
        assert n_in == num_features, (n_in, num_features)

    def __init__(
            self,
            fully_connected,
            normalization: Normalization,
            recurrent: StatefulLSTM,
            num_features: int = 513,
            input_dropout: float = 0.5,
            use_log: bool = False,
            use_powerspectrum: bool = False,
            separate_masks: bool = True,
            output_activation: str = 'sigmoid',
            fix_states: bool = False,
            vad: bool = False,
    ):
        super().__init__()
        self.fully_connected = fully_connected
        self.normalization = normalization
        self.recurrent = recurrent
        # ToDo implement log and powerspectrum
        if use_log or use_powerspectrum:
            raise NotImplementedError
        self.num_features = num_features
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.use_log = use_log
        self.use_powerspectrum = use_powerspectrum
        self.separate_masks = separate_masks
        self.output_activation = output_activation
        self.fix_states = fix_states
        self.vad = vad
        if vad:
            self.linear_vad = torch.nn.Linear(2*num_features, 1)

    def forward(self, x):
        """
        :param x: list of tensors of shape(C T F)
        :return:
        """
        num_channels = x[0].shape[0]
        h = [obs_single_channel for obs in x for obs_single_channel in obs]
        h = pack_sequence(h)
        if self.normalization:
            h = PackedSequence(self.normalization(h.data), h.batch_sizes) # only works with torch 1.0 and higher
        h = PackedSequence(self.input_dropout(h.data), h.batch_sizes)

        if not self.fix_states:
            del self.recurrent.states
        h = self.recurrent(h)
        h = PackedSequence(self.fully_connected(h.data), h.batch_sizes)
        out = pad_packed_sequence(h, batch_first=True)[0]
        out = rearrange(out, '(c b) t f -> b c t f', c=num_channels)
        target_logits = out[..., :self.num_features]
        target_mask = ACTIVATION_FN_MAP[self.output_activation]()(target_logits)
        out_dict = {
            M_K.SPEECH_MASK_PRED: target_mask,
            M_K.SPEECH_MASK_LOGITS: target_logits,
        }
        if self.separate_masks:
            noise_logits = out[..., self.num_features:]
            noise_mask = ACTIVATION_FN_MAP[self.output_activation]()(noise_logits)
            out_dict.update({
                M_K.NOISE_MASK_PRED: noise_mask,
                M_K.NOISE_MASK_LOGITS: noise_logits
            })
        if self.vad:
            vad_logits = self.linear_vad(out)
            vad = ACTIVATION_FN_MAP[self.output_activation]()(vad_logits)
            out_dict.update({
                M_K.VAD_LOGITS: vad_logits,
                M_K.VAD: vad
            })
        return out_dict

