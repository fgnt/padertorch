import numpy as np
import torch
from padertorch.contrib.je.modules.gmm import GMM
from padertorch.ops.losses import gaussian_kl_divergence
from torch import nn

from padertorch.contrib.je.modules.hmm_utils import batch_forward_backward, batch_viterbi, squeeze_sequence
from padertorch.utils import to_list


class HMM(GMM):
    """
    >>> hmm = HMM(2, 3, 2)
    >>> torch.exp(hmm.log_transition_mat)
    """
    def __init__(
            self, feature_size, num_units, states_per_unit=3,
            covariance_type='full', locs_init_std=1.0, scale_init=1.0,
            initial_state=False, final_state=False, viterbi_training=False
    ):
        self.num_units = num_units
        super().__init__(
            feature_size, num_units * states_per_unit,
            covariance_type=covariance_type,
            locs_init_std=locs_init_std, scale_init=scale_init
        )
        self.states_per_unit = states_per_unit
        weights_mask = np.zeros(num_units * states_per_unit)
        weights_mask[::states_per_unit] = 1
        self.log_weights_mask = nn.Parameter(
            torch.Tensor(np.log(weights_mask)), requires_grad=False
        )
        within_hmm_transition_mask = (np.eye(self.num_classes) + np.eye(self.num_classes, k=1))
        last_states = np.arange(self.states_per_unit - 1, self.num_classes - 1, self.states_per_unit)
        within_hmm_transition_mask[last_states, last_states+1] = 0.
        self.within_hmm_transition_mask = nn.Parameter(
            torch.Tensor(within_hmm_transition_mask), requires_grad=False
        )
        between_hmm_transition_mask = np.zeros_like(within_hmm_transition_mask)
        between_hmm_transition_mask[self.states_per_unit - 1::self.states_per_unit] = 1
        between_hmm_transition_mask *= weights_mask
        self.between_hmm_transition_mask = nn.Parameter(
            torch.Tensor(between_hmm_transition_mask), requires_grad=False
        )
        self.initial_state = 0 if initial_state else None
        self.final_state = self.num_classes - 1 if final_state else None
        self.viterbi_training = viterbi_training

    @property
    def log_class_probs(self):
        log_weights = self.log_weights + self.log_weights_mask
        log_probs = torch.log_softmax(log_weights, dim=-1)
        return log_probs

    @property
    def log_transition_mat(self):
        log_transition_mat = (
            self.within_hmm_transition_mask
                * torch.ones_like(self.within_hmm_transition_mask) * np.log(0.5)
            + self.between_hmm_transition_mask
                * (torch.max(self.log_class_probs, -100 * torch.ones_like(self.log_class_probs)) + np.log(0.5))
            + torch.log(self.within_hmm_transition_mask + self.between_hmm_transition_mask)
        )
        return log_transition_mat

    def forward(
            self, qz, seq_len=None, unit_sequence=None,
            no_onset=False, no_offset=False
    ):
        log_rho = -kl_divergence(qz, self.gaussians)

        no_onset = to_list(no_onset, log_rho.shape[0])
        no_offset = to_list(no_offset, log_rho.shape[0])
        log_startprob = np.array([
            -np.ones(self.num_classes)*np.log(self.num_classes) if non
            else self.log_class_probs.detach().cpu().numpy().astype(np.float)
            for non in no_onset
        ])
        log_transmat = self.log_transition_mat.detach().cpu().numpy().astype(np.float)
        framelogprob = log_rho.detach().cpu().numpy().astype(np.float)

        initial_state = [None if non else self.initial_state for non in no_onset]
        final_state = [None if noff else self.final_state for noff in no_offset]
        for i, noff in enumerate(no_offset):
            if not noff and self.final_state is None:
                mask = np.zeros(self.num_classes)
                mask[self.states_per_unit - 1::self.states_per_unit] = 1.
                framelogprob[i, -1] += np.log(mask)

        if unit_sequence is not None:
            states_per_unit = self.hmm.states_per_unit
            state_sequence = [
                state_sequence_from_unit_sequence(seq, states_per_unit)
                for seq in unit_sequence
            ]
        else:
            state_sequence = None

        if not self.training or self.viterbi_training:
            b, t, k = framelogprob.shape
            state_alignment = batch_viterbi(
                log_startprob, log_transmat, framelogprob, seq_len=seq_len,
                state_sequence=state_sequence,
                initial_state=initial_state, final_state=final_state
            )
            state_alignment = state_alignment.astype(np.int)
            class_posteriors = np.zeros_like(framelogprob)
            state_transitions = np.zeros((b, k, k))
            class_posteriors[np.arange(b)[:, None], np.arange(t), state_alignment] = 1.
            src_idx = state_alignment[:, :-1]
            dst_idx = state_alignment[:, 1:]
            np.add.at(state_transitions, (np.arange(b)[:, None], src_idx, dst_idx), 1)
        else:
            class_posteriors, state_transitions = batch_forward_backward(
                log_startprob, log_transmat, framelogprob, seq_len=seq_len,
                state_sequence=state_sequence,
                initial_state=initial_state, final_state=final_state
            )

        return (
            torch.Tensor(class_posteriors).to(log_rho.device),
            torch.Tensor(state_transitions).to(log_rho.device),
            log_rho
        )


def state_sequence_from_unit_sequence(unit_sequence, states_per_unit):
    return [
        unit * states_per_unit + i
        for unit in squeeze_sequence(unit_sequence)
        for i in range(states_per_unit)
    ]
