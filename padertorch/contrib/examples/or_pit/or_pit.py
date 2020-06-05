import torch
from torch.nn import functional as F
from einops import rearrange
from typing import Tuple, Optional

import padertorch as pt
from padertorch.contrib.examples.tasnet.tasnet import TasNet
from padertorch.summary import tbx_utils, review_dict


def one_and_rest_permutation_invariant_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
        fill_missing_with_zeros: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Computes the permutation invariant loss variant that is used in the
    One-And-Rest PIT model [1]. That is, the loss for the first estimated input
    is always computed agains a single target, and the loss for the second
    estimated input is computed against the sum of all remaining targets. This
    function returns the loss for the target assignment that yields the minimal
     loss.

    TODO: Make this function support a batch dimension and parallelize over
    this dimension

    .. note::

        This function does not support a batch dimension or sequence lengths
        (yet)! But this is not a problem when training on a chunk level (with
        constant lengths), like it is done for TasNet or OR-PIT.

    Args:
        inputs (2x...): The estimated inputs. This are always exactly two
            inputs: one for the single-target and one for the sum of the
            remaining targets.
        targets (Kx...): The targets. This can be any number of targets greater
            than 0.
        loss_fn: The callable loss function to compute for each permutation.
            This function should have the signature `loss_fn(inputs, targets)`
            and return a scalar `torch.Tensor`.
        fill_missing_with_zeros: This only as an impact if the number of passed
            targets is smaller than 2. If so, and `train_on_silence` is set to
            `True`, the missing targets are filled with 0s. If set to `False`,
            no loss will be computed for the inputs that would match a missing
            target.

    Returns:
        A tuple of an integer representing the index of the matched fist target
        and a scalar `torch.Tensor` containing the resulting loss value.

    References:
        [1] Naoya Takahashi and Sudarsanam Parthasaarathy and Nabarun Goswami
                and Yuki Mitsufuji. Recursive speech separation for unknown
                number of speakers
    """
    assert inputs.shape[0] == 2
    K = targets.shape[0]

    if K == 0:
        # No targets given
        if fill_missing_with_zeros:
            # Use 0s as targets
            loss = loss_fn(inputs, torch.zeros_like(inputs))
            perm = 0
        else:
            # There are no targets to compute a loss. Nothing to do
            return inputs.new_zeros(1), 0
    elif K == 1:
        if fill_missing_with_zeros:
            # If only one target is present, fill with 0. This can only work if
            # the used loss fn can handle it
            loss = loss_fn(inputs, torch.cat([
                targets, torch.zeros_like(targets)
            ], dim=0))
            perm = 0
        else:
            # Don't fill with 0s, only compute loss on the one remaining target
            loss = loss_fn(inputs[0], targets[0])
            perm = 0
    else:
        # Compute the loss for all possible targets
        losses = list()
        for i in range(K):
            # [1], eq. 3
            losses.append(
                loss_fn(inputs[0], targets[i]) +
                (1/(K-1)) * loss_fn(
                    inputs[1],
                    torch.sum(targets[[j for j in range(K) if i != j]], dim=0)
                )
            )

        # Find the minimum
        loss, perm = min(losses, key=lambda x: x[0])

    return loss, perm


class OneAndRestPIT(pt.Model):
    """
    One-And-Rest PIT model as proposed in [1]. This model recursively extracts
    speakers one at a time by using a Conv-TasNet as a separator.

    This implementation contains some additional things in comparison to [1]:
        - Different ways to handle the last iteration ("unroll-type"):
            1) "res-single": Iterate until the residual signal contains a
                                single speaker
            2) "res-silent": Iterate until the residual signal contains no
                                speech
            3) "est-silent": Iterate until the estimated (and residual) signal
                                contains no speech
        - Different choices for a stopping criterion for speaker counting
            ("stop-cond")
            1) Thresholding: Train the model to ouptput silcence when no
                        speaker is left and use a simple thresholding
            2) Flag: Compute a flag as an additional output and train it to
                        detect silence

    References:
        [1] Naoya Takahashi and Sudarsanam Parthasaarathy and Nabarun Goswami
            and Yuki Mitsufuji. Recursive speech separation for unknown number
            of speakers
    """

    @classmethod
    def finalize_dogmatic_config(cls, config):
        # Make sure that the config of the TasNet is consistent with our flag
        # output
        config['separator']['additional_out_size'] = config['flag_units']
        config['separator']['num_speakers'] = 2
        assert config['separator']['num_speakers'] == 2, (
            'The separator has to have two outputs for the OR-PIT!'
        )

    def __init__(
            self,
            separator: TasNet,
            finetune: bool = False,
            unroll_type: str = 'res-single',
            stop_condition: str = 'flag',
            threshold: float = 0.5,
            propagate_grad_between_iterations: bool = False,
            flag_reduction: str = 'mean',
            flag_units: int = 20,
    ) -> None:
        super().__init__()

        self.finetune = finetune
        self.unroll_type = unroll_type
        self.threshold = threshold
        self.propagate_grad_between_iterations = \
            propagate_grad_between_iterations
        self.flag_reduction = flag_reduction
        self.separator = separator
        self.flag_units = flag_units

        # Set up stopping condition. Make sure that a flag exists when it is
        # selected as a stopping condition
        if stop_condition == 'flag':
            assert flag_units > 0, (
                'Can\'t use flag stopping criterion if flag is disabled.'
            )

        try:
            self.stop_condition = {
                'threshold': self._stop_threshold,
                'flag': self._stop_flag,
                'none': lambda *x: False,
            }[stop_condition]
        except KeyError:
            raise ValueError(f'Unknown stopping condition: {stop_condition}')

        assert self.separator.num_speakers == 2, (
            'The separator has to have two outputs for the OR-PIT!'
        )
        self.separator = separator

        if flag_units > 0:
            # Initialize an NN for the flag. Since flag is a scalar, use 1 as
            # output dimension
            self.flag_nn = torch.nn.Linear(flag_units, 1)
        else:
            self.flag_nn = None

    def _compute_flag(self, flag_nn_output, out):
        if self.flag_reduction == 'mean':
            return torch.sigmoid(torch.mean(flag_nn_output, dim=(1, 2)))
        elif self.flag_reduction in ('res-weighted-mean', 'est-weighted-mean'):
            if self.flag_reduction == 'res-weighted-mean':
                weights = torch.mean(out['encoded_out'][:, 1, :] ** 2, dim=-1)
            else:
                weights = torch.mean(out['encoded_out'][:, 0, :] ** 2, dim=-1)

            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            return torch.sigmoid(
                torch.sum(flag_nn_output[:, :, 0] * weights, dim=1))
        elif self.flag_reduction == 'min':
            return torch.sigmoid(torch.min(flag_nn_output, dim=(1, 2)))
        elif self.flag_reduction == 'max':
            return torch.sigmoid(torch.max(flag_nn_output, dim=(1, 2)))
        else:
            raise ValueError(
                f'Unknown flag reduction type: {self.flag_reduction}'
            )

    def _forward_step(self, example):
        out = self.separator.forward(example)

        if self.flag_nn:
            flag_output = self.flag_nn(
                rearrange(out['additional_out'], 'b o t -> b t o'))
            out['pre_mean_flag'] = flag_output
            flag = self._compute_flag(flag_output, out)
            out['flag'] = flag

        return out

    def _forward(self, example, max_iterations=4, oracle_num_speakers=None):
        assert oracle_num_speakers is None or \
               oracle_num_speakers <= max_iterations

        outs = []
        residual_signal = pt.pad_sequence(example['y'], batch_first=True)
        B = residual_signal.shape[0]
        assert B == 1 or oracle_num_speakers is not None, (
            f'Counting (when oracle_num_speakers=None) is only supported for '
            f'a batch-size of 1. Otherwise handling of different numbers of '
            f'speakers in the same batch does not work!'
        )

        def _stop_oracle(out, k):
            if self.unroll_type == 'res-single':
                return k >= oracle_num_speakers - 2
            elif self.unroll_type == 'res-silent':
                return k >= oracle_num_speakers - 1
            elif self.unroll_type == 'est-silent':
                return k >= oracle_num_speakers
            else:
                raise ValueError(
                    f'Unknown unroll type: {self.unroll_type}')

        if oracle_num_speakers is not None:
            stop_condition = _stop_oracle
        else:
            stop_condition = self.stop_condition

        for k in range(max_iterations):
            # We want to do this in the beginning of the loop because we
            # might need to compute a loss on the residual signal after the
            # loop
            if not self.propagate_grad_between_iterations:
                residual_signal = residual_signal.detach()

            out = self._forward_step({
                'y': residual_signal,
                'num_samples': example['num_samples']
            })
            out.update(
                estimate=out['out'][:, 0],
                residual=out['out'][:, 1],
                encoded_estimate=out['encoded_out'][:, 0],
                encoded_residual=out['encoded_out'][:, 1],
            )
            outs.append(out)

            if stop_condition(out, k):
                break

            residual_signal = out['residual']

        # Combine the outputs according to the unrolling type
        estimates = [o['estimate'] for o in outs]

        if self.unroll_type == 'res-single':
            estimates.append(outs[-1]['residual'])
        elif self.unroll_type == 'res-silent':
            pass
        elif self.unroll_type == 'est-silent':
            estimates = estimates[:-1]
        else:
            raise ValueError(
                f'Unknown unroll type: {self.unroll_type}')

        if len(estimates) == 0:
            # Shape BxKxT
            estimates = torch.zeros((B, 0, *outs[0]['estimate'].shape[1:]))
        else:
            # Stack to shape BxKxT
            estimates = torch.stack(estimates, dim=1)

        return {
            'out': estimates,
            'outs': outs
        }

    def forward(self, example):
        assert example['num_speakers'][:-1] == example['num_speakers'][1:]

        if self.finetune:
            oracle_num_speakers = example['num_speakers'][0]
        else:
            # 0 forces the model to always do one iteration
            oracle_num_speakers = 0

        return self._forward(example, oracle_num_speakers=oracle_num_speakers)

    def _get_flag_target(self, current_iteration, num_speakers):
        if self.unroll_type == 'res-single':
            return current_iteration == num_speakers - 2
        elif self.unroll_type == 'res-silent':
            return current_iteration == num_speakers - 1
        elif self.unroll_type == 'est-silent':
            return current_iteration == num_speakers
        else:
            raise ValueError(f'Unknown unroll type: {self.unroll_type}')

    def review(self, inputs, outputs):
        scalars = {}

        s = inputs['s']
        sequence_lengths = inputs['num_samples']
        B = len(s)
        K = len(s[0])

        # Compute the permutation invariant loss over the targets and sum
        # of remaining targets
        reconstruction_loss = 0
        permutations = []
        for b, (targets, seq_len) in enumerate(
                zip(s, sequence_lengths)
        ):
            indices = list(range(K))
            _permutations = []
            for k in range(len(outputs['outs'])):
                l, perm = one_and_rest_permutation_invariant_loss(
                    outputs['outs'][k]['out'][b][:seq_len],
                    targets[:, :seq_len],
                    pt.log_mse_loss,
                    fill_missing_with_zeros=True
                )
                reconstruction_loss += l
                _permutations.append(indices[perm])
                del indices[perm]
                targets = targets[
                    [i for i in range(targets.shape[0]) if i != perm]]
            permutations.append(_permutations)

        reconstruction_loss = reconstruction_loss / B

        # Compute loss for flag
        if self.flag_units:
            flag_loss = 0
            for k, out in enumerate(outputs['outs']):
                flag = out['flag']
                target_flag = self._get_flag_target(k, K)

                if target_flag:
                    flag_target = torch.ones_like(flag)
                else:
                    flag_target = torch.zeros_like(flag)

                flag_loss += F.binary_cross_entropy(
                    flag, flag_target
                )

                scalars.update({
                    # Report one value of a batch of the flag to get a coarse
                    # idea about the actual flag values
                    f'flag_value/{target_flag}': flag[0],
                    f'flag_value/{target_flag}/{K}spk': flag[0]
                })

            scalars.update({
                'flag_loss': flag_loss, f'flag_loss/{K}spk': flag_loss,
            })

            loss = reconstruction_loss + flag_loss
        else:
            loss = reconstruction_loss

        # Report stuff
        scalars.update({
            'reconstruction_loss': reconstruction_loss,
            f'loss/{K}spk': loss,
            f'reconstruction_loss/{K}spk': reconstruction_loss
        })

        audios = {
            f'estimate/{K}spk': tbx_utils.audio(
                outputs['outs'][0]['estimate'][0],
                sampling_rate=8000),
            f'residual-estimate/{K}spk': tbx_utils.audio(
                outputs['outs'][0]['residual'][0],
                sampling_rate=8000)
        }

        return review_dict(
            scalars=scalars,
            audios=audios,
            loss=loss
        )

    def _stop_threshold(self, out, k):
        if self.unroll_type == 'res-silent':
            # Check residual signal for silence
            if torch.mean(out['residual'] ** 2) < self.threshold:
                return True
        elif self.unroll_type == 'est-silent':
            # Check estimated signal for silence
            if torch.mean(out['estimated'] ** 2) < self.threshold:
                return True

        return False

    def _stop_flag(self, out, k):
        return out['flag'] > self.threshold

    def decode(self, example: dict, max_iterations: int = 4,
               oracle_num_speakers: Optional[int] = None):
        return self._forward(example, max_iterations, oracle_num_speakers)
