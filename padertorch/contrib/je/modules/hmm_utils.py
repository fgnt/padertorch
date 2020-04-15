import numpy as np
from hmmlearn import _hmmc
from hmmlearn.utils import log_normalize
from scipy.special import logsumexp
# from pathos.multiprocessing import ProcessPool as ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def batch_forward_backward(
        log_startprob, log_transmat, framelogprob, seq_len=None,
        state_sequence=None, initial_state=None, final_state=None,
        max_workers=8
):
    B, T, K = framelogprob.shape
    log_startprob = np.broadcast_to(log_startprob, (B, K))
    if seq_len is None:
        seq_len = B*[T]
    if state_sequence is None:
        state_sequence = B*[None]
    if initial_state is None:
        initial_state = B*[None]
    if final_state is None:
        final_state = B*[None]

    def fwd_bwd(args):
        log_startprob, framelogprob, seq_len, state_sequence, initial_state, final_state = args
        posteriors = np.zeros_like(framelogprob)
        framelogprob = framelogprob[:seq_len]
        posteriors[:seq_len], transitions = forward_backward(
            log_startprob, log_transmat, framelogprob,
            state_sequence=state_sequence,
            initial_state=initial_state, final_state=final_state
        )
        return posteriors, transitions

    with ThreadPoolExecutor(max_workers) as executor:
        posteriors, transitions = list(zip(*executor.map(
            fwd_bwd, zip(log_startprob, framelogprob, seq_len, state_sequence, initial_state, final_state)
        )))
    return np.stack(posteriors), np.stack(transitions)


def batch_viterbi(
        log_startprob, log_transmat, framelogprob, seq_len=None,
        state_sequence=None, initial_state=None, final_state=None, max_workers=8
):
    B, T, K = framelogprob.shape
    log_startprob = np.broadcast_to(log_startprob, (B, K))
    if seq_len is None:
        seq_len = B*[T]
    if state_sequence is None:
        state_sequence = B*[None]
    if initial_state is None:
        initial_state = B*[None]
    if final_state is None:
        final_state = B*[None]

    def vit(args):
        log_startprob, framelogprob, seq_len, state_sequence, initial_state, final_state = args
        state_alignment = np.zeros(T)
        framelogprob = framelogprob[:seq_len]
        state_alignment[:seq_len] = viterbi(
            log_startprob, log_transmat, framelogprob,
            state_sequence=state_sequence,
            initial_state=initial_state, final_state=final_state
        )
        return state_alignment

    with ThreadPoolExecutor(max_workers) as executor:
        state_alignments = list(executor.map(
            vit, zip(log_startprob, framelogprob, seq_len, state_sequence, initial_state, final_state)
        ))
    return np.stack(state_alignments)


def forward_backward(
        log_startprob, log_transmat, framelogprob,
        state_sequence=None, initial_state=None, final_state=None
):
    log_startprob = np.array(log_startprob)
    log_transmat = np.array(log_transmat)
    framelogprob = np.array(framelogprob)
    T, K = framelogprob.shape
    if state_sequence is not None:
        state_sequence = squeeze_sequence(state_sequence)
        log_startprob, log_transmat, framelogprob = prepare_obvserved_state_sequence(
            log_startprob, log_transmat, framelogprob, state_sequence
        )
    else:
        if initial_state is not None:
            log_startprob = set_initial_state(log_startprob, initial_state)
        if final_state is not None:
            framelogprob = set_final_state(framelogprob, final_state)

    posteriors, transitions = _forward_backward(
        log_startprob, log_transmat, framelogprob
    )

    if state_sequence is not None:
        posteriors, transitions = invert_observed_state_sequence(
            posteriors, transitions, state_sequence=state_sequence, n_states=K
        )

    return posteriors, transitions


def viterbi(
        log_startprob, log_transmat, framelogprob,
        state_sequence=None, initial_state=None, final_state=None
):
    log_startprob = np.array(log_startprob)
    log_transmat = np.array(log_transmat)
    framelogprob = np.array(framelogprob)
    if state_sequence is not None:
        state_sequence = squeeze_sequence(state_sequence)
        log_startprob, log_transmat, framelogprob = prepare_obvserved_state_sequence(
            log_startprob, log_transmat, framelogprob, state_sequence
        )
        return state_sequence[_viterbi(log_startprob, log_transmat, framelogprob)]
    else:
        if initial_state is not None:
            log_startprob = set_initial_state(log_startprob, initial_state)
        if final_state is not None:
            framelogprob = set_final_state(framelogprob, final_state)
    return _viterbi(log_startprob, log_transmat, framelogprob)


def prepare_obvserved_state_sequence(
        log_startprob, log_transmat, framelogprob, state_sequence=None
):
    """
    transforms transition matrix to a large left to right hmm when state
    sequence is observed.

    Args:
        log_startprob:
        log_transmat:
        framelogprob:
        state_sequence:

    Returns:

    """
    seqlen = len(state_sequence)
    log_startprob = log_startprob[state_sequence]
    log_mask = np.log(np.eye(seqlen) + np.eye(seqlen, k=1))
    log_transmat = log_transmat[state_sequence[:, None], state_sequence] + log_mask
    framelogprob = framelogprob[:, state_sequence]
    framelogprob[-1] = framelogprob[0] = -np.inf
    framelogprob[0, 0] = framelogprob[-1, -1] = 0.
    return log_startprob, log_transmat, framelogprob


def invert_observed_state_sequence(
        posteriors, transitions, state_sequence, n_states
):
    """
    computes posteriors and transitions for original hmm states

    Returns:

    """
    n_steps = posteriors.shape[0]
    posteriors_ = np.zeros((n_steps, n_states))
    transitions_ = np.zeros((n_states, n_states))
    np.add.at(posteriors_, (np.arange(n_steps)[:, None], state_sequence), posteriors)
    np.add.at(transitions_, (state_sequence[:, None], state_sequence), transitions)
    return posteriors_, transitions_


def squeeze_sequence(seq):
    """
    remove similar consecutive states

    Args:
        seq:

    Returns:

    """
    return np.array([seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]])


def set_initial_state(logstartprob, initial_state):
    logstartprob = -np.inf*np.ones_like(logstartprob)
    logstartprob[initial_state] = 0.
    return logstartprob


def set_final_state(framelogprob, final_state):
    framelogprob[-1] = -np.inf
    framelogprob[-1, final_state] = 0.
    return framelogprob


def _forward_backward(log_startprob, log_transmat, framelogprob):
    logprob, alpha = _do_forward_pass(
        log_startprob, log_transmat, framelogprob
    )
    beta = _do_backward_pass(log_startprob, log_transmat, framelogprob)
    posteriors = _compute_posteriors(alpha, beta)
    expected_transitions = _compute_expected_transitions(
        log_transmat, framelogprob, alpha, beta
    )
    return posteriors, expected_transitions


def _do_forward_pass(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    fwdlattice = np.zeros((n_samples, n_components))
    _hmmc._forward(
        n_samples, n_components, log_startprob, log_transmat, framelogprob,
        fwdlattice
    )
    return logsumexp(fwdlattice[-1]), fwdlattice


def _do_backward_pass(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    bwdlattice = np.zeros((n_samples, n_components))
    _hmmc._backward(
        n_samples, n_components, log_startprob, log_transmat, framelogprob,
        bwdlattice
    )
    return bwdlattice


def _compute_expected_transitions(
        log_transmat, framelogprob, fwdlattice, bwdlattice):
    n_samples, n_components = framelogprob.shape
    log_xi_sum = np.full((n_components, n_components), -np.inf)
    _hmmc._compute_log_xi_sum(
        n_samples, n_components,
        fwdlattice, log_transmat, bwdlattice, framelogprob, log_xi_sum
    )
    return np.exp(log_xi_sum)


def _compute_posteriors(fwdlattice, bwdlattice):
    # gamma is guaranteed to be correctly normalized by logprob at
    # all frames, unless we do approximate inference using pruning.
    # So, we will normalize each frame explicitly in case we
    # pruned too aggressively.
    log_gamma = fwdlattice + bwdlattice
    log_normalize(log_gamma, axis=1)

    return np.exp(log_gamma)


def _viterbi(log_startprob, log_transmat, framelogprob):
    n_samples, n_components = framelogprob.shape
    ali, logprob = _hmmc._viterbi(
        n_samples, n_components, log_startprob, log_transmat, framelogprob
    )
    return ali
