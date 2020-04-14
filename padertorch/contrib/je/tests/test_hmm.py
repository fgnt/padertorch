import numpy as np
import numpy.testing as tc

from padertorch.contrib.je.modules import hmm_utils


def test_viterbi():
    log_startprob = np.array([0., -20., -20., -20.])
    log_transmat = np.log(
        [[.5,.5,0,0],
         [1/3,1/3,1/3,0],
         [0,0,.5,.5],
         [1/3,0,1/3,1/3]]
    )
    framelogprob = np.array(
        [[0.,-9.,-9.,-9.],
         [-9.,0.,-9.,-9.],
         [-9.,-9.,0.,-9.],
         [-9.,-9.,-9.,0.]]
    )
    z = hmm_utils.viterbi(log_startprob, log_transmat, framelogprob)
    tc.assert_almost_equal(z, [0,1,2,3], decimal=6)
    z = hmm_utils.viterbi(log_startprob, log_transmat, framelogprob, state_sequence=[2, 3])
    tc.assert_almost_equal(z, [2,2,2,3], decimal=6)
    z = hmm_utils.batch_viterbi(
        log_startprob, log_transmat, framelogprob[None][[0, 0, 0]], seq_len=[4, 4, 3], state_sequence=[None, [2,3], [2,3]]
    )
    tc.assert_almost_equal(z[0], [0,1,2,3], decimal=6)
    tc.assert_almost_equal(z[1], [2,2,2,3], decimal=6)
    tc.assert_almost_equal(z[2], [2,2,3,0], decimal=6)
    framelogprob = np.array(
        [[0.,-9.,-9.,-9.],
         [-9.,-9.,-9.,0.],
         [-9.,-9.,-9.,0.],
         [-9.,-9.,-9.,0.]]
    )
    z = hmm_utils.viterbi(log_startprob, log_transmat, framelogprob)
    tc.assert_almost_equal(z, [0,1,2,3], decimal=6)


def test_forward_backward():
    K = 4
    log_startprob = np.zeros(K)
    log_transmat = np.log(
        [[.5,.5,0,0],
         [1/3,1/3,1/3,0],
         [0,0,.5,.5],
         [1/3,0,1/3,1/3]]
    )
    framelogprob = np.array(
        [[0.,-9.,-9.,-9.],
         [-9.,0.,-9.,-9.],
         [0.,0.,-9.,-9.],
         [0.,-9.,-9.,-9.],
         [0.,-9.,-9.,-9.],
         [0.,-9.,-9.,-9.]]
    )
    posterior, transitions = hmm_utils.forward_backward(
        log_startprob, log_transmat, framelogprob
    )
    # tc.assert_almost_equal(z, [0,1,2,3], decimal=6)
    posterior, transitions = hmm_utils.forward_backward(
        log_startprob, log_transmat, framelogprob, state_sequence=[2, 3]
    )
    # tc.assert_almost_equal(z, [0,1,2,3], decimal=6)
    posterior, transitions = hmm_utils.batch_forward_backward(
        log_startprob, log_transmat, framelogprob[None][[0, 0, 0]],
        state_sequence=[None, None, [2, 3]],
        initial_state=[None, 0, None], final_state=[None, 1, None]
    )
    # tc.assert_almost_equal(z, [0,1,2,3], decimal=6)
    a = 1
