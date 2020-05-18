TasNet (Currently DPRNN-TasNet only)
=============

This directory contains scripts to train and evaluate a Dual-Path RNN (DPRNN)-TasNet model proposed in [1].
Currently, only the DPRNN-TasNet is provided, but the BLSTM-based TasNet [2] and Conv-TasNet [3] are planned.

Results
-------

The model can achieve up to 19dB SDR gain on the WSJ0-2mix database.

| trained with  | SI-SDRi  | SDRi  |
|---|---|---|
| default  | 16.4  | 16.7  |
| `with log_mse`  | 16.7  | 17.0  |

Training
--------

A storage root must be set with `export STORAGE=/path/to/your/storage`.
After installing `padercontrib`, a training can be started with

```bash
$ python -m padercontrib.pytorch.examples.dual_path_rnn.train
```

This creates a `Makefile` for easy re-running and evaluation. You can call `...train init` to just create the `Makefile` without starting the training run.
Please have a look at the the header of `train.py` for information on how to run on the PC2 computing cluster. 

Different Configurations
------------------------

Different loss functions can be selected by adjusting the loss weights with for example

```bash
$ python -m padercontrib.pytorch.examples.dual_path_rnn.train with trainer.loss_weights.log-mse=1 trainer.loss_weights.si-snr=0
```

There is a named config for simple access to `log-mse`:

```bash
$ python -m padercontrib.pytorch.examples.dual_path_rnn.train with log_mse
```

Available loss functions are: `log-mse`, `si-snr`.

The configuration that has the best performance in the paper (window size of 2) can be selected with the named config `with win2`.

Evaluation
----------

The evaluation can be initialized by using the `Makefile` that was created by the training script.
Go into the model directory and run

```bash
make evaluation
```

Then follow the instructions printed by the script.
If you want to evaluate the model on the PC2 computing cluster, have a look at the file header of `evaluate.py` for instructions on how to utilize mpi for parallelization.

References
----------

  [1] Luo, Yi, Zhuo Chen, and Takuya Yoshioka. “Dual-Path RNN: Efficient
        Long Sequence Modeling for Time-Domain Single-Channel Speech
        Separation.” ArXiv Preprint ArXiv:1910.06379, 2019.
        https://arxiv.org/pdf/1910.06379.pdf
  
  [2] Luo, Yi, and Nima Mesgarani. „TasNet: time-domain audio separation network for real-time, single-channel speech separation“, 1. November 2017. https://doi.org/10.1109/icassp.2018.8462116.
  
  [3] Luo, Yi, und Nima Mesgarani. „Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation“, 20. September 2018. https://doi.org/10.1109/taslp.2019.2915167.


