TasNet (Currently DPRNN-TasNet only)
=============

This directory contains scripts to train and evaluate different TasNet models:
 - TasNet (BLSTM-based) [2] (TODO),
 - ConvTasNet [3] (TODO), and
 - Dual-Path RNN TasNet [1].

Results
-------

The model can achieve up to 19dB SDR gain on the WSJ0-2mix database, though this training needs a lot of time.
With the default parameters, the following numbers can be obtained:

| trained with  | SI-SDRi  | SDRi  |
|---|---|---|
| DPRNN default  | 16.4  | 16.7  |
| DPRNN `with log_mse`  | 16.7  | 17.0  |

Training
--------

A storage root must be set with `export STORAGE=/path/to/your/storage`.
After installing `padertorch`, a training can be started with

```bash
$ python -m padertorch.contrib.examples.tasnet.train with database_json="${PATH_TO_YOUR_DATABASE_JSON}"
```

This creates a `Makefile` for easy re-running and evaluation. You can call `...train init` to just create the `Makefile` without starting the training run.
Please have a look at the the header of `train.py` for information on how to run on the PC2 computing cluster. 

Different Configurations
------------------------

Different loss functions can be selected by adjusting the loss weights with for example

```bash
$ python -m padertorch.contrib.examples.tasnet.train with database_json="${PATH_TO_YOUR_DATABASE_JSON}" trainer.loss_weights.log-mse=1 trainer.loss_weights.si-sdr=0
```

There is a named config for simple access to `log-mse`:

```bash
$ python -m padertorch.contrib.examples.tasnet.train with database_json="${PATH_TO_YOUR_DATABASE_JSON}" log_mse
```

Available loss functions are: `log-mse`, `si-sdr` and `log1p-mse`.

The configuration that has the best performance in the paper (window size of 2) can be selected with the named config `with win2`.

Evaluation
----------

The evaluation requires `dlp_mpi` and `pb_bss` as additional dependencies.
`dlp_mpi` can be installed via `pip install dlp_mpi` and `pb_bss` is available at [github.com/fgnt/pb_bss](github.com/fgnt/pb_bss).
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


