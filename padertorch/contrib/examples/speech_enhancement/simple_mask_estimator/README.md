Simple Mask Estimator
=============

This directory contains scripts to train and evaluate a simple mask estimator
inspired by .

Results
-------

The simple mask estimator model achieves the following results on
the CHiME 3 simulated evaluation set:


data type         |  pesq         | stoi         |      sdr 
:------------------|--------------|--------------|--------------:
observed           |  1.07       |  0.672       |  -0.79 dB
masked             |  1.22       |  0.736       |  5.68 dB
beamformed         |  1.91       |  0.958       |   17.10 dB  

Masked and observed are evaluated on the first channel of the 6ch track.

Training
--------

A storage root must be set with `export STORAGE_ROOT=/path/to/your/storage`.
After installing `padertorch`, a training can for example be started with

```bash
$ STORAGE_ROOT=/path/to/your/storage; python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.train
```

Evaluation
----------

The evaluation requires `dlp_mpi` and `pb_bss` as additional dependencies.
`dlp_mpi` can be installed via `pip install dlp_mpi` and `pb_bss` is available at [github.com/fgnt/pb_bss](github.com/fgnt/pb_bss).
The evaluation can be started by

```bash
$ STORAGE_ROOT=/path/to/your/storage; mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.evaluate
```
It always evaluates the latest model in the specified STORAGE_ROOT

If you want to evaluate a specific checkpoint, specify the path as an
additional argument to the call.

```bash
$ STORAGE_ROOT=/path/to/your/storage; mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.speech_enhancement.simple_mask_estimator.evaluate /path/to/checkpoint
```
References
----------

  [1] J. Heymann and L. Drude and A. Chinaev and R. Haeb-Umbach.
    “BLSTM supported GEV beamformer front-end for the 3rd CHiME challenge”
     Proc. Worksh. Automat. Speech Recognition, Understanding, 2015
        https://www.researchgate.net/publication/304407561_BLSTM_supported_GEV_beamformer_front-end_for_the_3RD_CHiME_challenge

