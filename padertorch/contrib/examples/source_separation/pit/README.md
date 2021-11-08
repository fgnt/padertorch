Utterance-level PIT
================

This directory contains scripts to train and evaluate the basic utterance-level permutation
invariant training model (uPIT)
for source separation [1].


Training
--------

Prerequisites

 - Set `${STORAGE_ROOT}` to the location you want to store your experiment results
 - Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
 - Prepare the JSON(s) for your database. Each example must be sorted by `num_samples` as the model
   uses the `PackedSequence` of PyTorch 
     
```bash
$ python -m padertorch.contrib.examples.source_separation.pit.train with database_json=${path_to_your_jsons}
 ```

You can initialize an experiment directory with `python -m ...pit.train init with ...` and start it with: 

```bash
$ make train
```

Make sure to set `train_dataset` and `validation_dataset` according to the datasets available in the supplied 
database JSONs (they are set to use WSJ0-2mix by default).

Evaluation
----------

Start an evaluation with 

```bash
$ python -m padertorch.contrib.examples.source_separation.pit.evaluate with model_path=${path_to_the_model_dir} database_json=${path_to_the_json} "datasets=['your','datasets']"
```

If you want to speed up your evaluation, you can also call
```bash
$ mpiexec -np ${n_jobs} python padertorch.contrib.examples.source_separation.pit.evaluate with model_path=${path_to_the_model_dir} database_json=${path_to_the_json} "datasets=['your','datasets']"
```
to parallelize your evaluation over several CPU cores.

Important configuration values
------------------------------

 - `batch_size`: Set the batch size
 - `trainer.stop_trigger`: Set the number of iterations or epochs to perform (e.g, `trainer.stop_trigger=(100,'epoch')` for 100 epochs)
 

References
----------

  [1] Morten Kolbæk, Dong Yu, Zheng-Hua Tan, Jesper Jensen. „Multi-talker Speech Separation with Utterance-level 
  Permutation Invariant Training of Deep Recurrent Neural Networks“, March 18 2017.
