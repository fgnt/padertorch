One-and-Rest-PIT
================

This directory contains scripts to train and evaluate a One-and-Rest-PIT model [1].
This is a recursive separation model that uses a time-domain separation network at its core.
By default, it uses the DPRNN implementation from `padertorch.examples.tasnet` as a separator.

Training
--------

Prerequisites

 - Set `${STORAGE_ROOT}` to the location you want to store your experiment results
 - Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
 - Prepare the JSON(s) for different numbers of speakers. Each example must have the key `'speaker_id'` and a list as 
     value whose length corresponds to the number of speaker in the mixture

The training procedure of the OR-PIT consists of two steps: no fine-tuning and fine-tuning.     
The training for the first step can be run with:
     
```bash
$ python -m padertorch.contrib.examples.source_separation.or_pit.train with database_jsons=${path_to_your_jsons}
 ```

You can initialize an experiment directory with `python -m ...or_pit.train init with ...` and start it with: 

```bash
$ make train
```

The `database_jsons` can be a single file or a comma-separated list of files, if you want to supply multiple files.
Make sure to set `train_datasets` and `validation_datasets` according to the datasets available in the supplied 
database JSONs (they are set to use WSJ0-2mix and WSJ0-3mix by default).

The fine-tune experiment can be initialized with:

```bash
$ make fine-tune
```

It creates a new storage dir and uses the same configuration (including number of epochs, data, etc.) as the base training.

Evaluation
----------

Start an evaluation with 

```bash
$ python -m padertorch.contrib.examples.source_separation.or_pit.evaluate with model_path=${path_to_the_model_dir} database_json=${path_to_the_json} "datasets=['your','datasets']"
```

Enable audio exporting with `dump_audio=True`.

Important configuration values
------------------------------

 - `batch_size`: Set the batch size
 - `trainer.stop_trigger`: Set the number of iterations or epochs to perform (e.g, `trainer.stop_trigger=(100,'epoch')` for 100 epochs)
 - `trainer.model.finetune`: Enables fine-tuning
 - `trainer.model.stop_condition`: The criterion to use for stopping during evaluation. Can be `'flag'` or `'threshold'`.
 - `trainer.model.unroll_type`: Determines how many iterations to perform for a given number of speakers. Can be `'res-single'` (iterate until the residual output contains a single speaker), `'res-silent'` (iterate until the residual signal is silent) or `'est-silent'` (iterate until the estimated signal is silent)
 

References
----------

  [1] Takahashi, Naoya, Sudarsanam Parthasaarathy, Nabarun Goswami, and Yuki Mitsufuji. „Recursive speech 
        separation for unknown number of speakers“, 5. April 2019.
