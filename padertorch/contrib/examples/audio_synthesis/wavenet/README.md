# WaveNet Vocoder

This example trains and evaluates a WaveNet vocoder synthesising waveforms
from log mel spectrograms. The WaveNet is trained on the LibriSpeech corpus.

## Training

The training script needs a JSON file that describes the structure of your
database in the following format:
```
{
    "datasets": {
        <dataset_A>: {
            <example_id_A0>: {
                "audio_path": {
                    "observation": <path/to/wav>
                },
                "num_samples": <num_samples>
            },
            <example_id_A1>: {
                ...
            },
            ...
        },
        <dataset_B>: {
            <example_id_B0>: {
                ...
            },
            ...
        },
        ...
    }
}
```

To start the training, first define a path to where the trained models should
be saved:
```bash
export STORAGE_ROOT=<your desired storage root>; python -m padertorch.contrib.examples.audio_synthesis.wavenet.train
```
Your trained models can be found in `$STORAGE_ROOT/wavenet/<timestamp>`.

Note that the data input pipeline only extracts STFTs while the log mel 
extraction and normalization is done in the model.

## Evaluation

The evaluation script loads the best checkpoint (lowest achieved loss on the 
validation set) and performs autoregressive waveform synthesis.
For test-time synthesis nv-wavenet needs to be installed.
Do note that nv-wavenet requires a GPU with Compute Capability 6.0 or later
(https://developer.nvidia.com/cuda-gpus), i.e., you can neither run the
evaluation on a CPU nor, e.g., on a GTX 980.
If nv-wavenet is not installed yet run
```bash
cd /path/to/padertorch/padertorch/modules/wavenet/nv_wavenet
```
Update the Makefile with the appropriate ARCH, e.g., ARCH=sm_70 for Compute Capability 7.0.
Then run
```bash                  
make                                                                                                                                                         
python build.py install
```

To run an evaluation, provide the evaluation script with the path to your trained model:
```bash
mpiexec -np $(nproc --all) python -m padertorch.contrib.examples.audio_synthesis.wavenet.evaluate with exp_dir=<path/to/trainer/storage_dir>
```
It requires [dlp_mpi](https://github.com/fgnt/dlp_mpi) to be installed.

Evaluation results can be found in `<exp_dir>/eval/<timestamp>`.
For each example the root mean squared error between the true waveform and the 
synthesised one is saved to a file `rmse.json`.
The 10 best and worst synthesised waveforms are saved in a sub directory `audio`.

If you want to run evaluation on only a few examples run
```bash
python -m padertorch.contrib.examples.audio_synthesis.wavenet.evaluate with exp_dir=<path/to/trainer/storage_dir> max_examples=10
```

## Results

| Training set                      | Test set   | RMSE  |
| :-----:                           | :-----:    | :---: |
| train_clean_100 + train_clean_360 | test_clean | 0.084 |
