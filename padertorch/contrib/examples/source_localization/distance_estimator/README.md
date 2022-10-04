Distance Estimation Using CRNNs
=============

This directory contains scripts to prepare the prerequisites, train and evaluate a distance estimator,
which makes use of CRNNs and follows the concepts proposed in [1].

Prerequisites
------------
_If an own database containing suitably annotated data with reverberated speech recordings can be provided
(cf. "Use of an own database" for details of the structure), the following prerequisites does not to be fulfilled._

By default, two databases are required for the distance estimator to work, 
First, the room impulse response (RIR) database which consists of different source and node positions and room dimensions is needed to describe the acoustic environment. 
The database used here originates from the [paderwasn](https://github.com/fgnt/paderwasn) acoustic sensor networks toolbox where it is used to generate data for synchronization experiments.
Second, the LibriSpeech database is needed for speech samples as source signals, which are subsequently reverberated
with RIRs from the RIR-database.
Of course another speech database for source signals can be used, but its support and functionality must be implemented by oneself.
Due to its free availability and handling, LibriSpeech is used by default in this project and the option for alternatives is no longer mentioned in what follows.
LibriSpeech contains silence during the recordings and without an utterance, a distance cannot be estimated.
Therefore, a JSON which contains Voice Activity Detection (VAD) information for LibriSpeech is needed in order for the distance estimator to work properly.
In the case of another speech source database, this information is invalid or no longer needed, if there is no silence in the recordings. 

A download script to download both databases and the VAD-JSON is provided as well as a script which creates the JSONs describing the databases.
They can be executed separately and the selection of passed arguments considers already existing components, like LibriSpeech.
For the individual execution, please refer to the docstrings in the scripts "download.py" and "create_jsons.py".
In order to ease the setup, both steps can be done with the provided Makefile, which is provided for the default setup with
LibriSpeech and the described RIR database and whose intended use is as follows:

Four rules are offered to cover all possible setup scenarios and can be selected by entering `make "desired_rule"` to the command line in the corresponding directory of the distance estimator: 
- `make complete` (default, when only `make` is called): both datasets are downloaded and both JSONs are created; `RIR_PATH` and `LIBRI_PATH` must be specified
- `make rir`: the RIR database is downloaded, its JSON is created and the existing LibriSpeech-JSON is updated with VAD information, the variables `RIR_PATH`, `LIBRI_JSON` and `VAD_JSON` must be specified. 
- `make librispeech_full`: The LibriSpeech database is downloaded and the JSON is created, the variable `LIBRI_PATH` must be specified
- `make librispeech_update`: The existing LibriSpeech-JSON gets updated with VAD information, the variables `LIBRI_JSON` and `VAD_JSON` must be specified

The variables `RIR_JSON` and `LIBRI_JSON` (unless LibriSpeech gets not updated) are optional if the JSON should be stored in a
specific path, otherwise the JSON is stored in the same path as the database and its files.
When LibriSpeech has to be downloaded anyway, the variable `VAD_JSON` describing the path where the JSON with the VAD information is stored, can be specified optionally.
In case of an existing LibriSpeech-JSON, the path where the JSON containing VAD information should be stored must be provided in any case.
 
All necessary variables for the desired make rule can be entered directly in the provided Makefile or 
instead of changing the variables in the Makefile before execution, they can be specified via the command line.
The following example would download the RIR database, create a JSON for it and add VAD information to the LibriSpeech-JSON,
where it is presumed that the LibriSpeech database already exists:
```bash
make rir RIR_PATH=/PATH/TO/RIR-DATABASE LIBRI_JSON=/PATH/TO/LIBRISPEECH-JSON VAD_JSON=/PATH/TO/VAD-JSON
```
Note that write permissions for the directory of the LibriSpeech-JSON must be available because the existing JSON gets modified and overwritten during the process.


Distance estimation
---------
By default, the distance between speaker and microphones is linearly quantized and the parameters of the neural net are optimized with respect to 
the mean absolute error (mae).

By default, the magnitude and phase of the STFT of each microphone signal are used as input feature for the training unless not differently specified.
However, in order to compare this to previous established input features like the diffuseness, other features can also be used as an input feature for the training:
the Inter-Channel Level Differences
(ILD),  the Inter-Channel Phase Differences (IPD), the phase information of both channels or the magnitude information
of one of the microphone channels. These features can be individually combined to compare the effects of each feature.

The distance is estimated for a pair of microphones. The considered pairs are specified by the variable `mic_pairs`, which is a list of tuples that describe the indexes of the microphones.
For the recordings used here, a circular hexagon consisting of 6 microphones was used as microphone arrangement, where the distance between opposite microphones amounts to 5 cm.
By default, the microphone pairs are chosen to represent these facing pairs to get an accurate value for the distance of the microphones.  
If the diffuseness is used as the input feature and `mic_pairs` is changed, an adjustment of `d_mic` is necessary to get correct results.

For a more detailed explanation, please take a look at [1].


Use of an own database
------
If an own database with real recordings should be utilized for the training, the JSON describing this database
has to comply to the following format:
```json

{
  "datasets": {
        
    "train": {
          ...
        },
        
    "dev": {
          "example_0": {
                "distance": 5.6812749622868,
                "node_position": [
                      1.9996483860787013,
                      1.456984505205563,
                      1.4
                ],
                "audio_path": "/PATH/TO/AUDIO/RECORDING",
                "source_position": [
                      6.178405833956299,
                      5.305993648987165,
                      1.4
                ],
                "offset": number of samples the speech ends before the recording ends,
                "onset": number of samples until the speech starts
          }
    },
        
    "eval": {
          ...
    }
  }
}
```
The key "audio_path" describing the path to the recording can have an individual name.
Therefore, this name has to be passed by adding the suffix `with audio_key="keyname"` to the call in the command line instead of the path to the LibriSpeech-JSON.
Then the own database is used and LibriSpeech is no longer required.
Further, it is sufficient to provide only the distance or both the source and node position.
In addition to that, "offset" and "onset" describing VAD information are also optional. However, if provided, they might lead to better results
since the distance estimator is trained with segments that actually contain speech and not longer periods of silence at the beginning or ending of the training segment.\
Remember to change the minimum distance _d_min_ and the maximum distance _d_max_ in the config-function in train.py (or by adding the suffix `"with d_min=x.x d_max=x.x"` to the call) 
as they probably vary from the distances occurring in the RIR database and also to update the number of classes in a similar fashion.
In addition to that, mic_pairs might need to be changed to match to the conditions and number of microphones of the recording hardware
used for the own database as well as the distance between the microphones, which is used to calculate the diffuseness.


Training
--------

A storage root must be set with `export STORAGE_ROOT=/path/to/your/storage`, where the trained models will be stored.
After installing `padertorch`, a training can for example be started with

```bash
python -m padertorch.contrib.examples.source_localization.distance_estimator.train with rir_json=/PATH/TO/RIR-JSON, libri_json=/PATH/TO/LIBRISPEECH-JSON
```

The training script creates the new directory "source_localization" and
adds for each training run and each feature combination a new sequentially numbered subdirectory "distance_estimator_x". 

The training setup and environment configurations can be made by adding the corresponding parameter in the command
(cf. [Running experiments with sacred](https://github.com/fgnt/padertorch/blob/master/doc/sacred.md#defining-the-configuration)):

```bash
python -m padertorch.contrib.examples.source_localization.distance_estimator.train with rir_json=/PATH/TO/RIR-JSON, libri_json=/PATH/TO/LIBRISPEECH-JSON
```

To utilize other features for the feature extraction, this can be done by adding for example `with feature=diffuseness`
to the command line instruction, whereby the diffuseness will be used as input feature.
As already mentioned, all supported features can be combined and may be called by adding all desired features as one space-separated string to the command:

```bash
python -m padertorch.contrib.examples.source_localization.distance_estimator.train with feature="stft diffuseness" rir_json=/PATH/TO/RIR-JSON, libri_json=/PATH/TO/LIBRISPEECH-JSON
```

Evaluation
----------

The evaluation requires `dlp_mpi` as additional dependency.
`dlp_mpi` can be installed via `pip install dlp_mpi`
The evaluation can be started by

```bash
mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.source_localization.distance_estimator.evaluate
```
by default, the latest modified model in the specified STORAGE_ROOT is evaluated without considering its training feature.
If you want to evaluate a model trained with a specific input feature, state the feature at the end of the call
and then the most recent model trained with the desired feature is evaluated.

```bash
mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.source_localization.distance_estimator.evaluate with feature=diffuseness
```
If you want to evaluate a specific or previous model, specify the path as an
additional argument to the call.

```bash
mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.source_localization.distance_estimator.evaluate with storage_dir=/PATH/TO/distance_estimator_{feature}_{id}
```

By default, the checkpoint with the best mae is evaluated.
If you want to evaluate a specific checkpoint, specify the name of the desired checkpoint as an
additional argument to the call.
This can be combined with the other methods described to select exactly what should be evaluated.

```bash
mpiexec -n $(nproc --all) python -m padertorch.contrib.examples.source_localization.distance_estimator.evaluate with checkpoint_name=desired_checkpoint_name.pth
```

A batch mode for the evaluation is supported, where multiple evaluation examples are combined to one batch in the evaluation iterator.
However, the batch size is preset as 1 and can be individually modified by adding the suffix `"with batch_size=x"` to the call.
This might influence the execution speed of the script and may vary on different systems, so some experiments
on the own system with an increased batch size could possibly accelerate the execution.  

Results
-------
The distance estimator model achieves the following results on acoustic source signals from the LibriSpeech database,
which were modified with RIRs to create simulated, reverberated microphone signals.
By reverberating the speech samples with these RIRs, changing speaker locations, distances and constellations with different rooms are part of the training material.
Details regarding the data modifications based on similar databases as the ones used here can be found in [1].
Note that these results can vary with the used segment length and that they can differ from the ones in the paper since other databases with other distances are used:
In the paper, a special simulated dataset with a minimum distance of 0.3 m and a maximum distance of 5 m, which was reverberated with speech samples from the Timit databases, was used. 
In this version of the distance estimator, the LibriSpeech for source signals and the RIR database for their reverberation are used
with a different number of examples (train/validation/evaluation size: 76800/9600/9600 instead of 100000/1000/10000 in the paper)
Hence, the minimum distance is here 0.5 m and the maximum distance is 7.9 meters.
The root mean squared error as another metric is used to illustrate the effect of large errors caused by an estimated
class which varies a lot from the original one.  

input feature      | mae / m | rmse / m | segment length / ms  | accuracy / % | pseudo accuracy / % |
:-----------------:|:-------:|:--------:|:--------------------:|:------------:| :------------------:|
Diffuseness        | 0.079   |   0.279  |       1000           |      83.4    |          93.9       |
ILD                | 0.093   |   0.343  |       1000           |      84.2    |          93.4      |
STFT               | 0.058   |   0.251  |       1000           |      88.3    |          97.1       |

The pseudo accuracy takes also confusions with direct neighbour classes into account (the next higher as well as the next lower distance class),
since this only evokes a small distance error and is a result of the unavoidable quantization error.


References
----------

  [1] Tobias Gburrek and Joerg Schmalenstroeer and Reinhold Haeb-Umbach,
      "On Source-Microphone Distance Estimation Using Convolutional Recurrent Neural Networks",
      Speech Communication; 14th ITG Conference, 29 September 2021 - 01 October 2021,
      [https://ieeexplore.ieee.org/document/9657505](https://ieeexplore.ieee.org/document/9657505)
