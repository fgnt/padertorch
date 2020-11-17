# Audio Tagging

This example trains and evaluates an audio tagging system based on WALNet [1]
trained on AudioSet. A more sophisticated model for audio tagging and weakly
labeled sound event detection which is also based on padertorch can be found at 
https://github.com/fgnt/pb_sed.

## Training

The training script needs a JSON file that describes the structure of your
database in the following format:
```
{
    "datasets": {
        <dataset_A>: {
            <example_id_A0>: {
                "audio_path": </path/to/wav>,
                "audio_length": <length in seconds>,
                "events": <list of sound events in example clip>,
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
It is expected that it contains datasets "validate" and "eval" (In our case
validate is a small part of unbalanced train set).

To start the training, first define a path to where the trained models should
be saved:
```bash
export STORAGE_ROOT=<your desired storage root>; python -m padertorch.contrib.examples.sound_recognition.audio_tagging.train
```
Your trained models can be found in `$STORAGE_ROOT/audio_tagging/<timestamp>`.

Note that the data input pipeline only extracts STFTs while the log mel 
extraction and normalization is done in the model.

## Evaluation

The evaluation script loads the best checkpoint (by default the checkpoint with
the highest achieved mAP on the  validation set) and runs evaluation on the 
eval set.

To run an evaluation, provide the evaluation script with the path to your trained model:
```bash
python -m padertorch.contrib.examples.sound_recognition.audio_tagging.evaluate with exp_dir=</path/to/trainer/storage_dir>
```

Evaluation results can be found in `<exp_dir>/eval/<timestamp>`.
In the file `overall.json` metrics averaged over all events can be found for
the validation and eval sets. In the file `event_wise.json` you can find
metrics for each event separately sorted by AP performance on the eval set.
Further, there are files `fn.json` and `fp.json` in which the system's false
negative and false positive predictions are saved.


## Results

| Training set    | Decision threshold tuning | Test set   | mAP   | mAUC  | lwlrap | mF1   |
| :-----:         | :-----:                   | :-----:    | :---: | :---: | :---:  | :---: |
| balanced_train  | validate                  | validate   | 22.02 | 92.16 | 48.4   | 31.76 |
| balanced_train  | validate                  | eval       | 23.28 | 93.55 | 49.69  | 25.73 |

Above Table reports mean Average Precision (mAP), mean Area Under ROC Curve
(mAUC), label weighted label-ranking average precision (lwlrap) and mean
F1-score (mF1) in %. Here, "mean" refers to macro-averaging over the
event-wise metrics. While mAP, mAUC and lwlrap do not rely on decision
thresholds, the computation of F1 scores requires thresholds. Therefore, the
event-specific decision thresholds are tuned on the validation set to give best
F1 scores. The big gap (>6%) between mF1 performance on the validation set
and eval set can be explained due to bad generalization of the decision
thresholds.

[1] Shah, Ankit and Kumar, Anurag and Hauptmann, Alexander G and Raj, Bhiksha.
"A closer look at weak label learning for audio events",
arXiv preprint arXiv:1804.09288, 2018
