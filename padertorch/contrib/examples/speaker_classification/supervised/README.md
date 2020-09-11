# Speaker Classification

This example performs a simple speaker classification on the *clean_100* and
*clean_360* datasets of the LibriSpeech corpus.

## Training
To start the training, first define a path to where the trained models should be saved:
```bash
export STORAGE_ROOT=<your desired storage root>; python -m padertorch.contrib.examples.speaker_classification.supervised.train with database_json=</path/to/json> dataset=<your_dataset>
```
Your trained models can be found in `$STORAGE_ROOT/speaker_clf`. During training,
only 80% of the dataset is used for training. 10% are left out for validation
and another 10% for evaluation.

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
                "speaker_id": <speaker-id>
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
If you train on LibriSpeech like we did, be aware that the speaker ID is defined
as `<speaker-id>-<chapter-id>` by LibriSpeech, where `<chapter-id>` is an
identifier for a book chapter.
Here, we perform a speaker identification across chapters so we omit the chapter
ID (the part of the speaker ID after the hyphen).
This is taken care of during the data preparation.
Generally, if the speaker ID contains one or more hyphens, the data preparation
will take the part before the **first** hyphen as the final speaker label for
classification.
If the speaker ID does not contain any hyphens, it will take the complete speaker
ID string as it is as speaker label.

## Evaluation

To run an evaluation, provide the evaluation script with the path to your
trained model:
```bash
mpiexec -np $(nproc --all) python -m padertorch.contrib.examples.speaker_classification.supervised.evaluate with model_path=<path/to/trained/model>
```
The evaluation script loads the best checkpoint (lowest achieved loss) and
performs a speaker classification on the evaluation data.
It requires [dlp_mpi](https://github.com/fgnt/dlp_mpi) to be installed.
For each misclassified example, symlinks to the example audio file and to an audio
example of the wrongly classified speaker are stored.

## Results

| Database | Dataset | Num. Speakers | Num. Eval Examples | Classification Accuracy |
| :------: | :-----: | :-----------: | :----------------: | :---------------------: |
| LibriSpeech | clean_100 | 251 | 2853 | 98.60% |
| LibriSpeech | clean_360 | 921 | 10401 | 94.72% |
