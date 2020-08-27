import itertools

import functools
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import sh
from tqdm import tqdm
from appdirs import user_cache_dir

import paderbox as pb

from ..utils import check_audio_files_exist

"""
To create the dataset, all files from the wsj database have to be converted to
wav and stored into one folder. The script 'combine_wsj0.sh' can be used for
this after changing the database and output paths.
After this, download the matlab-script
(wget http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip)
and install voicebox (view readme inside the downloaded zip file for details).
In the matlab script, change the directories accordingly (wsj0root and output
paths) and execute the script. This should create some subfolders in the
specified output folders and a few GB of wav data. By executing this script,
you can create a JSON-file which can be loaded using a JsonCallbackFetcher.

The transcriptions are taken from the "clean word" section of the wsj dataset.
For each (original wsj) utterance id there is a transcription in the "orth"
section. For each merl utterance id there is an annotation entry that maps it
to the wsj utterance id of the shortest utterance in the mix.
"""

# Define some keys that are used in the JSON format
DATASETS = 'datasets'
SPEAKER_ID = 'speaker_id'
GENDER = 'gender'
NUM_SAMPLES = 'num_samples'
LOG_WEIGHTS = 'log_weights'
TRANSCRIPTION = 'transcription'
OBSERVATION = 'observation'
SPEECH_SOURCE = 'speech_source'
AUDIO_PATH = 'audio_path'


def download_normalize_transcript():
    """
    Downloads Kaldis "normalize_transcript.pl" for WSJ to the working directory
    """
    cache_dir = Path(user_cache_dir('padertorch'))
    normalize_transcript_path = cache_dir / 'normalize_transcript.pl'
    if not normalize_transcript_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        pb.utils.process_caller.run_process(
            f'wget -O {normalize_transcript_path} https://raw.githubusercontent.com/kaldi-asr/kaldi/master/egs/wsj/s5/local/normalize_transcript.pl'
        )
    return normalize_transcript_path


def normalize_transcription(transcriptions):
    """ Passes the dirty transcription dict to a Kaldi Perl script for cleanup.

    We use the original Perl file, to make sure, that the cleanup is done
    exactly as it is done by Kaldi.

    :param transcriptions: Dirty transcription dictionary
    :param wsj_root: Path to WSJ database

    :return result: Clean transcription dictionary
    """
    normalize_transcript = download_normalize_transcript()

    result = pb.utils.process_caller.run_process(
        f'perl {str(normalize_transcript)} \'<NOISE>\'',
        input='\n'.join(f'{k} {v}' for k, v in transcriptions.items())
    ).stdout

    result = dict([
        line.split(maxsplit=1) for line in result.strip().split('\n')
    ])
    return result


def get_transcriptions(wsj0_root: Path):
    # WSJ saves transcriptions in *.dot and *.ptx files.
    word = dict()

    dot_files = list(wsj0_root.rglob('*.dot'))
    ptx_files = list(wsj0_root.rglob('*.ptx'))
    ptx_files = [ptx_file for ptx_file in ptx_files if Path(
        str(ptx_file).replace('.ptx', '.dot')) not in dot_files]

    for file_path in dot_files + ptx_files:
        with open(file_path) as fid:
            matches = re.findall(r"^(.+)\s+\((\S+)\)$", fid.read(), flags=re.M)
        word.update({utt_id: trans for trans, utt_id in matches})

    return normalize_transcription(word)


def load_spk2gender(spk2gender_file):
    spk2gender = dict()
    with spk2gender_file.open() as f:
        for line in f:
            line = line.split()
            spk2gender.update(
                {
                    line[0]: 'male' if line[1].lower() == 'm' else 'female'
                }
            )
    return spk2gender


@functools.lru_cache(maxsize=None)
def audio_length(file):
    return pb.io.audioread.audio_length(file)


def get_dataset(transcriptions, scenario_path, spk2gender, scenario):
    def get_path(index, example_id):
        path = scenario_path / f's{index + 1}' / f'{example_id}.wav'

        # Try to support the old and new versions of the database
        if index == 1 and not path.exists():
            path = scenario_path / f's{index + 1}' / f'{example_id}_2.wav'

        assert path.exists(), (
            f'File for speaker {index} not found for example {example_id}! '
            f'Expected {path} (with or without potential _2 postfix)'
        )

        return str(path.resolve())

    def task(file):
        # Filename is composed like utt1_snr1_utt2_snr2[_utt3_snr3].wav
        example_id = file.stem
        split = example_id.split('_')
        wsj_utterance_ids = split[::2]
        speaker_sdr = split[1::2]

        # Build example from information gathered from the mix file
        example = {
            AUDIO_PATH: {
                SPEECH_SOURCE: [
                    get_path(i, example_id)
                    for i in range(len(wsj_utterance_ids))
                ],
                OBSERVATION: str(file),
            },
            LOG_WEIGHTS: list(map(float, speaker_sdr)),
            NUM_SAMPLES: audio_length(file),
            SPEAKER_ID: [utt_id[:3] for utt_id in wsj_utterance_ids],
        }

        # Gender is optional
        if spk2gender is not None:
            example[GENDER] = [spk2gender[spk] for spk in example[SPEAKER_ID]]

        # Transcriptions are optional
        if transcriptions is not None:
            example[TRANSCRIPTION] = [
                str(transcriptions[id_]) for id_ in wsj_utterance_ids
            ]

        return example_id, example

    with ThreadPoolExecutor() as pool:
        files = list((scenario_path / 'mix').glob('*.wav'))
        dataset = dict(tqdm(
            pool.map(task, files), desc=scenario, total=len(files)))

    return dataset


@click.command()
@click.option(
    '--database_path',
    type=click.Path(file_okay=False, exists=True),
    help='Path where the generated data of the WSJ0-MIX database is located.',
)
@click.option(
    '--json_path',
    type=click.Path(dir_okay=False, writable=True),
    default='wsj0_mix.json',
    help='Output path for the generated JSON file. If the '
         'file exists, it gets overwritten.',
)
@click.option(
    '--wsj0_root',
    type=click.Path(file_okay=False, exists=True),
    help='Path to the WSJ0 root that was used by the matlab script for data '
         'generation',
    default=None,
)
@click.option(
    '--spk2gender_path',
    type=click.Path(exists=True),
    default=None,
    help='Path to Kaldi spk2gender file'
)
@click.option(
    '--num_speakers',
    default=['2', '3'],
    type=click.Choice(['2', '3']),
    multiple=True,
)
@click.option(
    '--scenarios',
    default=['cv', 'tr', 'tt'],
    type=click.Choice(['cv', 'tr', 'tt']),
    multiple=True,
)
@click.option(
    '--sample_rate',
    default=['wav8k'],
    type=click.Choice(['wav8k', 'wav16k']),
    multiple=True,
)
@click.option(
    '--signal_length',
    default=['min'],
    type=click.Choice(['min', 'max']),
    multiple=True,
    help='Decides whether audio is truncated or not. If both are selected, '
         'both datasets are created and written to the same JSON file'
)
def main(database_path, json_path, wsj0_root,
         sample_rate, signal_length, spk2gender_path, num_speakers,
         scenarios):
    """
    Creates JSON file for the WSJ0-2mix and WSJ0-3mix databases for use with
    the examples. Expects the data to be generated with the following file
    structure (this is the default for the matlab mixture creation scripts):

       <num_speakers>num_speakers/<min/max>/<s1/s2[/s3]/mix>/<filename>.wav
    """
    database_path = Path(database_path)
    json_path = Path(json_path)

    assert json_path.suffix == '.json'

    num_speakers = list(map(int, num_speakers))

    if wsj0_root is not None:
        wsj0_root = Path(wsj0_root)
        print(f'Reading transcriptions from WSJ ({wsj0_root})')
        transcriptions = get_transcriptions(wsj0_root)
    else:
        transcriptions = None

    if spk2gender_path is not None:
        print('Reading spk2gender file')
        spk2gender = load_spk2gender(Path(spk2gender_path))
    else:
        spk2gender = None

    print('Processing database')
    database = {}
    for signal_length_, num_speakers_, subset, sample_rate_ in itertools.product(
            signal_length, num_speakers,
            scenarios, sample_rate
    ):
        scenario = f'mix_{num_speakers_}_spk_{signal_length_}_{subset}'

        scenario_path = (
                database_path / f'{num_speakers_}speakers' / sample_rate_ /
                signal_length_ / subset
        )

        database[scenario] = get_dataset(
            transcriptions, scenario_path, spk2gender, scenario
        )

    print('Check that all wav files in the json exist.')
    check_audio_files_exist(database, speedup='thread')

    print('Finished check.')
    pb.io.dump_json(
        {DATASETS: database}, json_path, create_path=True, indent=4
    )


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
