"""
Create JSONs for the RIR database and LibriSpeech database suitable for
the use with the distance estimator.

If only one pair of database path is specified, it is assumed that the
corresponding JSON already exists and therefore it will not be not created.

However, "vad_json_path" has to be specified if LibriSpeech should be used for
source signals, which describes a path to a JSON containing VAD information for
LibriSpeech.
If LibriSpeech should not be used and only the JSON for the RIR database should
be created, the suffix "with only_rirs" must be added to the call.
Otherwise, VAD information will be added to the LibriSpeech-JSON for the
distance estimator to work properly because LibriSpeech contains silence during
the recordings and without an utterance, a distance cannot be estimated.
In the case of another source signal database, such VAD information might not
be necessary.
If the LibriSpeech-JSON already exists, it gets updated.
This can be done by adding the suffix "with update_librispeech" to the call,
where also "libri_json_path" has to be specified.
Otherwise, "libri_path" must be specified to create a new JSON for LibriSpeech.

For both databases, if the database path is provided, but no explicit path for
the database-JSON, the JSON gets saved in the same file as the database.

May be called with:
python -m padertorch.contrib.examples.source_localization.
distance_estimator.create_jsons with rir_path=/PATH/TO/RIR-DATABASE
rir_json_path=/PATH/TO/STORE/RIR-JSON
libri_path=PATH/TO/LIBRISPEECH/DATABASE
libri_json_path=PATH/TO/STORE/LIBRI-JSON vad_json_path=PATH/TO/VAD-JSON
"""

from pathlib import Path

import numpy as np
from paderbox.io import load_json, dump_json
from paderbox.io.audioread import audio_length
from sacred import Experiment


ex = Experiment()


@ex.config
def config():
    libri_mode = "full"

    rir_path = None
    rir_json_path = None
    if rir_path is not None:
        assert Path(rir_path).is_dir(), \
            f'rir_path "{rir_path}" is not a directory'
    if rir_json_path is not None and rir_json_path != "":
        msg = 'You have to define the path where the RIR database is stored.'
        assert rir_path is not None, msg
        filename = Path(rir_json_path)
        msg = f'JSON file for RIR database must end with ".json" ' \
              f'and not "{filename.suffix}"'
        assert filename.suffix == '.json', msg

    libri_path = None
    libri_json_path = None
    vad_json_path = None
    if libri_path is not None:
        assert Path(libri_path).is_dir(), \
            f'libri_path "{libri_path}" is not a directory'
    if libri_json_path is not None and libri_json_path != "":
        filename = Path(libri_json_path)
        msg = f'JSON file for LibriSpeech database must end with ".json" ' \
              f'and not "{filename.suffix}"'
        assert filename.suffix == '.json', msg
        msg = 'You have to define the path where' \
              ' the VAD-information JSON is stored.'
        assert libri_json_path and vad_json_path is not None, msg
    if libri_mode == "update":
        msg = \
            'You have to define the path where the LibriSpeech JSON is stored.'
        assert libri_json_path is not None, msg
    elif libri_mode == "full":
        msg = 'You have to define the path where LibriSpeech is stored.'
        assert libri_path is not None, msg


@ex.named_config
def update_librispeech():
    libri_mode = "update"


@ex.named_config
def only_rirs():
    libri_mode = None
    rir_path = None
    assert rir_path is not None, 'You have to define the path where' \
                                 ' the RIR database is stored.'


def create_json_rir(database_path, json_path):
    database_path = Path(database_path)
    if json_path == "" or json_path is None:
        json_path = database_path / "rir_db.json"
    rir_root = database_path / 'rirs/'
    setups = load_json(Path(database_path) / 'setups.json')
    simulation_descriptions = \
        load_json(Path(database_path) / 'simulation_descriptions.json')

    def create_dataset(iterable):
        rir_dataset = dict()
        counter = 0
        for example_id, example in iterable:
            example_dict = dict()
            example_dict['room_dimensions'] = \
                setups[example_id]['environment']['room_dimensions']
            example_dict['sound_decay_time'] = \
                setups[example_id]['environment']['sound_decay_time']
            for source in example['src_diary']:
                source_position = \
                    setups[example_id]["source_position"][
                        source['source_position']]
                example_dict["source_position"] = source_position
                rir_path = rir_root / f'{example_id}/' \
                                      f'src_{source["source_position"]}/'
                for node in range(4):
                    node_dict = dict()
                    node_dict['node_position'] = \
                        setups[example_id]['node_position'][f'node_{node}']
                    node_dict['node_orientation'] = \
                        setups[example_id]['node_orientation'][f'node_{node}']
                    node_dict['rir'] = rir_path / f'node_{node}.wav'
                    node_dict["distance"] = \
                        np.sqrt(np.sum((np.array(source_position)
                                - np.array(node_dict['node_position'])) ** 2))
                    node_dict.update(example_dict)
                    rir_dataset[f'rir_{counter}'] = node_dict
                    counter += 1
        return rir_dataset

    scenario = simulation_descriptions["scenario_4"]
    scenario_list = list(scenario.items())
    datasets = dict()
    rir_db = dict()
    datasets["train"] = create_dataset(scenario_list[:80])
    datasets["dev"] = create_dataset(scenario_list[80:90])
    datasets["eval"] = create_dataset(scenario_list[90:])
    rir_db["datasets"] = datasets
    dump_json(rir_db, json_path, sort_keys=True)
    print(f"RIR-DB-JSON was created at '{json_path}'")


def get_vad_json(vad_json_path):
    vad_json_path = Path(vad_json_path)
    if not vad_json_path.suffix == '.json':
        vad_json_path = vad_json_path / "speech_activity_librispeech.json"
    assert vad_json_path.exists(), vad_json_path
    return load_json(vad_json_path)["train_clean_100"]


def create_json_librispeech(database_path, libri_json_path,
                            vad_json_path, wav=False):
    vad_json = get_vad_json(vad_json_path)
    temp_json = dict()
    if wav:
        identifier = ".wav"
    else:
        identifier = ".flac"
    for key in vad_json.keys():
        key_information = key.split("-")
        audio_file = Path(database_path) / "train-clean-100" / \
            key_information[0] / key_information[1] / (key+identifier)
        temp_json[key] = {"speaker_id": key_information[0],
                          "chapter_id": key_information[1],
                          "num_samples": audio_length(str(audio_file)),
                          "audio_path": {"observation": str(audio_file)},
                          "onset": vad_json[key]["onset"],
                          "offset": vad_json[key]["offset"]}
    librispeech_json = {"datasets": {"train_clean_100": temp_json}}
    if libri_json_path == "" or libri_json_path is None:
        libri_json_path = Path(database_path) / "librispeech.json"
    dump_json(librispeech_json, libri_json_path)
    print(f"Full LibriSpeech-JSON was created at '{libri_json_path}'")


def add_vad_information(libri_json_path, vad_json_path):
    vad_json = get_vad_json(vad_json_path)
    full_libri_json = load_json(libri_json_path)
    try:
        libri_json = full_libri_json["datasets"]["train_clean_100"]
    except KeyError:
        try:
            libri_json = full_libri_json["datasets"]["train-clean-100"]
        except KeyError:
            print("Inappropriate LibriSpeech-JSON naming scheme")
            return
    for key in vad_json.keys():
        libri_json[key]["onset"] = vad_json[key]["onset"]
        libri_json[key]["offset"] = vad_json[key]["offset"]
    dump_json(full_libri_json, libri_json_path)
    print(f"VAD information were added to existing LibriSpeech-JSON "
          f"at '{libri_json_path}'")


@ex.automain
def main(rir_path, rir_json_path, libri_path, libri_json_path,
         vad_json_path, libri_mode):
    if libri_mode == "full":
        create_json_librispeech(libri_path, libri_json_path, vad_json_path)
    elif libri_mode == "update":
        add_vad_information(libri_json_path, vad_json_path)
    if rir_path is not None:
        create_json_rir(rir_path, rir_json_path)
