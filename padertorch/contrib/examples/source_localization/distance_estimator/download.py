"""
Download the room impulse response (RIR) database, a JSON file containing voice
activity information (VAD) for LibriSpeech as well as LibriSpeech itself,
if it is required.

Both database storage paths can either be a path to the data files or a path
for where to download the files, depending on the needs and prerequisites.
In the former case, it is checked whether the existing data files meet the
requirements or have to be downloaded.

If only one path is specified, only the corresponding database is downloaded.

The JSON for the VAD information regarding the necessary LibriSpeech datasets
train_clean_100, dev-clean and test-clean has to be downloaded in any case
if LibriSpeech should be used, so its path must then be specified.
The path for this file is assumed to be the same as for the LibriSpeech
database, unless otherwise stated, which can be done by adding the suffix
"with vad_json_path=/PATH/TO/VAD_JSON" to the call.

May be called with:
"python -m padertorch.contrib.examples.source_localization.distance_estimator.
download with rir_path=/PATH/TO/RIR_DB libri_path=/PATH/TO/LIBRISPEECH"

"""

from pathlib import Path

from sacred import Experiment

from paderbox.io.download import download_file, download_file_list


ex = Experiment()


@ex.config
def config():
    libri_path = None
    rir_path = None
    vad_json_path = None
    if vad_json_path is None and libri_path is not None:
        vad_json_path = Path(libri_path) / "speech_activity_librispeech.json"
        msg = "Please specify a path where to store the VAD information JSON"
        assert vad_json_path is not None, msg
    libri_urls = [
        'http://www.openslr.org/resources/12/train-clean-100.tar.gz',
        'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'https://www.openslr.org/resources/12/test-clean.tar.gz'
        ]
    vad_json_url = 'https://zenodo.org/record/7071619/files/' \
                   'speech_activity_librispeech.json?download=1'
    rir_url = 'https://zenodo.org/record/5679070/files/async_wasn.tar.gz'


def check_files(rir_path, libri_path):
    """"
    Check whether path files describe an existing database JSON or if the
    relevant files are available.
    Otherwise, the database has to be downloaded.

    Parameters:
        rir_path (str): Storage path for the RIR database
        libri_path (str): Storage path for LibriSpeech
    """
    download_list = list()
    iterator = list()
    if rir_path is not None:
        iterator.append(rir_path)
    if libri_path is not None:
        iterator.append(libri_path)
    for path in iterator:
        path_zip = path.split('/')[-1]
        file_name = path_zip.split('.')[-1]
        if file_name == "json" and Path(path).exists():
            break
        elif path is rir_path:
            requirements = ["rirs", "sro_trajectories", "setups.json",
                            "simulation_descriptions.json"]
            for item in requirements:
                if not Path(Path(rir_path) / item).exists():
                    download_list.append("rir")
                    break
        else:
            requirements = ["train-clean-100", "dev-clean", "test-clean"]
            for item in requirements:
                if not Path(Path(libri_path) / item).exists():
                    download_list.append("librispeech")
                    break
    return download_list


@ex.automain
def download(rir_path, libri_path, rir_url,
             libri_urls, vad_json_url, vad_json_path):
    if vad_json_path is not None:
        if not Path(vad_json_path).suffix == '.json':
            vad_json_path = \
                Path(vad_json_path) / "speech_activity_librispeech.json"
        download_file(vad_json_url, vad_json_path, exist_ok=True)
        print(f"Downloaded VAD-JSON to {vad_json_path}")
    if libri_path is None and rir_path is None:
        return
    for dataset in check_files(rir_path, libri_path):
        if dataset == "librispeech":
            for dataset_url in libri_urls:
                download_file_list([dataset_url], libri_path, exist_ok=True)
            print(f'Successfully downloaded "{dataset}" to "{libri_path}"')
        elif dataset == "rir":
            download_file_list([rir_url], rir_path, exist_ok=True)
            print(f'Successfully downloaded "{dataset}" to "{rir_path}"')
