"""
Example call:

python -m padertorch.contrib.examples.audio_synthesis.wavenet.evaluate with exp_dir=/path/to/exp_dir
"""
from pathlib import Path

import numpy as np
import torch
from dlp_mpi import COMM, IS_MASTER, MASTER, split_managed
from lazy_dataset.database import JsonDatabase
from paderbox.io import load_json, dump_json
from paderbox.io.new_subdir import get_new_subdir
from padertorch import Model
from padertorch.contrib.examples.audio_synthesis.wavenet.data import \
    prepare_dataset
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver
from scipy.io import wavfile

ex = Experiment('wavenet-eval')


@ex.config
def config():
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the exp_dir on the command line.'
    storage_dir = get_new_subdir(
        Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
    )
    database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    test_set = 'test_clean'
    max_examples = None
    device = 0
    ex.observers.append(FileStorageObserver.create(storage_dir))


@ex.automain
def main(_run, exp_dir, storage_dir, database_json, test_set, max_examples, device):
    if IS_MASTER:
        commands.print_config(_run)

    exp_dir = Path(exp_dir)
    storage_dir = Path(storage_dir)
    audio_dir = storage_dir / 'audio'
    audio_dir.mkdir(parents=True)

    config = load_json(exp_dir / 'config.json')

    model = Model.from_storage_dir(exp_dir, consider_mpi=True)
    model.to(device)
    model.eval()

    db = JsonDatabase(database_json)
    test_data = db.get_dataset(test_set)
    if max_examples is not None:
        test_data = test_data.shuffle(rng=np.random.RandomState(0))[:max_examples]
    test_data = prepare_dataset(
        test_data, audio_reader=config['audio_reader'], stft=config['stft'],
        max_length=None, batch_size=1, shuffle=True
    )
    squared_err = list()
    with torch.no_grad():
        for example in split_managed(
            test_data, is_indexable=False,
            progress_bar=True, allow_single_worker=True
        ):
            example = model.example_to_device(example, device)
            target = example['audio_data'].squeeze(1)
            x = model.feature_extraction(example['stft'], example['seq_len'])
            x = model.wavenet.infer(
                x.squeeze(1),
                chunk_length=80_000,
                chunk_overlap=16_000,
            )
            assert target.shape == x.shape, (target.shape, x.shape)
            squared_err.extend([
                (ex_id, mse.cpu().detach().numpy(), x.shape[1])
                for ex_id, mse in zip(
                    example['example_id'], ((x-target)**2).sum(1)
                )
            ])

    squared_err_list = COMM.gather(squared_err, root=MASTER)

    if IS_MASTER:
        print(f'\nlen(squared_err_list): {len(squared_err_list)}')
        squared_err = []
        for i in range(len(squared_err_list)):
            squared_err.extend(squared_err_list[i])
        _, err, t = list(zip(*squared_err))
        print('rmse:', np.sqrt(np.sum(err)/np.sum(t)))
        rmse = sorted(
            [(ex_id, np.sqrt(err/t)) for ex_id, err, t in squared_err],
            key=lambda x: x[1]
        )
        dump_json(rmse, storage_dir / 'rmse.json', indent=4, sort_keys=False)
        ex_ids_ordered = [x[0] for x in rmse]
        test_data = db.get_dataset('test_clean').shuffle(
            rng=np.random.RandomState(0))[:max_examples].filter(
            lambda x: x['example_id'] in ex_ids_ordered[:10] + ex_ids_ordered[-10:],
            lazy=False
        )
        test_data = prepare_dataset(
            test_data, audio_reader=config['audio_reader'], stft=config['stft'],
            max_length=10., batch_size=1, shuffle=True
        )
        with torch.no_grad():
            for example in test_data:
                example = model.example_to_device(example, device)
                x = model.feature_extraction(example['stft'], example['seq_len'])
                x = model.wavenet.infer(
                    x.squeeze(1),
                    chunk_length=80_000,
                    chunk_overlap=16_000,
                )
                for i, audio in enumerate(x.cpu().detach().numpy()):
                    wavfile.write(
                        str(audio_dir / f'{example["example_id"][i]}.wav'),
                        model.sample_rate, audio
                    )
