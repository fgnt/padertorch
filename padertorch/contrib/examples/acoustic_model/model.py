
"""

python -m cbj_lib.run <path to sim folder> -- python -m cbj.pytorch.run_am

"""

from dataclasses import dataclass

import numpy as np

import torch
import einops
import editdistance


import paderbox as pb
from paderbox.database.iterator import AlignmentReader

from paderbox.database.iterator import FilterException

from paderbox.database.chime import Chime4
import padertorch as pt


def get_blstm_stack(
        input_size: int,
        hidden_size: (tuple, list),
        output_size: int,
        bidirectional=True,
        batch_first=False,
        dropout=0.
):
    """

    >>> blstm = get_blstm_stack(2, [3], 3, dropout=0.5)
    >>> blstm
    LSTM(2, 3, num_layers=2, dropout=0.5, bidirectional=True)

    """
    assert batch_first is False, batch_first

    for size in hidden_size:
        if size != output_size:
            raise ValueError(
                f'{input_size}, {hidden_size}, {output_size}: '
                f'Only shared hidden and output dimension is supported.'
            )

    num_layers = len(hidden_size) + 1

    if dropout is None or dropout is False:
        dropout = 0

    return torch.nn.LSTM(
        input_size,
        hidden_size=output_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        batch_first=batch_first,
        dropout=dropout,
    )


def kaldi_stft(time_signal):
    # ToDo: window
    return pb.transform.stft(
        time_signal=time_signal,
        size=512,
        shift=160,
        axis=-1,
        window_length=400,
        pad=False,
        fading=False,

    )


def kaldi_istft(
        stft_signal,
):
    # ToDo: window
    return pb.transform.istft(
        stft_signal=stft_signal,
        size=512,
        shift=160,
        window_length=400,
        fading=False,
    )


def levenshtein_distance(
    hypothesis, truth
):
    assert isinstance(hypothesis, (tuple, list)), (type(hypothesis), hypothesis)
    assert isinstance(truth, (tuple, list)), (type(truth), truth)
    return editdistance.eval(hypothesis, truth)


class AcousticExperiment(pt.Model):
    """
    >>> pb.notebook.pprint(AcousticExperiment.get_config())
    {'factory': 'model.AcousticExperiment',
     'db': 'Chime4',
     'egs_path': '/net/vol/jenkins/kaldi/2018-01-10_15-43-29_a0b71317df1035bd3c6fa49a2b6bb33c801b56ac/egs/chime3/s5',
     'input_feature': {'type': 'mfcc',
      'kwargs': {'num_mel_bins': 40,
       'num_ceps': 40,
       'low_freq': 20,
       'high_freq': -400,
       'delta_order': 2}},
     'blstm_input_size': 120,
     'blstm_hidden_size': [150, 150],
     'blstm_output_size': 150,
     'blstm_bidirectional': True,
     'blstm_dropout': 0.3,
     'dense_input_size': 300,
     'dense_hidden_size': [500, 500],
     'dense_output_size': 1983,
     'dense_activation': 'relu',
     'dense_dropout': 0.3}

    """

    def __init__(
            self,
            db='Chime4',
            egs_path='/net/vol/jenkins/kaldi/2018-01-10_15-43-29_a0b71317df1035bd3c6fa49a2b6bb33c801b56ac/egs/chime3/s5',
            input_feature={
                'type': 'mfcc',
                'kwargs': {
                    # check that these parameters are the mfcc_high_res
                    'num_mel_bins': 40,
                    'num_ceps': 40,
                    'low_freq': 20,
                    'high_freq': -400,
                    'delta_order': 2,
                }
            },
            blstm_input_size=40 * 3,
            blstm_hidden_size=(256, 256),
            blstm_output_size=256,
            blstm_bidirectional=True,
            blstm_dropout=0.3,
            dense_input_size=256 * 2,  # 2 times blstm_output_size
            dense_hidden_size=(512, 512),
            # ToDo: may be read dense_ouput_size from kaldi if db has to be specified anyway, should be None or not specified as default
            dense_output_size=1983,  # Chime4 specific
            dense_activation='relu',
            dense_dropout=0.3,
    ):
        super().__init__()
        if db == 'Chime4':
            self.db = Chime4(
                egs_path=egs_path
            )
        else:
            raise NotImplementedError(db)
        self.input_feature = input_feature
        self.blstm = get_blstm_stack(
            input_size=blstm_input_size,
            hidden_size=blstm_hidden_size,
            output_size=blstm_output_size,
            bidirectional=blstm_bidirectional,
            batch_first=False,
            dropout=blstm_dropout
        )
        self.dense = pt.modules.fully_connected_stack(
            input_size=dense_input_size,
            hidden_size=dense_hidden_size,
            output_size=dense_output_size,
            activation=dense_activation,
            dropout=dense_dropout,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_iterable(self, dataset):
        if isinstance(self.db, Chime4):
            it = self.db.get_iterator_by_names(dataset)
            it = it.map(AlignmentReader(
                alignments=self.db.state_alignment,
                example_id_map_fn=self.db.example_id_map_fn
            ))
            return it
        else:
            raise TypeError(self.db)

    def transform(self, example):
        if self.input_feature['type'] == 'stft':
            Observation = np.abs(kaldi_stft([
                pb.io.load_audio(file)
                for file in example['audio_path']['observation']
            ]))
        elif self.input_feature['type'] in ['mfcc']:
            Observation = np.array(pb.kaldi.mfcc.compute_mfcc_feats(
                example['audio_path']['observation'],
                is_scp=True,
                stacked=True,
                **self.input_feature['kwargs'],
            ))
        else:
            raise ValueError(self.input_feature)

        Obs = einops.rearrange(
            Observation,
            'channel time freq -> time channel freq',
        ).astype(np.float32)

        example_id = example['example_id']

        if 'alignment' not in example:
            raise FilterException(f'{example_id} has not alignment')

        alignment = torch.tensor([example['alignment'].astype(np.int64)] * 6).t()

        return self.NNInput(
            Observation=Obs,
            alignment=alignment,
            kaldi_transcription=example['kaldi_transcription'].split(),
        )

    @dataclass
    class NNInput:
        Observation: torch.tensor
        alignment: torch.tensor
        kaldi_transcription: tuple

    def forward(self, example: NNInput):
        Observation = example.Observation

        hidden, _ = self.blstm(Observation)
        predict = self.dense(hidden)

        return self.NNOutput(
            predict=predict
        )

    @dataclass
    class NNOutput:
        predict: torch.tensor

    def review(self, example: NNInput, outputs: NNOutput):

        predict = outputs.predict.data.cpu().numpy()

        assert np.all(np.isfinite(predict)), predict

        ce = self.criterion(
            torch.einsum('tbf->tfb', outputs.predict), # why not use einops here?
            example.alignment,
        )

        with torch.no_grad():
            predicted = torch.argmax(outputs.predict.data, -1)
            tmp = (predicted == example.alignment)
            accuracy = tmp.sum().item() / tmp.numel()

        return {
            'losses': {
                'ce': ce,
            },
            'scalars': {
                'acc': accuracy,
            }
        }
