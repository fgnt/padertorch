import functools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import einops


import paderbox as pb
import padertorch as pt

import pb_bss.distribution.cwmm
import pb_bss.permutation_alignment

from cbj.pytorch.feature_extractor import FeatureExtractor
from padertorch.contrib.examples.acoustic_model.model import get_blstm_stack


class AuxiliaryLoss:
    def __init__(
            self,
            stft_size,
            permutation_alignment=False,
            iterations=5,
    ):
        self.mm = pb_bss.distribution.CWMMTrainer()
        self.permutation_alignment = permutation_alignment
        self.stft_size = stft_size
        self.iterations = iterations

    def __call__(self, predict: torch.Tensor, Observation: np.ndarray):
        # predict.shape == D T F K

        predict = torch.mean(predict, dim=-4)
        # predict.shape == T F K

        predict_np = pt.utils.to_numpy(predict.detach()).transpose(1, 2, 0)
        # predict_np.shape == F K T
        # observation.shape == D T F

        mixture_model = self.mm.fit(
            Observation.T,
            initialization=predict_np,
            iterations=self.iterations,
        )

        pdf = mixture_model.complex_watson.log_pdf(
            pb_bss.distribution.complex_watson.normalize_observation(
                einops.rearrange(Observation, 'D T F -> F () T D'.lower())
            )
        )
        # pdf.shape == F K T

        # ToDo: perm solver for X iterations

        if self.permutation_alignment:
            pdf = pb_bss.permutation_alignment.DHTVPermutationAlignment.from_stft_size(
                stft_size=self.stft_size
            )(einops.rearrange(pdf, 'F K T -> K F T'.lower()))

            pdf = einops.rearrange(pdf, 'K F T -> T F K'.lower())
        else:
            pdf = einops.rearrange(pdf, 'F K T -> T F K'.lower())
        # pdf.shape == T F K

        # Normalize pdf -> smaller gradient
        pdf = pdf / (np.mean(pdf ** 2, axis=-2, keepdims=True))

        # pdf = pdf / 1000

        # mean produces a smaller gradient than sum

        if self.permutation_alignment:
            def aux_loss_fn(
                    predict,
                    pdf,
            ):
                return -torch.mean(predict * pdf)

            # mixture_model

            aux_loss = pt.loss.pit_loss(
                einops.rearrange(
                    predict,
                    'T F K -> T K F'.lower()
                ),
                predict.new_tensor(
                    einops.rearrange(
                        pdf.astype(np.float32),
                        'T F K -> T K F'.lower()
                    )
                ),
                loss_fn=aux_loss_fn
            )
        else:
            aux_loss = -torch.mean(predict * predict.new_tensor(pdf.astype(np.float32)))

        return aux_loss


class CWMMLikelihood:
    def __init__(self):
        pass

    # def __call__(self, *args, **kwargs):


class Model(pt.Model, pt.train.hooks.Hook):
    use_guide = True

    def pre_step(self, trainer):
        pass
        # if trainer.iteration >= 10000:
        #     self.use_guide = False

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """
        >>> from IPython.lib.pretty import pprint
        >>> pprint(Model.get_config())
        {'factory': 'model.Model',
         'blstm': {'factory': 'padertorch.contrib.examples.acoustic_model.model.get_blstm_stack',
          'input_size': 257,
          'hidden_size': [512],
          'output_size': 512,
          'bidirectional': True,
          'batch_first': False,
          'dropout': 0.3},
         'dense': {'factory': 'padertorch.modules.fully_connected.fully_connected_stack',
          'input_size': 1024,
          'hidden_size': [1024, 1024],
          'output_size': 514,
          'activation': 'relu',
          'dropout': 0.3},
         'feature_extractor': {'factory': 'cbj.pytorch.feature_extractor.FeatureExtractor',
          'type': 'stft',
          'size': 512,
          'shift': 128,
          'window_length': 512,
          'pad': True,
          'fading': True,
          'output_size': 257},
         'db': {'factory': 'paderbox.database.wsj_bss.database.WsjBss',
          'json_path': '/home/cbj/storage/database_jsons/wsj_bss.json',
          'datasets_train': None,
          'datasets_eval': None,
          'datasets_test': None},
         'sources': 2,
         'aux_loss': {'factory': 'model.AuxiliaryLoss',
          'stft_size': 512,
          'permutation_alignment': False,
          'iterations': 5}}
        """

        config['db'] = {
            'factory': pb.database.wsj_bss.WsjBss,
        }

        config['feature_extractor'] = {
            'factory': FeatureExtractor,
            'type': 'stft',
            'size': 1024 // 2,
            'shift': 256 // 2,
        }

        config['sources'] = 2

        config['blstm'] = {
            'factory': get_blstm_stack,
            'input_size': config['feature_extractor']['output_size'],
            # 'hidden_size': [256, 256],
            'hidden_size': [600, 600],
            'output_size': 600,
            'bidirectional': True,
            # 'dropout': 0.3,
            'dropout': 0.,
        }

        assert config['feature_extractor']['output_size'] == config['blstm']['input_size'], config

        config['dense'] = {
            'factory': pt.modules.fully_connected_stack,
            'input_size': config['blstm']['output_size'] * 2,
            # 'hidden_size': [500, 500],
            'hidden_size': [600],
            'output_size': config['feature_extractor']['output_size'] * config['sources'],
            'activation': 'relu',
            'dropout': 0.0,
        }

        assert config['dense']['input_size'] == config['blstm']['output_size'] * 2, config

        config['aux_loss'] = {
            'factory': AuxiliaryLoss,
            'stft_size': config['feature_extractor']['size'],
        }

    def __init__(
            self,
            blstm: get_blstm_stack,
            dense: pt.modules.fully_connected_stack,
            feature_extractor: FeatureExtractor,
            db,
            sources,
            # permutation_alignment=False,
            aux_loss,
            output_activation='softmax',
            pit_loss='mse',
    ):
        super().__init__()
        self.blstm = blstm
        self.dense = dense
        self.feature_extractor = feature_extractor
        self.criterion = torch.nn.CrossEntropyLoss()
        self.db = db
        self.sources = sources
        self.aux_loss = aux_loss

        self.pit_loss = pit_loss

        if output_activation == 'softmax':
            self.output_activation = torch.nn.Softmax(dim=-1)
        else:
            self.output_activation = pt.ops.mappings.ACTIVATION_FN_MAP[output_activation]()

    def get_iterable(self, dataset):
        if isinstance(self.db, pb.database.wsj_bss.WsjBss):
            it = self.db.get_iterator_by_names(dataset)
            it = it.map(pb.database.iterator.AudioReader(
                audio_keys=['speech_source', 'rir'],
                read_fn=self.db.read_fn,
            ))
            it = it.map(functools.partial(
                pb.database.wsj_bss.scenario_map_fn,
                channel_mode='all',
                truncate_rir=False,
                snr_range=(20, 30),  # Too high, reviewer won't like this
                rir_type='image_method'
            ))

            return it
        else:
            raise TypeError(self.db)

    def transform(self, example):
        # example['audio_data']['speech_image']
        # example['audio_data']['noise_image']
        Observation = self.feature_extractor(example['audio_data']['observation'])
        # shape D T F

        if self.use_guide:
            assert self.sources == 2, self.sources
            Speech_image = self.feature_extractor(example['audio_data']['speech_image'])
            Clean = np.abs(Speech_image).astype(np.float32)
        else:
            Clean = None
            Speech_image = None

        return self.NNInput(
            Observation=Observation,
            Feature=np.abs(Observation).astype(np.float32),
            Target=Clean,
            Speech_image=Speech_image,
        )

    @dataclass
    class NNInput:
        Observation: np.ndarray
        Feature: torch.tensor
        Target: torch.tensor
        Speech_image: np.ndarray
        # alignment: torch.tensor
        # kaldi_transcription: tuple

    def forward(self, example: NNInput):
        if isinstance(example, (tuple, list)):
            # Batch Mode
            return [
                self.forward(e) for e in example
            ]

        tensor, _ = self.blstm(example.Feature)
        predict = self.dense(tensor)
        # shape = list(predict.shape)
        # shape[-1] = shape[-1] // self.sources
        # shape += [self.sources]
        # Split last dimension to frequencies times speakers
        # Average above channel
        # Apply Softmax above speakers
        # torch.mean(predict.reshape(shape), dim=-4)

        predict = einops.rearrange(
            predict,
            'D T (F K) -> D T F K'.lower(),
            k=self.sources,
        )

        return self.output_activation(predict)

    def review(self, inputs: NNInput, predict, summary=None):
        """

        >>> np.set_string_function(lambda a: f'array(shape={a.shape}, dtype={a.dtype})')

        >>> Observation = pb.utils.random_utils.normal([3, 4, 5], np.complex128)
        >>> Feature = np.abs(Observation).astype(np.float32)
        >>> example = Model.NNInput(Observation=Observation, Feature=Feature)
        >>> example = pt.data.example_to_device(example)
        >>> example

        >>> model = Model.from_config(Model.get_config({'feature_extractor': {'output_size': 5}}))
        >>> print(model)
        Model(
          (blstm): LSTM(5, 256, num_layers=3, dropout=0.3, bidirectional=True)
          (dense): Sequential(
            (dropout_0): Dropout(p=0.3)
            (linear_0): Linear(in_features=512, out_features=500, bias=True)
            (relu_0): ReLU()
            (dropout_1): Dropout(p=0.3)
            (linear_1): Linear(in_features=500, out_features=500, bias=True)
            (relu_1): ReLU()
            (dropout_2): Dropout(p=0.3)
            (linear_2): Linear(in_features=500, out_features=10, bias=True)
            (softmax): Softmax()
          )
          (criterion): CrossEntropyLoss()
        )
        >>> predict = model(example)
        >>> predict.shape
        torch.Size([4, 5, 2])
        >>> review = model.review(example, predict)
        >>> review

        """
        from padertorch.contrib.cb.summary import ReviewSummary

        if summary is None:
            summary = ReviewSummary()

        if isinstance(inputs, (tuple, list)):
            assert isinstance(predict, (tuple, list)), (type(predict), predict)
            assert len(predict) == len(inputs), (len(predict), len(inputs), predict, inputs)
            # Batch Mode
            for i, p in zip(inputs, predict):
                summary = self.review(inputs=i, predict=p, summary=summary)

            return summary

        # predict.shape == D T F K

        # loss = regularisation + loss

        if inputs.Target is not None:
            # inputs.Target.shape: K D T F
            # inputs.Observation.shape: D T F
            # predict.shape: T F K

            target = inputs.Target  # .mean(-3)
            # target.shape: K T F
            # target = target / (target.sum(-3, keepdim=True) + 1e-6)

            assert inputs.Feature.shape == predict.shape[:-1], (inputs.Feature.shape, predict.shape)

            predict_amp = einops.rearrange(
                inputs.Feature[..., None] * predict,
                'D T F K -> (D T) K F'.lower()
            )


            # pit mse
            if self.pit_loss == 'mse':
                mask_mse_loss = pt.ops.loss.pit_loss(
                    predict_amp,
                    einops.rearrange(target, 'K D T F -> (D T) K F'.lower()),
                )

            elif 'ips' in self.pit_loss:
                # pit ips: Ideal Phase Sensitive
                # pit nips: Nonnegativ Ideal Phase Sensitive

                cos_phase_diff = np.cos(
                    np.angle(inputs.Observation)
                    - np.angle(inputs.Speech_image)
                ).astype(np.float32)

                cos_phase_diff = target.new(cos_phase_diff)

                if 'ips' == self.pit_loss:
                    mask_mse_loss = pt.ops.losses.loss.pit_loss(
                        predict_amp,
                        einops.rearrange(target * cos_phase_diff,
                                         'K D T F -> (D T) K F'.lower())
                    )
                elif 'nips' == self.pit_loss:
                    mask_mse_loss = pt.ops.losses.loss.pit_loss(
                        predict_amp,
                        einops.rearrange(target * cos_phase_diff,
                                         'K D T F -> (D T) K F'.lower())
                    )
                else:
                    raise ValueError(self.pit_loss)
            else:
                raise ValueError(self.pit_loss)

            summary.add_scalar(f'reconstruction_{self.pit_loss}', mask_mse_loss)
            summary.add_to_loss(mask_mse_loss)

            for i, p in enumerate(einops.rearrange(target, 'K D T F -> D K T F'.lower())[0]):
                summary.add_image(f'clean_{i}', pt.summary.spectrogram_to_image(p))
        else:
            aux_loss = self.aux_loss(predict, inputs.Observation)
            summary.add_to_loss(aux_loss)
            summary.add_scalar('aux_loss', aux_loss)

        for i, mask in enumerate(
                einops.rearrange(
                    predict,
                    'D T F K -> D K T F'.lower()
                )[0, :, :, :]
        ):
            summary.add_image(f'mask_{i}', pt.summary.mask_to_image(mask))

        summary.add_image('Observation', pt.summary.stft_to_image(inputs.Observation[0]))

        return summary
