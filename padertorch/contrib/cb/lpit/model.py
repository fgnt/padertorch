import functools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional
import einops
from einops import rearrange


import paderbox as pb
import padertorch as pt
from paderbox.speech_enhancement import ideal_binary_mask

import pb_bss.distribution.cwmm
import pb_bss.permutation_alignment

from cbj.pytorch.feature_extractor import FeatureExtractor
from padertorch.contrib.examples.acoustic_model.model import get_blstm_stack

from padertorch.contrib.cb.summary import ReviewSummary


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
            sample_channel=False,
            use_pack_sequence=False,
    ):
        super().__init__()
        self.blstm = blstm
        self.dense = dense
        self.feature_extractor = feature_extractor
        self.criterion = torch.nn.CrossEntropyLoss()
        self.db = db
        self.sources = sources
        self.aux_loss = aux_loss
        self.sample_channel = sample_channel
        self.use_pack_sequence = use_pack_sequence

        self.pit_loss = pit_loss

        if output_activation == 'softmax':
            self.output_activation = torch.nn.Softmax(dim=-1)
        else:
            self.output_activation = pt.ops.mappings.ACTIVATION_FN_MAP[output_activation]()

    def get_iterable(self, dataset, snr_range=(10, 20)):
        if isinstance(self.db, pb.database.wsj_bss.WsjBss):
            it = self.db.get_iterator_by_names(dataset)
            it = it.shuffle(reshuffle=True)
            it = it.map(pb.database.iterator.AudioReader(
                audio_keys=['speech_source', 'rir'],
                # read_fn=self.db.read_fn,
                read_fn=pb.io.load_audio,
            ))

            def sample_channel(example):
                channel = np.random.randint(
                    example['audio_data']['rir'].shape[-2]
                )
                example['audio_data']['rir'] = np.take(
                    example['audio_data']['rir'],
                    (channel,),
                    axis=-2,
                )
                return example

            if dataset in ['train_si284']:
                if self.sample_channel:
                    it = it.map(sample_channel)
            elif dataset in ['cv_dev93']:
                pass
            else:
                raise ValueError(dataset)

            it = it.map(functools.partial(
                pb.database.wsj_bss.scenario_map_fn,
                # channel_mode='all',
                # truncate_rir=False,
                # snr_range=(20, 30),  # Too high, reviewer won't like this
                snr_range=snr_range,  # Too high, reviewer won't like this
                # rir_type='image_method',
                sync_speech_source=True,
                add_speech_reverberation_direct=True,
                add_speech_reverberation_tail=False,
                # normalize=True,
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
            assert self.sources in [2, 3], self.sources
            if self.sources == 2:
                pass
                # assert self.pit_loss not in ['bce', 'mse_ibm'], self.pit_loss
            elif self.sources in [2, 3]:
                assert self.pit_loss in ['bce', 'mse_ibm', 'ce'], self.pit_loss
            else:
                raise ValueError(self.sources)


            Speech_image = self.feature_extractor(example['audio_data']['speech_image'])
            Speech_reverberation_direct = self.feature_extractor(example['audio_data']['speech_reverberation_direct'])
            Noise_image = self.feature_extractor(example['audio_data']['noise_image'])
            # Speech_image_abs = np.abs(Speech_image).astype(np.float32)
            Speech_reverberation_direct_abs = np.abs(Speech_reverberation_direct).astype(np.float32)

            if self.pit_loss == 'ce':

                # Use Speech_image or Speech_reverberation_direct
                # Consider tail?
                binary_mask = np.ascontiguousarray(
                    ideal_binary_mask(
                        np.array([*Speech_reverberation_direct, Noise_image]),
                        source_axis=-4,
                        one_hot=False,
                    )
                )
            else:
                binary_mask = np.ascontiguousarray(
                    ideal_binary_mask(
                        np.array([*Speech_reverberation_direct, Noise_image]),
                        source_axis=-4
                    ), np.float32
                )
        else:
            # Speech_image_abs = None
            Speech_reverberation_direct = None
            Speech_reverberation_direct_abs = None
            # Speech_image = None
            binary_mask = None

        return self.NNInput(
            Observation=Observation,
            Feature=np.abs(Observation).astype(np.float32),
            # Speech_image_abs=Speech_image_abs,
            # Speech_image=Speech_image,
            Speech_reverberation_direct=Speech_reverberation_direct,
            Speech_reverberation_direct_abs=Speech_reverberation_direct_abs,
            binary_mask=binary_mask,
        )

    @dataclass
    class NNInput:
        Observation: np.ndarray
        Feature: torch.tensor
        # Speech_image_abs: torch.tensor
        # Speech_image: np.ndarray
        Speech_reverberation_direct: np.ndarray
        Speech_reverberation_direct_abs: torch.tensor
        binary_mask: np.ndarray
        # alignment: torch.tensor
        # kaldi_transcription: tuple

    def forward(self, example: NNInput):
        """
        >>> import paderbox as pb
        >>> import padertorch as pt
        >>> from dataclasses import  asdict
        >>> example = Model.NNInput(Feature=np.zeros([6, 400, 257], dtype=np.float32), Target=None, Speech_image=None, Observation=None, target_mask=None)
        >>> example = pt.data.example_to_device(example)
        >>> output: Model.NNOutput = Model.from_config(Model.get_config())(example)
        >>> pb.notebook.pprint(output.predict)
        Tensor(shape=(6, 400, 257, 2), dtype=float32)
        >>> pb.notebook.pprint(output.predict_logit)
        Tensor(shape=(6, 400, 257, 2), dtype=float32)

        >>> example = pt.data.example_to_device(example)
        >>> output2: Model.NNOutput = Model.from_config(Model.get_config(updates=dict(use_pack_sequence=True)))([example])

        """
        if isinstance(example, (tuple, list)):
            if self.use_pack_sequence:
                import padertorch as pt
                list_of_features = [ex.Feature for ex in example]
                Feature = pt.ops.pack_sequence_include_channel(
                    list_of_features
                )

                tensor, _ = self.blstm(Feature)

                predict = pt.ops.sequence_elementwise(self.dense, tensor)

                predict = pt.ops.sequence_elementwise(
                    einops.rearrange,
                    predict,
                    'TBD (F K) -> TBD F K'.lower(),
                    k=self.sources,
                )

                predict = pt.ops.unpack_sequence_include_channel_like(
                    predict,
                    like=list_of_features,
                )

                return [self.NNOutput(
                    predict_logit=p,
                    predict=self.output_activation(p),
                ) for p in predict]
            else:

                # Batch Mode
                return [
                    self.forward(e)
                    for e in example
                ]

        assert self.blstm.batch_first is False, (self.blstm, self.blstm.batch_first)
        assert example.Feature.shape[0] < 30, example.Feature.shape
        tensor, _ = self.blstm(
            rearrange(example.Feature, 'd t f -> t d f')
        )
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
            'T D (F K) -> D T F K'.lower(),
            k=self.sources,
        )

        return self.NNOutput(
            predict_logit=predict,
            predict=self.output_activation(predict),
        )

    @dataclass
    class NNOutput:
        predict_logit: torch.tensor
        predict: torch.tensor

    def review(self, inputs: NNInput, output: NNOutput, summary=None, add_images=True):
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

        if summary is None:
            summary = ReviewSummary()

        if isinstance(inputs, (tuple, list)):
            assert isinstance(output, (tuple, list)), (type(output), output)
            assert len(output) == len(inputs), (len(output), len(inputs), output, inputs)
            # Batch Mode
            last_index = len(inputs)-1

            for index, (i, o) in enumerate(zip(inputs, output)):
                # Only the last call should add the images.
                bool_add_images = (index == last_index)

                summary = self.review(inputs=i, output=o, summary=summary, add_images=bool_add_images)

            return summary

        predict = output.predict

        # predict.shape == D T F K

        # loss = regularisation + loss

        if inputs.Speech_reverberation_direct_abs is not None:
            # inputs.Target.shape: K D T F
            # inputs.Observation.shape: D T F
            # predict.shape: T F K

            # target = inputs.Speech_reverberation_direct_abs  # .mean(-3)
            # target.shape: K T F
            # target = target / (target.sum(-3, keepdim=True) + 1e-6)

            assert inputs.Feature.shape == predict.shape[:-1], (inputs.Feature.shape, predict.shape)

            D, T, F, K = predict.shape

            assert D < 30, D
            assert K < 30, K

            def get_predict_amp():
                predict_amp = einops.rearrange(
                    inputs.Feature[..., None] * predict,
                    'D T F K -> D T K F'.lower()
                )
                return predict_amp

            # pit mse
            if self.pit_loss == 'mse':
                predict_amp = get_predict_amp()
                mask_mse_loss = sum([pt.ops.loss.pit_loss(
                    predict_amp[d],
                    einops.rearrange(inputs.Speech_reverberation_direct_abs, 'K D T F -> D T K F'.lower())[d],
                ) for d in range(D)]) / D
            elif self.pit_loss in ['ce']:
                loss_fn = torch.nn.functional.cross_entropy
                estimate = einops.rearrange(
                    output.predict_logit,
                    'D T F K -> D T K F'.lower()
                )

                mask_mse_loss = sum([pt.ops.loss.pit_loss(
                    estimate[d],
                    einops.rearrange(inputs.binary_mask, 'D T F -> D T F'.lower())[d],
                    loss_fn=loss_fn,
                ) for d in range(D)]) / D

                if add_images:
                    tmp_mask = einops.rearrange(inputs.binary_mask,
                                                'D T F -> D T F'.lower())[0]
                    summary.add_mask_image(
                        f'target_mask',
                        einops.rearrange(
                            [tmp_mask == k for k in range(K)],
                            'k t f -> t (k f)'
                        ),
                    )

            elif self.pit_loss in ['bce', 'mse_ibm']:

                if self.pit_loss == 'bce':
                    # loss_fn = torch.nn.BCELoss()
                    loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

                    estimate = einops.rearrange(
                        output.predict_logit,
                        'D T F K -> D T K F'.lower()
                    )
                elif self.pit_loss == 'mse_ibm':
                    loss_fn = torch.nn.functional.mse_loss
                    estimate = einops.rearrange(
                        output.predict,
                        'D T F K -> D T K F'.lower()
                    )
                else:
                    raise ValueError(self.pit_loss)

                target_mask = inputs.binary_mask
                if self.sources == 2:
                    assert K == 2, K
                    assert target_mask.shape[0] == 3
                    # Drop noise mask
                    target_mask = target_mask[:-1, ...]

                target_mask = einops.rearrange(
                    target_mask, 'K D T F -> D T K F'.lower()
                )

                mask_mse_loss = sum([pt.ops.loss.pit_loss(
                    estimate[d],
                    target_mask[d],
                    loss_fn=loss_fn,
                ) for d in range(D)]) / D

                if add_images:
                    summary.add_mask_image(
                        f'target_mask',
                        einops.rearrange(
                            inputs.binary_mask,
                            'k d t f -> d t (k f)'
                        )[0],
                    )
            elif 'ips' in self.pit_loss:
                # pit ips: Ideal Phase Sensitive
                # pit nips: Nonnegativ Ideal Phase Sensitive


                # ToDo: is Speech_reverberation_direct still a valid signal?

                cos_phase_diff = np.cos(
                    np.angle(inputs.Observation)
                    - np.angle(inputs.Speech_reverberation_direct)
                ).astype(np.float32)

                cos_phase_diff = inputs.Speech_reverberation_direct_abs.new(cos_phase_diff)

                # Allow different permutations for different channels

                if 'ips' == self.pit_loss:
                    pass
                elif 'nips' == self.pit_loss:
                    cos_phase_diff = torch.nn.ReLU()(cos_phase_diff)
                else:
                    raise ValueError(self.pit_loss)

                predict_amp = get_predict_amp()
                mask_mse_loss = sum([pt.ops.losses.loss.pit_loss(
                    predict_amp[d],
                    einops.rearrange(inputs.Speech_reverberation_direct_abs * cos_phase_diff,
                                     'K D T F -> D T K F'.lower())[d]
                ) for d in range(D)]) / D
            else:
                raise ValueError(self.pit_loss)

            summary.add_scalar(f'reconstruction_{self.pit_loss}', mask_mse_loss)
            summary.add_to_loss(mask_mse_loss)

            if add_images:
                summary.add_spectrogram_image(
                    f'Speech_reverberation_direct',
                    einops.rearrange(inputs.Speech_reverberation_direct_abs, 'K D T F -> D T (K F)'.lower())[0],
                )
        else:
            aux_loss = self.aux_loss(predict, inputs.Observation)
            summary.add_to_loss(aux_loss)
            summary.add_scalar('aux_loss', aux_loss)

        if add_images:
            summary.add_mask_image(
                f'mask',
                einops.rearrange(
                    predict,
                    'D T F K -> D T (K F)'.lower()
                )[0, :, :],
            )

        if add_images:
            summary.add_stft_image('Observation', inputs.Observation[0])

        return summary
