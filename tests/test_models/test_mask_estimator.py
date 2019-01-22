import unittest
import padertorch as pt
from paderbox.database import keys as DB_K
from paderbox.database.iterator import AudioReader
import numpy as np
from paderbox.transform import stft
import torch
import paderbox as pb


K = pt.modules.mask_estimator.MaskKeys
AUDIO_KEYS = [DB_K.OBSERVATION, DB_K.SPEECH_IMAGE, DB_K.NOISE_IMAGE]

def transform(example):
    example.update({
        key: value for key, value in example[DB_K.AUDIO_DATA].items()
    })
    example[K.OBSERVATION_STFT] = stft(example[DB_K.OBSERVATION])
    example[K.OBSERVATION_ABS] = np.abs(example[K.OBSERVATION_STFT]).astype(np.float32)
    example[DB_K.NUM_FRAMES] = example[K.OBSERVATION_STFT].shape[-2]
    speech = stft(example[DB_K.SPEECH_IMAGE])
    noise = stft(example[DB_K.NOISE_IMAGE])
    target_mask, noise_mask = pb.speech_enhancement.biased_binary_mask(
        np.stack([speech, noise], axis=0),
    )
    example[K.SPEECH_MASK_TARGET] = target_mask.astype(np.float32)
    example[K.NOISE_MASK_TARGET] = noise_mask.astype(np.float32)
    return example

def get_iterators():
    db = pb.database.chime.Chime3()
    compose = pt.data.transforms.Compose(
        AudioReader(audio_keys=AUDIO_KEYS),
        transform
    )
    return (
        db.get_iterator_by_names(db.datasets_train).map(compose),
        db.get_iterator_by_names(db.datasets_eval).map(compose),
    )


class TestMaskEstimatorModel(unittest.TestCase):
    # TODO: Test forward deterministic if not train
    C = 4

    def setUp(self):
        self.model_class= pt.models.mask_estimator.MaskEstimatorModel
        self.model = self.model_class.from_config(
            self.model_class.get_config({}, {}))
        self.T = 100
        self.B = 4
        self.F = 513
        self.num_frames = [100, 90, 80, 70]
        self.inputs = {
            K.OBSERVATION_ABS: [
                np.abs(np.random.normal(
                    size=(self.C, num_frames_, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ],
            K.SPEECH_MASK_TARGET: [
                np.abs(np.random.choice(
                    [0, 1],
                    size=(self.C, num_frames_, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ],
            K.NOISE_MASK_TARGET: [
                np.abs(np.random.choice(
                    [0, 1],
                    size=(self.C, num_frames_, self.F)
                )).astype(np.float32)
                for num_frames_ in self.num_frames
            ]
        }

    def run_time_test(self):
        it_tr, it_dt = get_iterators()

        config = pt.Trainer.get_config(
            updates=pb.utils.nested.deflatten({
                'model.cls': self.model_class,
                'storage_dir': None,  # will be overwritten
                'max_trigger': None,  # will be overwritten
            })
        )

        pt.train.run_time_tests.test_run_from_config(
            config, it_tr, it_dt,
            test_with_known_iterator_length=False,
        )
        pt.train.run_time_tests.test_run_from_config(
            config, it_tr, it_dt,
            test_with_known_iterator_length=True,
        )

    def test_signature(self):
        assert callable(getattr(self.model, 'forward', None))
        assert callable(getattr(self.model, 'review', None))

    def test_forward(self):
        inputs = pt.data.batch_to_device(self.inputs)
        model_out = self.model(inputs)
        for mask, num_frames in zip(model_out[K.SPEECH_MASK_PRED],
                                    self.num_frames):
            expected_shape = (self.C, num_frames, self.F)
            assert mask.shape == expected_shape, mask.shape
        for mask, num_frames in zip(model_out[K.SPEECH_MASK_LOGITS],
                                    self.num_frames):
            expected_shape = (self.C, num_frames, self.F)
            assert mask.shape == expected_shape, mask.shape

    def test_review(self):
        inputs = pt.data.batch_to_device(self.inputs)
        mask = self.model(inputs)
        review = self.model.review(inputs, mask)

        assert 'loss' in review, review.keys()
        assert 'loss' not in review['scalars'], review['scalars'].keys()

    def test_minibatch_equal_to_single_example(self):
        inputs = pt.data.batch_to_device(self.inputs)
        model = self.model
        model.eval()
        mask = model(inputs)
        review = model.review(inputs, mask)
        actual_loss = review['loss']

        reference_loss = list()

        for observation, target_mask, noise_mask in zip(
            self.inputs[K.OBSERVATION_ABS],
            self.inputs[K.SPEECH_MASK_TARGET],
            self.inputs[K.NOISE_MASK_TARGET],
        ):
            inputs = {
                K.OBSERVATION_ABS: [observation],
                K.SPEECH_MASK_TARGET: [target_mask],
                K.NOISE_MASK_TARGET: [noise_mask]
            }
            inputs = pt.data.batch_to_device(inputs)
            mask = model(inputs)
            review = model.review(inputs, mask)
            reference_loss.append(review['loss'])

        reference_loss = torch.sum(torch.stack(reference_loss))

        np.testing.assert_allclose(
            actual_loss.detach().numpy(),
            reference_loss.detach().numpy(),
            atol=1e-3
        )


class TestMaskEstimatorSingleChannelModel(TestMaskEstimatorModel):
    C = 1
