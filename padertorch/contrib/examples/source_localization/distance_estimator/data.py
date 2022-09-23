from functools import partial

import numpy as np
from scipy.signal import lfilter, fftconvolve

from lazy_dataset.core import FilterException
from lazy_dataset.database import JsonDatabase
from paderbox.io.audioread import load_audio
from paderbox.transform.module_stft import get_stft_center_frequencies, STFT
from paderbox.utils.random_utils import str_to_random_generator
from padertorch import Configurable
from padertorch.data.utils import collate_fn


def reverb_signal(signal, rirs, mode='valid'):
    """
    Wrapper for reverberation (filtering in frequency domain)

    Args:
        signal:
            Signal to be reverberated
        rirs:
            Room impulse responses to be used for reverberation
        mode: {'valid', 'same'}, defined as in numpy.convolve
            'same':
                Mode 'same' returns output of length of ``sig´´.
                Boundary effects are visible.
            'valid':
                Mode 'valid' returns output of length
                ``size(sig) - size(rirs) + 1``.
                The convolution product is only given for points
                where the signal and the filter overlap completely.
                Boundary effects are omitted.
    """
    if type(rirs) == list or (rirs.ndim > 1, np.shape(rirs)[0] > 1):
        reverberated = [fftconvolve(signal, rir, mode) for rir in rirs]
        reverberated = np.asarray(reverberated)
    else:
        reverberated = fftconvolve(signal, rirs, mode)
    return reverberated


class PSD:
    """
    Helper class tu calculate the PSD
    """
    def __init__(self, smoothing_factor):
        self.smoothing_factor = smoothing_factor

    def __call__(self, x):
        psd = lfilter([1 - self.smoothing_factor], [1, -self.smoothing_factor],
                      x[0] * np.conj(x[1]), axis=0)
        return psd


class Coherence:
    """
    Helper class to calculate estimates of the coherence
    """
    def __init__(self, smoothing_factor):
        self.psd_estimator = PSD(smoothing_factor)

    def __call__(self, x):
        psd_x1_x1 = self.psd_estimator(np.tile(x[0, None, :, :], (2, 1, 1)))
        psd_x1_x2 = self.psd_estimator(x)
        psd_x2_x2 = self.psd_estimator(np.tile(x[1, None, :, :], (2, 1, 1)))
        denominator = np.sqrt(psd_x1_x1 * psd_x2_x2)
        # avoid divison by zero
        coherence = psd_x1_x2 / np.maximum(denominator,
                                           np.finfo(denominator.dtype).eps)
        return coherence


class Diffuseness:

    def __init__(self, psd_smoothing_factor=0.95, d_mic=0.05,
                 fft_length=1024, sample_rate=16000, sound_velocity=343):
        """
        Calculate the coherent-to-diffuse power ratio (CDR).
        The result is in time-frequency-domain.

            Args:
                psd_smoothing_factor (float):
                    Smoothing factor for the psd
                d_mic (float):
                    Distance of the microphones
                fft_length (int):
                    Length of the FFT used for STFT calculation
                sample_rate (int):
                    Sample rate of the signals
                sound_velocity (int):
                    Sound velocity, which is by default 343 m/s
        """
        self.center_frequencies = \
            get_stft_center_frequencies(fft_length, sample_rate)
        self.coherence_diffuse_field = \
            self._get_coherence_diffuse_field(d_mic, sound_velocity)
        self.coherence_estimator = Coherence(psd_smoothing_factor)

    def _get_coherence_diffuse_field(self, d_mic, sound_velocity):
        coherence_diffuse_field = \
            np.sinc(2 * self.center_frequencies * d_mic / sound_velocity)
        return coherence_diffuse_field

    def __call__(self, x):
        coherence = self.coherence_estimator(x)
        magnitude_threshold = 1.0 - 1e-11
        critical = np.abs(coherence) > magnitude_threshold
        coherence[critical] = (
                magnitude_threshold * coherence[critical]
                / np.abs(coherence[critical])
        )
        cdr = self.coherence_diffuse_field ** 2 * np.real(coherence) ** 2
        cdr -= self.coherence_diffuse_field ** 2 * np.abs(coherence) ** 2
        cdr += self.coherence_diffuse_field ** 2
        cdr -= 2 * self.coherence_diffuse_field * np.real(coherence)
        cdr += np.abs(coherence) ** 2
        # A positive value for further processing (i.e. np.sqrt) is needed.
        # This adjustment can be necessary because of numerical issues
        # or silence during the speech sample
        cdr = np.maximum(cdr, np.finfo(cdr.dtype).eps)
        cdr = - np.sqrt(cdr)
        cdr += self.coherence_diffuse_field * np.real(coherence)
        cdr -= np.abs(coherence) ** 2
        cdr /= np.abs(coherence) ** 2 - 1
        cdr = np.maximum(cdr.real, 0.)
        return 1 / (1 + cdr)


class AudioPreparation:
    def __init__(self, source_sig_path=None, audio_key=None,
                 signal_length=8192, random_speech_samples=True, **kwargs):
        """Preparation of microphone signals

        Function, that creates artificial room recordings.
        Firstly, elements of the room impulse response (RIR) dataset
        are used to generate the microphone signals by reverberating speech
        samples from LibriSpeech.

        If real audio recordings from rooms can be provided,they are loaded and
        returned without manipulation

        Args:
            source_sig_path (str):
                Path to a JSON for source signal database to create the
                microphone signals by reverberating source signal samples from
                this database with RIRs from the RIR database.
                Must be provided if no audio_key is specified.
            audio_key (str):
                States the key under which the real recording can be found in
                the database JSON.
                Must be provided if source_sig_path is not specified,
                as it is then assumed that real recordings should be used.
            signal_length (int):
                Length (in samples) of the microphone signals to be generated
            random_speech_samples:
                If True, random source signal samples are used for the creaton
                of the microphone signals
        """
        self.speech_set = None
        if audio_key is None:
            msg = "To generate the audio data, a path to a source signal" \
                  " database is needed"
            assert source_sig_path is not None, msg
            speech_db_part = kwargs.get('speech_db_part', 'train_clean_100')
            assert speech_db_part in ["dev_clean", "test_clean",
                                      "train_clean_100", "train_clean_360"]
            self.speech_set = \
                JsonDatabase(source_sig_path).get_dataset(speech_db_part)
        self.random_speech_samples = random_speech_samples
        if source_sig_path is None:
            msg = "The key under which the audio recording can be found in " \
                  "the JSON of the database is required."
            assert audio_key is not None, msg
        self.signal_length = signal_length
        self.mic_pair = kwargs.get('mic_pair')
        self.audio_key = audio_key

    def __call__(self, example):
        if self.speech_set is not None:
            rirs = self.prepare_rirs(example)
            speech = self.prepare_speech(example, rirs.shape[-1])
            mic_signals = reverb_signal(speech, rirs)
        else:
            # use of an own database is assumed,
            # where the creation of microphone signals is not needed
            mic_signals = self.prepare_recording(example)
        return mic_signals

    def prepare_rirs(self, example):
        rirs = load_audio(example['rir'], dtype=np.float32)
        assert rirs.shape[0] >= 2
        if rirs.shape[0] > 2:
            if 'mic_pair' in example.keys():
                mic_pair = example['mic_pair']
                rirs = rirs[np.asarray(mic_pair)]
            else:
                rirs = rirs[np.asarray(self.mic_pair)]
        return rirs

    def prepare_speech(self, example, len_rirs):
        min_len = self.signal_length + len_rirs - 1
        start_offset = 0
        end = -1
        if self.random_speech_samples:
            speech_key = str_to_random_generator(
                example["example_id"]).choice(len(self.speech_set))
            counter = 0
        while end <= min_len + start_offset:
            if self.random_speech_samples:
                speech_sample = self.speech_set[speech_key + counter]
                counter += 1
            else:
                speech_sample = self.speech_set.random_choice()
            end = speech_sample['offset']
            start_offset = speech_sample['onset']
        audio_path = speech_sample['audio_path']['observation']
        audio = load_audio(audio_path, dtype=np.float32)
        slice_start = np.random.randint(start_offset, end - min_len)
        prepared_audio = audio[slice_start:slice_start + min_len]
        return prepared_audio

    def prepare_recording(self, example):
        """For the use with an own database"""
        audio = load_audio(example[self.audio_key], dtype=np.float32)
        assert audio.shape[0] >= 2
        if audio.shape[0] > 2:
            if 'mic_pair' in example.keys():
                mic_pair = example['mic_pair']
                audio = audio[np.asarray(mic_pair)]
            else:
                audio = audio[np.asarray(self.mic_pair)]
        recording_len = audio.shape[-1]
        start_offset = 0
        end = recording_len
        if "offset" in example.keys() and "onset" in example.keys():
            end = example['offset']
            start_offset = example['onset']
        if end <= self.signal_length + start_offset:
            # current recording has not the sufficient length for the specified
            # desired signal length
            # example is skipped later during prefetch or catch
            raise FilterException
        slice_start = \
            np.random.randint(start_offset, recording_len - self.signal_length)
        prepared_audio = audio[:, slice_start:slice_start + self.signal_length]
        return prepared_audio


class FeatureExtraction:
    def __init__(self, audio_preparator=None, feature='stft',
                 frame_shift=160, frame_len=400, fft_length=1024,
                 sample_rate=16000, d_mic=0.05, f_min=None, f_max=None):
        """Feature extractor

            Function, that extracts different features or combinations of them
            (consisting of the stft, diffuseness, ipd, ild, phase and
            magnitude information) from the microphone signals.

                Args:
                    audio_preparator (AudioPreparation):
                        Object of AudioPreparation to generate the microphone
                        signals
                    feature (str):
                        String of all features that should be extracted and
                        used as input to the net
                    frame_shift (int):
                        Shift of the frames of the STFT
                    frame_len (int):
                        Length of the frames of the STFT
                    fft_length (int):
                        Length of the FFT used for STFT calculation
                    sample_rate (int):
                        Sample rate of the signals
                    d_mic:
                        Distance between the microphones of the analysed
                        microphone pair.
                    f_min (int):
                        Minimum frequency for lower bound if size of the input
                        features should be restricted to some relevant size
                    f_max (int):
                        Maximum frequency for upper bound if size of the input
                        features should be restricted to some relevant size
                """
        allowed_features = \
            ["stft", "diffuseness", "ild", "ipd", "phase", "mag"]
        msg = f'Wrong feature specified: "{feature}" not in {allowed_features}'
        assert all([i in allowed_features for i in feature.split()]), msg
        self.feature = feature
        self.center_frequencies = \
            get_stft_center_frequencies(fft_length, sample_rate)
        if f_min is not None and f_max is None:
            self.f_min = f_min
            self.f_max = sample_rate / 2
        elif f_max is not None and f_min is None:
            self.f_min = 0
            self.f_max = f_max
        elif f_max is not None and f_min is not None:
            self.f_min = f_min
            self.f_max = f_max
            self.frequency_mask = np.logical_and(
                self.center_frequencies >= self.f_min,
                self.center_frequencies <= self.f_max
            )
        else:
            self.frequency_mask = None
        self.stft = \
            STFT(frame_shift, fft_length, frame_len, fading=None, pad=False)
        self.sample_rate = sample_rate
        self.audio_preparator = audio_preparator
        FeatureExtraction.d_mic = d_mic

    def _filter_frequencies(self, x):
        if self.frequency_mask is not None:
            return x[:, :, self.frequency_mask]
        else:
            return x

    def __call__(self, example):
        mic_signals = self.audio_preparator(example)
        mic_stft = self.stft(mic_signals)
        for index, sub_feature in enumerate(self.feature.split()):
            if index == 0:
                result = getattr(
                    FeatureExtraction,
                    "extract_features_" + sub_feature
                )(mic_stft)
            else:
                feature_result = getattr(
                    FeatureExtraction,
                    "extract_features_" + sub_feature
                )(mic_stft)
                result = np.concatenate([result, feature_result])
        result = self._filter_frequencies(result)
        result = np.transpose(result, (0, 2, 1))
        example['features'] = result
        return example

    @staticmethod
    def extract_features_stft(mic_stft):
        return np.concatenate([np.abs(mic_stft), np.angle(mic_stft)])

    @staticmethod
    def extract_features_mag(mic_stft):
        mag = np.abs(mic_stft[0])
        mag = np.expand_dims(mag, axis=0)
        return mag

    @staticmethod
    def extract_features_phase(mic_stft):
        return np.angle(mic_stft)

    @staticmethod
    def extract_features_diffuseness(mic_stft):
        diffuseness = Diffuseness(d_mic=FeatureExtraction.d_mic)
        value = diffuseness(mic_stft)
        value = np.expand_dims(value, axis=0)
        return value

    @staticmethod
    def extract_features_ild(mic_stft):
        # due to numerical issues, in the subsequent calculations
        # log10(<0) might occur
        mic_stft = np.maximum(mic_stft, np.finfo(mic_stft.dtype).eps)
        ild = 20 * np.log10(np.abs(mic_stft[0])) \
            - 20 * np.log10(np.abs(mic_stft[1]))
        ild = np.expand_dims(ild, axis=0)
        return ild

    @staticmethod
    def extract_features_ipd(mic_stft):
        phase_difference = np.angle(mic_stft[1]) - np.angle(mic_stft[0])
        phase_difference = np.expand_dims(phase_difference, axis=0)
        feats = np.concatenate(
            [np.cos(phase_difference), np.sin(phase_difference)])
        return feats


class DataProvider(Configurable):
    """
    Helper class to prepare the datasets with shuffling,
    prefetching and the extraction of the features as well as data labels

        Args:
            feature_extractor (FeatureExtraction:
                Instance of FeatureExtraction to get the input features
            prefetch_buffer (int):
                Number of elements that are prefetched and buffered
            num_workers (int):
                Number of threads/processes used by the backend of lazy_dataset
            shuffle_buffer (int):
                If set, only a shuffles a window this size is shuffled
            batch_size (int):
                Number of examples per batch
            quant_step (float):
                Step size used to quantize the distances in order to
                classify the distances(The smaller quant_step, the
                higher the resolution of the distance classes)
            d_min (float):
                Lower limit (in m) of the considered range of the
                distance value
    """
    def __init__(self, feature_extractor=None, prefetch_buffer=None,
                 max_workers=8, shuffle_buffer=None, batch_size=None,
                 quant_step=0.1, d_min=0.5):
        self.feature_extractor = feature_extractor
        self.prefetch_buffer = prefetch_buffer
        self.num_workers = 0 if prefetch_buffer is None \
            else min(prefetch_buffer, max_workers)
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.quant_step = quant_step
        self.d_min = d_min

    def create_label(self, example, key='label'):
        if "distance" not in example.keys():
            assert "source_position" in example.keys \
                   and "node_position" in example.keys(), "Not all keys there"
            example["distance"] = \
                np.sqrt(np.sum((np.array(example["source_position"])
                                - np.array(example["node_position"])) ** 2))
        example[key] = \
            np.round((example['distance'] - self.d_min) / self.quant_step)
        return example

    def prepare_iterable(self, dataset, shuffle=True, prefetch=True,
                         reps=1, batch=True):
        def to_array(example):
            example['distance'] = \
                np.asarray(example['distance']).astype(np.float32)
            example['features'] =\
                np.asarray(example['features']).astype(np.float32)
            example['label'] = np.asarray(example['label']).astype(np.int_)
            return example

        dataset = dataset.map(self.create_label)

        if self.feature_extractor is not None:
            dataset = dataset.map(partial(self.feature_extractor))

        if shuffle:
            dataset = dataset.shuffle(reshuffle=True,
                                      buffer_size=self.shuffle_buffer)

        if prefetch and self.prefetch_buffer and self.num_workers:
            dataset = dataset.prefetch(self.num_workers, self.prefetch_buffer,
                                       catch_filter_exception=True)
        else:
            dataset = dataset.catch()

        if reps > 1:
            dataset = dataset.tile(reps)

        if batch:
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(collate_fn)
            dataset = dataset.map(to_array)

        return dataset
