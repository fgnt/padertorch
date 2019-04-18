
import paderbox.database.keys as pb_keys


class Keys:
    ACTIVITIES = "activities"
    AUDIO_DATA = pb_keys.AUDIO_DATA
    AUDIO_PATH = pb_keys.AUDIO_PATH
    DELTAS = "deltas"
    ENERGY = "energy"
    EVENTS = pb_keys.EVENTS
    FRAGMENT_ID = "fragment_id"
    NUM_SAMPLES = pb_keys.NUM_SAMPLES
    NUM_FRAMES = pb_keys.NUM_FRAMES
    PHONES = pb_keys.PHONES
    ROOMS = "rooms"
    SCENE = pb_keys.SCENE
    SPECTROGRAM = "spectrogram"
    MEL_SPECTROGRAM = "mel_spectrogram"
    STFT = "stft"
    WORDS = pb_keys.WORDS

    @classmethod
    def lable_keys(cls):
        return [
            cls.ACTIVITIES, cls.EVENTS, cls.PHONES, cls.ROOMS, cls.SCENE,
            cls.WORDS
        ]
