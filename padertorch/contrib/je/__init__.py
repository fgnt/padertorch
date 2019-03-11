
import paderbox.database.keys as pb_keys


class Keys:
    ACTIVITIES = "activities"
    AUDIO_DATA = pb_keys.AUDIO_DATA
    AUDIO_PATH = pb_keys.AUDIO_PATH
    EVENTS = pb_keys.EVENTS
    NUM_SAMPLES = pb_keys.NUM_SAMPLES
    NUM_FRAMES = pb_keys.NUM_FRAMES
    PHONES = pb_keys.PHONES
    ROOMS = "rooms"
    SCENE = pb_keys.SCENE
    SPECTROGRAM = "spectrogram"
    STFT = "stft"
    WORDS = pb_keys.WORDS

    @classmethod
    def lable_keys(self):
        return [
            self.ACTIVITIES, self.EVENTS, self.PHONES, self.ROOMS, self.SCENE,
            self.WORDS
        ]
