import numpy as np
from padertorch.contrib.je.data.transforms import Transform as BaseTransform
from padertorch.contrib.je.data.transforms import segment_axis


class Transform(BaseTransform):
    def __init__(
            self,
            input_sample_rate=16000,
            target_sample_rate=16000,
            frame_step=160,
            frame_length=480,
            fft_length=512,
            n_mels=80,
            fmin=20,
            storage_dir=None,
            segment_length=200,
            label_key='speaker_id'
    ):
        super().__init__(
            input_sample_rate=input_sample_rate,
            target_sample_rate=target_sample_rate,
            frame_step=frame_step,
            frame_length=frame_length,
            fft_length=fft_length,
            fading=True,
            pad=True,
            n_mels=n_mels,
            fmin=fmin,
            label_key=label_key,
            storage_dir=storage_dir
        )
        self.segment_length = segment_length

    def finalize(self, example, training=False):
        # split channels
        log_mel = example['log_mel']
        if self.segment_length is not None:
            mel_segment_step = mel_segment_length = self.segment_length
            log_mel = segment_axis(
                [log_mel], axis=1,
                segment_step=[mel_segment_step],
                segment_length=[mel_segment_length],
                pad=False
            )[0]
            log_mel = log_mel.reshape((-1, mel_segment_length, self.n_mels))

        fragments = []
        if log_mel.shape[0] > 0:
            log_mel = np.split(log_mel, log_mel.shape[0], axis=0)
            for i in range(len(log_mel)):
                fragments.append(
                    {
                        'example_id': example['example_id'],
                        'log_mel': np.moveaxis(
                            log_mel[i].squeeze(0), 0, 1
                        ).astype(np.float32),
                        self.label_key: example[self.label_key]
                    }
                )
        return fragments
