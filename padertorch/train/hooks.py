import padertorch as pt
from paderbox.utils.nested import nested_op
from collections import defaultdict
from padertorch.train.trigger import IntervalTrigger
from tensorboardX import SummaryWriter
import numpy as np
import torch
from cached_property import cached_property

__all__ = [
    'SummaryHook',
    'ValidationHook',
]
class BaseHook:
    def __init__(self, trigger_step):
        self.trigger = IntervalTrigger.new(trigger_step)
        assert self.priority <= 50

    def pre_function(self, trainer):
        """
        function is called before each iteration of the train iterator
        :param trainer:
        :return:
        """
        pass

    def post_function(self, trainer, example, model_output, review):
        """
        function is called after each train step
        :param trainer:
        :param example:
        :param model_output:
        :param review:
        :return:
        """
        pass

    @property
    def priority(self):
        return 0

class SummaryHook(BaseHook):
    def __init__(self, trigger_step, summary_prefix='training'):
        super().__init__(trigger_step)
        self.reset_summary()
        self.summary_prefix = summary_prefix
        self.storage_dir = None


    @cached_property
    def writer(self):
        return SummaryWriter(str(self.storage_dir),
                             filename_suffix=self.summary_prefix)

    def reset_summary(self):
        # Todo: add figures
        self.summary = dict(
            losses=defaultdict(list),
            scalars=defaultdict(list),
            histograms=defaultdict(list),
            audios=dict(),
            images=dict()
        )

    def update_summary(self, review):
        for key, loss in review.get('losses', dict()).items():
            self.summary['losses'][key].append(loss.item())
        for key, scalar in review.get('scalars', dict()).items():
            self.summary['scalars'][key].append(
                scalar.item() if torch.is_tensor(scalar) else scalar)
        for key, histogram in review.get('histograms', dict()).items():
            self.summary['histograms'][key] = np.concatenate(
                [self.summary['histograms'].get(key, np.zeros(0)),
                 histogram.clone().cpu().data.numpy().flatten()]
            )[-10000:]  # do not hold more than 10K values in memory
        for key, audio in review.get('audios', dict()).items():
            self.summary['audios'][key] = audio  # snapshot
        for key, image in review.get('images', dict()).items():
            self.summary['images'][key] = image  # snapshot

    def dump_summary(self, iteration, timer):
        prefix = self.summary_prefix
        for key, loss in self.summary['losses'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(loss), iteration)
        for key, scalar in self.summary['scalars'].items():
            self.writer.add_scalar(
                f'{prefix}/{key}', np.mean(scalar), iteration)
        for key, scalar in timer.as_dict.items():
            if key in ['time_per_data_loading', 'time_per_train_step']:
                if 'time_per_step' in timer.as_dict.keys():
                    time_per_step = timer.as_dict['time_per_step']
                    if len(time_per_step) != len(scalar):
                        print(
                            'Warning: padertorch.Trainer timing bug.'
                            f'len(time_per_step) == {len(time_per_step)} '
                            f'!= len(scalar) == {len(scalar)}'
                        )
                    scalar = (
                        scalar.sum() / time_per_step.sum()
                    )
                    if key == 'time_per_data_loading':
                        key = 'time_rel_data_loading'
                    elif key == 'time_per_train_step':
                        key = 'time_rel_train_step'
                else:
                    # Something went wrong, most likely an exception.
                    pass
            self.writer.add_scalar(
                f'{prefix}/{key}', scalar.mean(), iteration)
        for key, histogram in self.summary['histograms'].items():
            self.writer.add_histogram(
                f'{prefix}/{key}', np.array(histogram), iteration
            )
        for key, audio in self.summary['audios'].items():
            if isinstance(audio, (tuple, list)):
                assert len(audio) == 2, (len(audio), audio)
                self.writer.add_audio(
                    f'{prefix}/{key}', audio[0],
                    iteration, sample_rate=audio[1]
                )
            else:
                self.writer.add_audio(
                    f'{prefix}/{key}', audio,
                    iteration, sample_rate=16000
                )
        for key, image in self.summary['images'].items():
            self.writer.add_image(f'{prefix}/{key}', image, iteration)
        self.reset_summary()

    def post_function(self, trainer, example, model_out, review):
        if self.storage_dir is None:
            self.storage_dir = trainer.storage_dir
        else:
            assert self.storage_dir == trainer.storage_dir
        self.update_summary(review)
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch)\
                or trainer.iteration == 1:
            self.dump_summary(trainer.iteration, trainer.timer)

    @property
    def priority(self):
        return 10



class ValidationHook(SummaryHook):
    def __init__(self, trigger_step, iterator):
        super().__init__(trigger_step, 'validation')
        self.iterator = iterator

    def pre_function(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch):
            print('Starting Validation')
            evaluation = trainer.validate(self.iterator)
            [self.update_summary(review) for review in evaluation]
            self.dump_summary(trainer.iteration, trainer.timer)
            print('Finished Validation')

    @property
    def priority(self):
        return 40
