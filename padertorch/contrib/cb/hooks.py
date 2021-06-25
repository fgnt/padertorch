import signal
from padertorch.train.hooks import StopTrainingHook, StopTraining, Hook
from padertorch.train.trigger import Trigger, IntervalTrigger


class CPUTimeLimitExceededHookTrigger(Trigger):
    """
    Graceful shutdown of Training.

    Shutdown after next iteration (i.e. as fast as possible, finish validation)
    $ ccssignal XCPU <reqid>
    Use `ccsalloc ... --notifyjob=XCPU,60m ...` to let ccs send the signal.

    Shutdown after next epoch (i.e. finish current epoch, good iterator state)
    $ ccssignal USR1 <reqid>  # Shutdown after next epoch

    """
    def __init__(self):
        self._SIGXCPU_received = False
        self._SIGUSR1_received = False
        signal.signal(signal.SIGXCPU, self.handler_SIGXCPU)
        signal.signal(signal.SIGUSR1, self.handler_SIGUSR1)

        self.epoch_trigger = IntervalTrigger(1, 'epoch')

    def handler_SIGXCPU(self, signum, frame):
        print('Received SIGXCPU: CPU time limit exceeded')
        print('Graceful shutdown training')
        self._SIGXCPU_received = True

    def handler_SIGUSR1(self, signum, frame):
        print('Received SIGUSR1: User-defined signal 1.')
        print(f'Graceful shutdown training when epoch '
              f'{self.epoch_trigger.last + 1} is finished')
        self._SIGUSR1_received = True

    def set_last(self, iteration, epoch):
        pass

    def __call__(self, iteration, epoch):
        return (
            (
                self.epoch_trigger(iteration, epoch)
                and self._SIGUSR1_received
            )
            or self._SIGXCPU_received
        )


class CPUTimeLimitExceededHook(StopTrainingHook):
    def __init__(self):
        # Do not call super, to prevent a copy of this trigger
        self.trigger = CPUTimeLimitExceededHookTrigger()


class PyroHook(Hook):

    pyro_inspector = None

    def pre_step(self, trainer):
        from cbj.pyro_inspect import PyroInspector
        if self.pyro_inspector is None:
            self.pyro_inspector = PyroInspector(2)
            self.pyro_inspector.__enter__()

    def close(self, trainer):
        if self.pyro_inspector is not None:
            self.pyro_inspector.__exit__()


if __name__ == '__main__':

    import os
    import time
    from threading import Thread

    hook = CPUTimeLimitExceededHook()

    pid = os.getpid()

    def killer():
        time.sleep(2.5)
        os.kill(pid, signal.SIGXCPU)

    thread = Thread(target=killer)
    thread.start()

    class Trainer:
        iteration = 0
        epoch = 0


    try:
        while True:
            print(time.perf_counter())
            if hook.pre_step(Trainer()):
                break
            time.sleep(1)
    except StopTraining:
        print('StopTraining')
    thread.join()


    hook = CPUTimeLimitExceededHook()

    def killer():
        time.sleep(0.5)
        os.kill(pid, signal.SIGUSR1)

    thread = Thread(target=killer)
    thread.start()

    class Trainer:
        iteration = 0
        epoch = 0

    try:
        while True:
            Trainer.iteration += 1
            if (Trainer.iteration % 5) == 0:
                Trainer.epoch += 1
            print(time.perf_counter(), Trainer.iteration, Trainer.epoch)
            if hook.pre_step(Trainer()):
                break
            time.sleep(1)
    except StopTraining:
        print('StopTraining')
    thread.join()
