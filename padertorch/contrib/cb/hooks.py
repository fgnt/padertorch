import signal
from padertorch.train.hooks import StopTrainingHook
from padertorch.train.trigger import Trigger


class CPUTimeLimitExceededHookTrigger(Trigger):
    def __init__(self):
        self._SIGXCPU_received = False
        signal.signal(signal.SIGXCPU, self.handler)

    def handler(self, signum, frame):
        print('Received SIGXCPU: CPU time limit exceeded')
        print('Graceful shutdown training')
        self._SIGXCPU_received = True

    def set_last(self, iteration, epoch):
        pass

    def __call__(self, iteration, epoch):
        return self._SIGXCPU_received


class CPUTimeLimitExceededHook(StopTrainingHook):
    def __init__(self):
        # Do not call super, to prevent a copy of this trigger
        self.trigger = CPUTimeLimitExceededHookTrigger()


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


    while True:
        print(time.perf_counter())
        if hook.pre_step(Trainer()):
            break
        time.sleep(1)

    thread.join()
