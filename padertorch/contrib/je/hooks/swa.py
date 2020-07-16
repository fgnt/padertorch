from padertorch.train.hooks import TriggeredHook
from paderbox.utils.nested import nested_op


class SWAHook(TriggeredHook):
    """
    performs stochastic weight averaging of the trainers model or a submodule of it
    """
    def __init__(self, trigger, submodule=None):
        """

        Args:
            trigger:
            submodule:
        """
        super().__init__(trigger)
        self.submodule = [] if submodule is None else submodule.split('.')
        self.swa_module = None
        self.count = 0

    def state_dict(self):
        return {
            "swa_module": self.swa_module,
            "count": self.count
        }

    def load_state_dict(self, state_dict):
        self.swa_module = state_dict["swa_module"]
        self.count = state_dict["count"]

    def get_module(self, trainer):
        module = trainer.model
        for attr_name in self.submodule:
            module = getattr(module, attr_name)
        return module

    def pre_step(self, trainer):
        if self.trigger(iteration=trainer.iteration, epoch=trainer.epoch) \
                and trainer.iteration != 0:
            print('SWA')
            module = self.get_module(trainer)
            self.count += 1
            if self.swa_module is None:
                self.swa_module = module.state_dict()
            else:
                r = 1 / self.count
                self.swa_module = nested_op(
                    lambda x, y: (1-r) * x.to(y.device) + r * y,
                    self.swa_module,
                    module.state_dict()
                )
