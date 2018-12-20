import padertorch
from sacred import Experiment as Exp
exp = Exp('Timit Classifier')


class Classifier(padertorch.Module):
    @classmethod
    def get_defaults_from_init_signature(self):
        defaults = super().get_defaults_from_init_signature()
        defaults['encoder'] = {
            'cls': 'pytorch_sanity.config_tg.DenseEncoder',
            'kwargs': {'layers': 5},
            'DenseStack': {},
            'pytorch_sanity.config_tg.DenseEncoder': {'layers': 3,
                                                      'nonlinearity': 'sigmoid'},
            'pytorch_sanity.config_tg.RecurrentEncoder': {'layers': 4},
        }
        return defaults

    def __init__(self, net):
        pass

@exp.config
def config():
    pass


@exp.automain
def main(_config):
    pass
