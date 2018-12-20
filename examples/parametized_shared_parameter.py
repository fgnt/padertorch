import padertorch as pts
from IPython.lib.pretty import pprint
from padertorch.configurable_utils import deflatten

class Load(pts.configurable.Configurable):
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    def __call__(self, arg):
        print(self.__class__.__name__, arg, self.sample_rate)
        return arg + 5


class FeatureExtractor(pts.configurable.Configurable):
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    def __call__(self, arg):
        print(self.__class__.__name__, arg, self.sample_rate)
        return arg + 7


class Compose(pts.configurable.Configurable):
    def __init__(self, layer1, layer2, sample_rate=8000):
        self.layer1 = layer1
        self.layer2 = layer2

    def __call__(self, arg):
        print(self.__class__.__name__, arg)
        return self.layer2(self.layer1(arg)) + 11

    @classmethod
    def get_config(
            cls,
            updates=None,
            config=None,
    ):
        config = super().get_config(updates=updates, config=config)
        config['kwargs']['layer1']['kwargs']['sample_rate'] = config['kwargs']['sample_rate']
        config['kwargs']['layer2']['kwargs']['sample_rate'] = config['kwargs']['sample_rate']
        return config


class Model(pts.configurable.Configurable):
    """
    >>> pprint(Model.get_config())
    {'cls': 'parametized_shared_parameter.Model',
     'kwargs': {'transform': {'cls': 'parametized_shared_parameter.Compose',
       'kwargs': {'sample_rate': 8000,
        'layer1': {'cls': 'parametized_shared_parameter.Load',
         'kwargs': {'sample_rate': 8000}},
        'layer2': {'cls': 'parametized_shared_parameter.FeatureExtractor',
         'kwargs': {'sample_rate': 8000}}}}}}
    """
    @classmethod
    def get_signature(self):
        defaults = super().get_signature()
        defaults['transform'] = deflatten({
            'cls': Compose,
            'kwargs.sample_rate': 8000,
            'kwargs.layer1.cls': Load,
            'kwargs.layer2.cls': FeatureExtractor,

        }, sep='.')
        return defaults

    def __init__(self, transform):
        self.transform = transform


import sacred
import sacred.run
import sacred.commands
exp = sacred.Experiment('Shared Parameter')

@exp.config
def config():

    model = {}
    Model.get_config(  # second alternative update
        deflatten({
            'transform.kwargs.sample_rate': 44100,
        }, sep='.'),
        model,
    )


@exp.automain
def main(_config, _run: sacred.run.Run):
    """
    """
    sacred.commands.print_config(_run)

    model = Model.from_config(_config['model'])

    print('Model config')
    pprint(model.config)


if __name__ == '__main__':
    pass