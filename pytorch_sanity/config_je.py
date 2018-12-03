"""DESIRED.

Will be evaluated in terms of WTFs/min.
"""
import importlib
from copy import deepcopy
from sacred import Experiment as Exp


class VAE:
    @classmethod
    def get_defaults(cls):
        return {
            'encoder_cls': 'pytorch_sanity.config_je.DenseEncoder',
            'encoder': None
        }


class DenseEncoder:
    @classmethod
    def get_defaults(cls):
        return {'layers': 2, 'nonlinearity': 'elu'}


class RecurrentEncoder:
    @classmethod
    def get_defaults(cls):
        return {
            'layers': 2,
            'bidirectional': False,
            'recurrent_cls': 'pytorch_sanity.config_je.GRU',
            'recurrent': None
        }


class GRU:
    @classmethod
    def get_defaults(cls):
        return {'nonlinearity': 'tanh'}


class LSTM:
    @classmethod
    def get_defaults(cls):
        return {'peephole': False}


def import_class(name: str):
    splitted = name.split('.')
    module_name = '.'.join(splitted[:-1])
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(
            f'Tried to import module {module_name} to import class '
            f'{splitted[-1]}. During import an error happened. '
            f'Make sure that\n'
            f'\t1. This is the class you want to import.\n'
            f'\t2. You activated the right environment.\n'
            f'\t3. The module exists and has been installed with pip.\n'
            f'\t4. You can import the module (and class) in ipython.\n'
        )
        raise
    return getattr(module, splitted[-1])


def update_config(config, updates=None):
    """

    :param config: config dict
    :param updates: updates dict. Note that update entries which are not valid
    for the current config are ignored.
    :return:
    """
    blacklist = list()
    for key in deepcopy(config).keys():
        if key in blacklist:
            continue
        if isinstance(config[key], dict):
            update_config(
                config[key],
                updates[key] if updates and key in updates else dict()
            )
        elif updates and key in updates:
            config[key] = updates.pop(key)
        if key.endswith('_cls'):
            opts_key = key[:-4]
            config[opts_key] = import_class(config[key]).get_defaults()
            blacklist.append(opts_key)
            update_config(
                config[opts_key],
                updates[opts_key] if updates and opts_key in updates else dict()
            )


exp = Exp('vae')


@exp.config
def config():
    options = VAE.get_defaults()
    update_config(
        options,
        dict(
            # encoder_cls="pytorch_sanity.config_je.RecurrentEncoder",
            # encoder=dict(recurrent_cls="pytorch_sanity.config_je.LSTM")
        )
    )


@exp.automain
def main(_config):
    print(_config)

