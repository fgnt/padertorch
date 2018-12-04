"""DESIRED.

Will be evaluated in terms of WTFs/min.
"""
import importlib
from copy import deepcopy
from sacred import Experiment as Exp
from warnings import warn
import inspect


class Module:
    @classmethod
    def _get_defaults(cls):
        sig = inspect.signature(cls.__init__)
        sig = sig.replace(
            parameters=list(sig.parameters.values())[1:]  # drop self
        )
        return {k: v.default for k, v in sig.parameters.items()}

    @classmethod
    def get_config(cls, updates=None):
        config = cls._get_defaults()
        update_config(config, updates)
        return config


class VAE(Module):
    def __init__(
            self,
            encoder_cls='pytorch_sanity.config_je.DenseEncoder',
            encoder_kwargs=None
    ):
        pass


class DenseEncoder(Module):
    def __init__(self, layers=2, nonlinearity='elu'):
        pass


class RecurrentEncoder(Module):
    def __init__(
            self,
            layers=2,
            bidirectional=False,
            recurrent_cls='pytorch_sanity.config_je.GRU',
            recurrent_kwargs=None,
    ):
        pass


class GRU(Module):
    def __init__(self, nonlinearity='tanh'):
        pass


class LSTM(Module):
    def __init__(self, peephole=False):
        pass


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


def update_config(config, updates=None, strict=True):
    """

    :param config: config dict
    :param updates: updates dict. Note that update entries which are not valid
    for the current config are ignored.
    :return:
    """
    blacklist = list()
    for key in sorted(deepcopy(config).keys()):
        if key in blacklist:
            continue
        if isinstance(config[key], dict):
            print(key)
            update_config(
                config[key],
                updates.pop(key) if updates and key in updates else dict(),
                strict=strict
            )
        elif updates and key in updates:
            config[key] = updates.pop(key)
        if key.endswith('_cls'):
            kwargs_key = f'{key[:-4]}_kwargs'
            config[kwargs_key] = import_class(config[key]).get_config()
            blacklist.append(kwargs_key)
            update_config(
                config[kwargs_key],
                updates.pop(kwargs_key)
                if updates and kwargs_key in updates else dict(),
                strict=strict
            )
    if strict:
        assert not updates
    elif updates:
        warn(f"Updates not used: {updates}")


exp = Exp('vae')


@exp.config
def config():
    config_ = VAE.get_config()
    update_config(
        config_,
        dict(
            encoder_cls="pytorch_sanity.config_je.RecurrentEncoder",
            encoder_kwargs=dict(recurrent_cls="pytorch_sanity.config_je.LSTM")
        ),
        strict=False
    )


@exp.automain
def main(_config):
    pass
