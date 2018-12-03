"""DESIRED.

Will be evaluated in terms of WTFs/min.
"""
import importlib
from copy import deepcopy
from sacred import Experiment as Exp
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from dataclasses import asdict, dataclass, field


class Parameterized:
    @dataclass
    class opts:
        pass

    def __init__(self, opts):
        if hasattr(opts, '__dataclass_fields__'):
            self.opts = opts
        else:
            self.opts = self.recursive_dataclass_init(self.opts, opts)

    @classmethod
    def recursive_dataclass_init(cls, data_cls, opts_dict):
        fields = data_cls.__dataclass_fields__
        opts = opts_dict.copy()
        for k, v in opts_dict.items():
            if k not in fields:
                raise KeyError(
                    f'{k} is not a field of the dataclass {data_cls}'
                )
            if hasattr(fields[k].type, '__dataclass_fields__'):
                opts[k] = cls.recursive_dataclass_init(
                    fields[k].type, v)
        return data_cls(**opts)

    @classmethod
    def recursive_replace(cls, data_cls: Type[opts], replace_dict: dict):
        fields = data_cls.__dataclass_fields__
        for k, v in replace_dict.items():
            if isinstance(v, dict):
                if k not in fields:
                    raise KeyError(
                        f'{k} is not a field of the dataclass {data_cls}'
                    )
                if isinstance(getattr(data_cls, k), dict):
                    setattr(data_cls, k, v)
                else:
                    if not hasattr(getattr(data_cls, k),
                                   '__dataclass_fields__'):
                        raise ValueError(
                            f'Provided a dictionary for {k} but the field is not '
                            f'nested'
                        )
                    replaced = cls.recursive_replace(getattr(data_cls, k), v)
                    setattr(data_cls, k, replaced)
            else:
                setattr(data_cls, k, v)
        return data_cls

    @classmethod
    def recursive_default_update(cls, data_cls, opts_dict):
        fields = data_cls.__dataclass_fields__
        opts = opts_dict.copy()
        for k, v in opts_dict.items():
            if k not in fields:
                raise KeyError(
                    f'{k} is not a field of the dataclass {data_cls}'
                )
            if '_cls' in k:
                if isinstance(opts_dict[k], str):
                    imported_class = import_class(opts_dict[k])
                else:
                    imported_class = opts_dict[k]
                opts_key = k.replace('_cls', '')
                if imported_class == fields[k].default:
                    default_dict = asdict(fields[opts_key].default)
                    default_dict.update(opts[opts_key])
                else:
                    if not opts_key in opts_dict:
                        opts[opts_key] = dict()
                    default_dict = opts[opts_key]
                opts[opts_key] = imported_class.get_defaults(opts[opts_key])
        return data_cls(**opts)

    @classmethod
    def get_defaults(cls, update_dict={}) -> opts:
        return cls.recursive_default_update(cls.opts, update_dict)



class GRU(Parameterized):
    @dataclass
    class opts:
        nonlinearity: int = 'tanh'


class LSTM(Parameterized):
    @dataclass
    class opts:
        peephole: bool = False

class DenseEncoder(Parameterized):
    @dataclass
    class opts:
        layers: int = 2
        nonlinearity: str = 'elu'


class RecurrentEncoder(Parameterized):
    @dataclass
    class opts:
        layers: int = 2
        bidirectional: bool = False
        recurrent_cls: Parameterized = GRU
        recurrent: Parameterized.opts = GRU.opts(nonlinearity='sigmoid')

class VAE(Parameterized):
    @dataclass
    class opts:
        encoder: Parameterized.opts = DenseEncoder.opts()
        encoder_cls: Parameterized = DenseEncoder


def import_class(name: str):
    if name == 'GRU':
        return GRU
    elif name == 'LSTM':
        return LSTM
    elif name == 'DenseEncoder':
        return DenseEncoder
    elif name == 'RecurrentEncoder':
        return RecurrentEncoder
    else:
        raise ValueError(name)

exp = Exp('vae')

@exp.config
def config():
    opts = {}

@exp.named_config
def new_encoder():
    opts=dict(encoder_cls='RecurrentEncoder',
              encoder = dict(layers=3)
              )

@exp.named_config
def new_recurrent():
    opts=dict(encoder_cls='RecurrentEncoder',
              encoder = dict(layers=3,
                             recurrent_cls='LSTM',
                             recurrent=dict(peephole=True))
              )



@exp.automain
def main(_config):
    print(asdict(VAE.get_defaults(_config['opts'])))

# python pytorch_sanity/config_jh.py
# {'encoder': {'layers': 2, 'nonlinearity': 'elu'}, 'encoder_cls': <class '__main__.DenseEncoder'>}
# python pytorch_sanity/config_jh.py with new_encoder --force
# {'encoder': {'layers': 3, 'bidirectional': False, 'recurrent_cls': <class '__main__.GRU'>, 'recurrent': {'nonlinearity': 'sigmoid'}}, 'encoder_cls': 'RecurrentEncoder'}
# python pytorch_sanity/config_jh.py with new_recurrent --force
# {'encoder': {'layers': 3, 'bidirectional': False, 'recurrent_cls': 'LSTM', 'recurrent': {'peephole': True}}, 'encoder_cls': 'RecurrentEncoder'}

