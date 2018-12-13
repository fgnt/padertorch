"""DESIRED.

Will be evaluated in terms of WTFs/min.
"""
import collections
import dataclasses
from paderbox.utils.misc import merge_dict
from dataclasses import dataclass
import inspect
import json
import difflib
from IPython.lib.pretty import pretty

from pytorch_sanity.config_je import import_class


def maybe_str_to_instance(maybe_str):
    if isinstance(maybe_str, str):
        # ToDo: use paderflow eval
        if '.' in maybe_str:
            maybe_str = import_class(maybe_str)
        else:
            maybe_str = eval(maybe_str)
    return maybe_str


def get_default_type(parameter: inspect.Parameter):
    """
    >>> def get_parameter(default=inspect.Parameter.empty, annotation=inspect.Parameter.empty):
    ...     return inspect.Parameter(
    ...         'abc', inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ...          default=default, annotation=annotation)
    >>> class GRU2(GRU): pass
    >>> get_default_type(get_parameter(default=GRU()))
    <class 'model_challenge_cb_v3.5.GRU'>
    >>> get_default_type(get_parameter(annotation=GRU))
    <class 'model_challenge_cb_v3.5.GRU'>
    >>> get_default_type(get_parameter(default=GRU2(), annotation=GRU))
    <class 'model_challenge_cb_v3.5.GRU2'>
    >>> get_default_type(get_parameter(default=GRU(), annotation=GRU2))
    Traceback (most recent call last):
    ...
    Exception: The default value (<class 'model_challenge_cb_v3.5.GRU'>)
    must be in instance of the
    annotation type (<class 'model_challenge_cb_v3.5.GRU2'>)
    or dict.
    >>> get_default_type(get_parameter(default={'a': 1}, annotation=GRU2))
    <class 'model_challenge_cb_v3.5.GRU2'>
    >>> get_default_type(get_parameter(default=1, annotation=GRU2))
    Traceback (most recent call last):
    ...
    Exception: The default value (<class 'int'>)
    must be in instance of the
    annotation type (<class 'model_challenge_cb_v3.5.GRU2'>)
    or dict.
    >>> get_default_type(get_parameter(default={}))
    <class 'inspect._empty'>
    """
    if isinstance(parameter.default, ParametrizedMeta):
        raise TypeError(
            f'Expect an instance of {Parametrized!r} for '
            f'default ({parameter.default!r}) and not a class.'
        )
    if issubclass(parameter.annotation, Parametrized):
        if isinstance(parameter.default, Parametrized):
            if not isinstance(parameter.default, (parameter.annotation)):
                raise Exception(
                    f'The default value ({type(parameter.default)!r})\n'
                    f'must be in instance of the\n'
                    f'annotation type ({parameter.annotation!r})\n'
                    f'or dict.'
                )
            return type(parameter.default)
        elif parameter.default is inspect.Parameter.empty \
                or parameter.default is None \
                or isinstance(parameter.default, dict):
            return parameter.annotation
        else:
            raise Exception(
                f'The default value ({type(parameter.default)!r})\n'
                f'must be in instance of the\n'
                f'annotation type ({parameter.annotation!r})\n'
                f'or dict.'
            )
    elif isinstance(parameter.default, Parametrized):
        return type(parameter.default)
    # elif parameter.default is inspect.Parameter.empty \
    #     and parameter.annotation is inspect.Parameter.empty:
    #     return inspect.Parameter.empty
    else:
        return inspect.Parameter.empty
        # raise RuntimeError(
        #     f'Something went went wrong for {parameter}\n'
        #     f'Annotation: {parameter.annotation}\n'
        #     f'Default: {parameter.default}\n'
        # )


def nested_update(d1, d2):
    keys = list(  # https://stackoverflow.com/a/7961390/5766934
        collections.OrderedDict.fromkeys(
            list(d1.keys()) + list(d2.keys())
        )
    )
    res = {}
    for k in keys:
        if isinstance(d1.get(k), dict) and isinstance(d2.get(k), dict):
            res[k] = nested_update(d1[k], d2[k])
        elif k in d2:
            res[k] = d2[k]
        else:
            res[k] = d2[k]
    return res


class ParametrizedMeta(type):
    def __call__(cls, *args, **kwargs):
        STATE_ATTR = '_state'
        sig: inspect.Signature = inspect.signature(cls.__init__)
        # Drop self
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])

        # Split cls keywords and normal keywords
        kwargs_cls = {
            k[:-len('_cls')]: maybe_str_to_instance(v)
            for k, v in kwargs.items()
            if k.endswith('_cls')
        }
        kwargs = {k: v for k, v in kwargs.items() if not k.endswith('_cls')}

        # args to kwargs
        try:
            kwargs = sig.bind_partial(*args, **kwargs).arguments
        except Exception as e:
            raise Exception(
                    f'Invalid args ({args}) and kwargs ({kwargs}) for {cls}\n'
                    f'with signature {sig}'
                ) from e
        del args

        # Split Parametrized instance to dict and cls
        for k in list(kwargs.keys()):
            if isinstance(kwargs[k], Parametrized):
                kwargs_cls[k] = get_default_type(
                    inspect.Parameter(
                        k, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=kwargs[k], annotation=kwargs[k].__class__,
                    )
                )
                kwargs[k] = getattr(kwargs[k], STATE_ATTR)

        parameters = []
        for name, p in sig.parameters.items():
            p: inspect.Parameter
            p = p.replace(
                annotation=get_default_type(p),
                default=(
                    getattr(p.default, STATE_ATTR)
                    if isinstance(p.default, Parametrized)
                    else
                    p.default
                )
            )
            if name in kwargs_cls:
                new_cls = kwargs_cls[name]
                if not issubclass(new_cls, p.annotation) or p.default is None:
                    p = p.replace(
                        annotation=new_cls,
                        default={},
                    )
                if name in kwargs:
                    kwargs[name] = nested_update(p.default, kwargs[name])
            elif issubclass(p.annotation, Parametrized):
                kwargs_cls[name] = p.annotation

            parameters.append(p)

        sig = sig.replace(parameters=parameters)

        try:
            ba: inspect.BoundArguments = sig.bind(**kwargs)
        except Exception as e:
            raise Exception(
                f'Invalid args ({args}) and kwargs ({kwargs}) for {cls}\n'
                f'with (modified) signature {sig}'
            ) from e
        del kwargs
        #
        ba.apply_defaults()
        kwargs = ba.arguments
        for k, v_cls in kwargs_cls.items():
            cls_kwargs = kwargs[k]
            if cls_kwargs is None:
                cls_kwargs = {}
            kwargs[k] = v_cls(**cls_kwargs)

        state = {}
        for k, v in kwargs.items():
            if isinstance(v, Parametrized):
                state[k] = getattr(v, STATE_ATTR)
                module = v.__class__.__module__
                if module != '__main__':
                    state[k + '_cls'] = f'{module}.{v.__class__.__qualname__}'
                else:
                    state[k + '_cls'] = f'{v.__class__.__qualname__}'
            else:
                state[k] = v

        try:
            obj = super().__call__(**kwargs)
        except TypeError as e:
            raise Exception(kwargs) from e

        setattr(obj, STATE_ATTR, dict(sorted(state.items())))
        return obj


class Parametrized(metaclass=ParametrizedMeta):
    _state = None

    def __repr__(self):
        # prettyfied = pretty(self._state, newline="\n    ")
        # return f'{self.__class__.__name__}(\n    {prettyfied}\n)'
        prettyfied = pretty(self._state, newline="\n    ")
        return f'{self.__class__.__name__}({self._state})'


class GRU(Parametrized):
    def __init__(
            self,
            nonlinearity='tanh',
    ):
        super().__init__()
        self.nonlinearity = nonlinearity


class LSTM(Parametrized):
    def __init__(
            self,
            peephole=False,
    ):
        super().__init__()
        self.peephole = peephole


class DenseEncoder(Parametrized):
    def __init__(self, layers=2, nonlinearity='elu'):
        super().__init__()
        self.layers = layers
        self.nonlinearity = nonlinearity


class RecurrentEncoder(Parametrized):
    def __init__(
            self,
            layers=2,
            bidirectional=False,
            recurrent: GRU = None,
            # recurrent: GRU = GRU(nonlinearity='sigmoid'),
    ):
        super().__init__()
        self.layers = layers
        self.bidirectional = bidirectional
        self.recurrent = recurrent


class VAE(Parametrized):
    def __init__(
            self,
            # encoder: DenseEncoder = DenseEncoder(layers=3),
            encoder: DenseEncoder = dict(layers=3),
    ):
        super().__init__()
        self.encoder = encoder


def config(difficulty):
    """
    >>> from pprint import pprint
    >>> pprint(VAE())
    VAE({'encoder': {'layers': 3, 'nonlinearity': 'elu'}, 'encoder_cls': 'model_challenge_cb_v3_5.DenseEncoder'})
    >>> pprint(VAE()._state)
    {'encoder': {'layers': 3, 'nonlinearity': 'elu'},
     'encoder_cls': 'model_challenge_cb_v3_5.DenseEncoder'}

    Replace the DenseEncoder with RecurrentEncoder
     -> encoder parameters changed
     -> VAE encoder options dropped (i.e. dict(layers=3) -> {})
    >>> pprint(VAE(**config(0))._state)  # DenseEncoder -> RecurrentEncoder
    {'encoder': {'bidirectional': False,
                 'layers': 2,
                 'recurrent': {'nonlinearity': 'tanh'},
                 'recurrent_cls': 'model_challenge_cb_v3_5.GRU'},
     'encoder_cls': 'model_challenge_cb_v3_5.RecurrentEncoder'}

    bidirectional: False -> True
    >>> pprint(VAE(**config(1))._state)
    {'encoder': {'bidirectional': True,
                 'layers': 2,
                 'recurrent': {'nonlinearity': 'tanh'},
                 'recurrent_cls': 'model_challenge_cb_v3_5.GRU'},
     'encoder_cls': 'model_challenge_cb_v3_5.RecurrentEncoder'}

    recurrent_cls: GRU -> LSTM
    >>> pprint(VAE(**config(2))._state)
    {'encoder': {'bidirectional': True,
                 'layers': 2,
                 'recurrent': {'peephole': True},
                 'recurrent_cls': 'model_challenge_cb_v3_5.LSTM'},
     'encoder_cls': 'model_challenge_cb_v3_5.RecurrentEncoder'}

    Classical call:
    >>> pprint(VAE(RecurrentEncoder(layers=5))._state)
    {'encoder': {'bidirectional': False,
                 'layers': 5,
                 'recurrent': {'nonlinearity': 'tanh'},
                 'recurrent_cls': 'model_challenge_cb_v3_5.GRU'},
     'encoder_cls': 'model_challenge_cb_v3_5.RecurrentEncoder'}
    """
    tree = lambda: collections.defaultdict(tree)
    options = tree()
    if difficulty >= 0:
        options['encoder_cls'] = 'RecurrentEncoder'
    if difficulty >= 1:
        options['encoder']['bidirectional'] = True
    if difficulty >= 2:
        options['encoder']['recurrent_cls'] = 'LSTM'
        options['encoder']['recurrent']['peephole'] = True
    if difficulty >= 3:
        options['encoder_cls'] = 'DenseEncoder'

    return json.loads(json.dumps(options))


if __name__ == '__main__':
    print(config(0))
    print('VAE ', VAE())
    print('VAE state', VAE()._state)
    print('VAE fd', VAE(**{}))
    print('VAE fd0', VAE(**config(0)))
    print('VAE fd1', VAE(**config(1)))
    print('VAE fd2', VAE(**config(2)))
    print('VAE fd3', VAE(**config(3)))
    state = VAE(**config(2))._state
    print('state', state)
    print('VAE state', VAE(**state))
    print('state', state)
    print('VAE state', VAE(**state))
    print('VAE RecurrentEncoder state', VAE(RecurrentEncoder(layers=3)), VAE(RecurrentEncoder(layers=5)))
    # print(VAE.from_dict(config(3)))
