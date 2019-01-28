"""Provide nested pytorch modules with JSON-serializable configuration.

An exemplary use-case is a compound module which consists of multiple modules
itself, e.g. a structured variational autoencoder or a farfield speech
enhancer.

You can instantiate a child of `Configurable` either with an `__init__`
resulting in an unparameterized module. If you instantiate it with
`from_config` you get a configured module.

If modules contain modules, look out for examples on how to override
`get_signature`. In most cases, when you want to provide an instance as
a parameter to the `__init__` you can instead provide the parameters which were
used for that instance in your modified `get_signature`.

"""
import sys
import json
import inspect
import importlib
import collections
from numbers import Number
from typing import Union
from pathlib import Path
from collections import OrderedDict


class Configurable:
    """

    Example::

        from padertorch.configurable import Configurable
        class MyModule(Configurable):
            def __init__(self, a=1):
                pass
        MyModule.get_config()

    Results in::

        {'cls': 'MyModule', 'kwargs': {'a': 1}}

    """
    _config = None

    @property
    def config(self):
        if self._config is None:
            p_name = f'{Configurable.__module__}.{Configurable.__qualname__}'
            name = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
            raise Exception(
                f'The config property of a {p_name} object\n'
                f'is only available, when the object is '
                f'produced from "from_config".\n'
                f'You tried to get it for an instance of {name}.'
            )
        return self._config

    @config.setter
    def config(self, value):
        if self._config is None:
            self._config = value
        else:
            p_name = f'{Configurable.__module__}.{Configurable.__qualname__}'
            name = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
            raise Exception(
                f'The config property of a {p_name} object\n'
                f'can only be set once.\n'
                f'You tried to set it for an instance of {name}.'
            )

    @classmethod
    def get_signature(cls):
        """
        Checks signature of __init__. If parameters have defaults, return
        these in a dictionary.

        Returns:

        """

        sig = inspect.signature(cls)
        defaults = {}
        param: inspect.Parameter
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
        return defaults

    @classmethod
    def get_config(
            cls,
            updates=None,
            out_config=None,
    ):
        """
        Provides configuration to allow instantiation with
        module = Module.from_config(Module.get_config())

        Args:
            updates: dict with values to be modified w.r.t. defaults.
                Sub-configurations are updated accordingly if top-level-keys
                are changed. An Exception is raised if update_dict has unused
                entries.
            out_config: Provide an empty dict which is a Sacred config local
                variable. This allow Sacred to influence dependent parameters.

        Returns: Config

        """
        config = out_config

        if cls.__module__ == '__main__':
            # When a class is defined in the main script, it will be
            # __main__.<ModelName>, but it should be <script>.<ModelName>.
            # This fix it, when the script is called with
            # "python -m <script> ..."
            # but not when it is called with "python <script>.py ..."
            cls = import_class(class_to_str(cls))

        if config is None:
            config = {
                'cls': class_to_str(cls)
            }
        elif 'cls' not in config:
            config['cls'] = class_to_str(cls)
        else:
            config['cls'] = class_to_str(config['cls'])

        # This assert is for sacred that may change values in the config dict.
        if inspect.isclass(import_class(config['cls'])) \
                and issubclass(import_class(config['cls']), Configurable):
            # When subclass of Configurable expect proper subclass
            assert issubclass(import_class(config['cls']), cls), (config['cls'], cls)

        if hasattr(import_class(config['cls']), 'get_signature'):
            defaults = import_class(config['cls']).get_signature()
        else:
            defaults = Configurable.get_signature.__func__(
                import_class(config['cls'])
            )

        config['kwargs'] = {
            **recursive_class_to_str(defaults),
            **config.get('kwargs', {}),
            **config.get(config['cls'], {}),
        }

        for key in list(config.keys()):
            if key not in ['cls', 'kwargs']:
                del config[key]

        if updates is None:
            updates = dict()

        try:
            update_config(config['kwargs'], updates)
        except ConfigUpdateException as ex:
            raise Exception(
                f'{cls.__module__}.{cls.__qualname__}: '
                f'Updates that are not used anywhere: {ex}'
            )

        test_config(config, updates)

        return config

    @classmethod
    def get_config_v2(
            cls,
            updates=None,
            out_config=None,
    ):

        if cls.__module__ == '__main__':
            # When a class is defined in the main script, it will be
            # __main__.<ModelName>, but it should be <script>.<ModelName>.
            # This fix it, when the script is called with
            # "python -m <script> ..."
            # but not when it is called with "python <script>.py ..."
            cls = import_class(class_to_str(cls))
        
        config = DogmaticConfig.normalize({
            'cls': cls,
            'kwargs': updates
        }, kwargs_normalize=False)

        if out_config is not None:
            config = NestedChainMap(
                DogmaticConfig.sacred_dogmatic_to_dict(out_config),
                config,
            ).to_dict()

        config = DogmaticConfig(config).to_dict()

        out_config.clear()
        out_config.update(config)

        test_config(config, updates)

        return out_config

    @classmethod
    def from_config(
            cls,
            config,
    ) -> 'Configurable':
        # assert do not use defaults
        assert 'cls' in config, (cls, config)
        assert issubclass(import_class(config['cls']), cls), \
            (config['cls'], cls)
        new = config_to_instance(config)
        # new.config = config
        return new


def test_config(config, updates):
    # Test if the kwargs are valid
    sig = inspect.signature(import_class(config['cls']))

    # Remove default -> force completely described
    sig = sig.replace(
        parameters=[p.replace(
            default=inspect.Parameter.empty
        ) for p in sig.parameters.values()]
    )
    try:
        bound_arguments: inspect.BoundArguments = sig.bind(
            **config['kwargs']
        )
        bound_arguments.apply_defaults()
    except TypeError as ex:
        unexpected_keyword = 'got an unexpected keyword argument '
        if unexpected_keyword in str(ex):
            _, keyword = str(ex).split(unexpected_keyword)
            keyword = keyword.strip().strip("'")
            available_keys = sig.parameters.keys()

            import difflib
            suggestions = difflib.get_close_matches(
                keyword, available_keys,
            )

            # CB: Is it better to print with assigned values or without?
            sig_wo_anno = sig.replace(
                parameters=[p.replace(
                    annotation=inspect.Parameter.empty,
                    default=config['kwargs'].get(
                        p.name, inspect.Parameter.empty
                    ),
                ) for p in sig.parameters.values()]
            )
            raise TypeError(
                f'{config["cls"]} {ex}\n'
                f'Did you mean one of these {suggestions}?\n'
                f'Call signature: {sig_wo_anno}\n'
                f'Where\n'
                f'     kwargs.keys(): {config["kwargs"].keys()}\n'
                f'     error msg: {ex}'
            ) from ex
        else:
            raise TypeError(
                f'The test, if the call {config["cls"]}(**kwargs) would '
                f'be successful, failed.\n'
                f'Where\n'
                f'     kwargs: {config["kwargs"]}\n'
                f'     signature: {sig}\n'
                f'     updates: {updates}\n'
                f'     error msg: {ex}'
            ) from ex

    # Guarantee that config is json serializable
    try:
        _ = json.dumps(config)
    except TypeError as ex:
        from IPython.lib.pretty import pretty
        pretty_config = pretty(config)
        raise ValueError(
            'Invalid config.\n'
            'See above exception msg from json.dumps and '
            'below the sub config:\n'
            f'{pretty_config}'
        )

def import_class(name: str):
    if not isinstance(name, str):
        return name
    splitted = name.split('.')
    module_name = '.'.join(splitted[:-1])
    if module_name == '':
        module_name = '__main__'
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
    try:
        return getattr(module, splitted[-1])
    except AttributeError as ex:
        raise AttributeError(
            f'Module {module} has no attribute {splitted[-1]}.'
            f' Original: {ex!r}'
        )


def resolve_main_python_path() -> str:
    """Can only resolve, if you run scripts with `python -m`."""
    return getattr(sys.modules['__main__'].__spec__, 'name', '__main__')


def class_to_str(cls):
    """
    >>> import padertorch
    >>> class_to_str(padertorch.Model)
    'padertorch.base.Model'
    >>> class_to_str('padertorch.Model')
    'padertorch.base.Model'

    ToDo: fix __main__ for scripts in packages that are called with shell
          path (path/to/script.py) and not python path (path.to.script).
    """

    if isinstance(cls, str):
        cls = import_class(cls)

    module = cls.__module__

    if module == '__main__':
        # Try to figure out the module.
        # Could be done, when the script is started with "python -m ..."
        module = resolve_main_python_path()

    if module != '__main__':
        return f'{module}.{cls.__qualname__}'
    else:
        return f'{cls.__qualname__}'


def recursive_class_to_str(
        config: Union[str, Number, None, dict, list, tuple, Path]
) -> Union[str, Number, dict, list, tuple]:
    """
    Recursively traverses a config and transforms all class or
    Path instances into their string representation while passing allowed data
    types and failing otherwise.

    changes Configurable Objects to import path string
    changes Path to str

    :param config: The raw config, maybe containing class types and Path
        instances.
    :return: a JSON serializable version of the config.
    >>> from padertorch import Model
    >>> recursive_class_to_str([{'cls': 'padertorch.Model'}])
    [{'cls': 'padertorch.base.Model'}]
    >>> recursive_class_to_str([{'cls': Model, Model: {}}])
    [{'cls': 'padertorch.base.Model', 'padertorch.base.Model': {}}]
    >>> recursive_class_to_str([None])
    [None]
    """
    if config is None or isinstance(config, (str, Number)):
        return config
    elif isinstance(config, dict):
        if 'cls' in config:
            return {
                k if isinstance(k, str) else class_to_str(k):
                class_to_str(v) if k == 'cls' else recursive_class_to_str(v)
                for k, v in config.items()
            }
        else:
            return config.__class__({
                k: recursive_class_to_str(v)
                for k, v in config.items()
            })
    elif isinstance(config, (list, tuple)):
        return config.__class__([
            recursive_class_to_str(l)
            for l in config
        ])
    elif isinstance(config, Path):
        return str(config)
    else:
        raise TypeError('config is of unusable type'
                        f' {type(config)}.\n'
                        f' config:\n{config!r}')


class ConfigUpdateException(Exception):
    pass


def update_config(config, updates):
    """

    :param config: config dict
    :param updates: updates dict.
    :return:
    """
    # TODO: tuple and lists (e.g. Trainer models and optimizers)
    if 'cls' in config or 'cls' in updates:
        if 'cls' in updates:
            config['cls'] = class_to_str(updates['cls'])
        sub_updates = {
            **updates.get('kwargs', {}),
            **updates.get(config['cls'], {}),
            **updates.get(import_class(config['cls']), {}),
        }

        cls = import_class(config['cls'])
        if hasattr(cls, 'get_config'):
            # inplace
            print(cls)
            cls.get_config(
                updates=sub_updates,
                out_config=config,
            )
        else:
            Configurable.get_config.__func__(
                cls,
                updates=sub_updates,
                out_config=config,
            )
    else:
        for key in sorted(list(config.keys())):
            if isinstance(config[key], dict):
                update_config(
                    config[key],
                    updates.pop(key) if updates and key in updates else dict(),
                )
            elif updates and key in updates:
                new_value = updates.pop(key)
                if isinstance(new_value, dict):
                    config[key] = {}
                    update_config(
                        config[key],
                        new_value,
                    )
                else:
                    config[key] = new_value

        for key in list(updates.keys()):
            if isinstance(updates[key], dict):
                config[key] = {}
                update_config(
                    config[key],
                    updates.pop(key),
                )
            else:
                config[key] = updates.pop(key)


def config_to_instance(config):
    if isinstance(config, dict):
        if 'cls' in config:
            assert 'kwargs' in config, config
            assert len(config) == 2, (config.keys(), config)
            new = import_class(config['cls'])(
                **config_to_instance(config['kwargs'])
            )
            try:
                new.config = config
            except AttributeError:
                pass
            return new
        else:
            d = config.__class__()
            for k, v in config.items():
                d[k] = config_to_instance(v)
            return d
    elif isinstance(config, (tuple, list)):
        return config.__class__([
            config_to_instance(l) for l in config
        ])
    else:
        return config


class NestedChainMap(collections.ChainMap):
    """

    This class is similar to collections.ChainMap.
    Differences:
     - works on nested dictionaries
     - has a mutable_idx in the signature
        - mutable_idx == 0 => same behaviour as ChainMap
        - mutable_idx != 0 => works like a sacred.DogmaticDict
          - setting a value does not guarantee that getitem returns this value
            When a earlier mapping defines the value for that key, __getitem__
            will return that value


    >>> from IPython.lib.pretty import pprint
    >>> a = {'1': {'1_1': 1}, '2': {'2_2': 2}, '3': 3}
    >>> b = {'1': {'1_2': 3}, '2': {'2_1': 3, '2_2': 4}}
    >>> c = NestedChainMap(a, b, mutable_idx=-1)
    >>> pprint(c)
    NestedChainMap({'1': {'1_1': 1}, '2': {'2_2': 2}, '3': 3},
                   {'1': {'1_2': 3}, '2': {'2_1': 3, '2_2': 4}})
    >>> pprint(c.to_dict())
    {'1': {'1_2': 3, '1_1': 1}, '2': {'2_1': 3, '2_2': 2}, '3': 3}
    >>> c['1']['1_1'] = 100  # will be ignored
    >>> c['1']['1_2'] = 200  # will set the value
    >>> pprint(c)
    NestedChainMap({'1': {'1_1': 1}, '2': {'2_2': 2}, '3': 3},
                   {'1': {'1_2': 200, '1_1': 100}, '2': {'2_1': 3, '2_2': 4}})
    >>> pprint(c.to_dict())
    {'1': {'1_2': 200, '1_1': 1}, '2': {'2_1': 3, '2_2': 2}, '3': 3}
    """
    def __init__(
            self,
            *maps,
            mutable_idx=0,
    ):
        self.subs = {}
        super().__init__(*maps)
        self.mutable_idx = mutable_idx

    def __delitem__(self, key):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def __iter__(self):
        # Python 3.7 ChainMap.__iter__
        d = {}
        for mapping in reversed(self.maps):
            d.update(mapping)  # reuses stored hash values if possible
        return iter(d)

    def __setitem__(self, key, value):
        try:
            self.maps[self.mutable_idx][key] = value
        except IndexError as e:
            raise IndexError(
                f'{e}\n'
                '\n'
                f'self.set_idx: {self.mutable_idx}\n'
                f'len(self.maps): {len(self.maps)}\n'
                f'key: {key}\n'
                f'value: {value}\n'
                f'self: {self}'
            )

    def __getitem__(self, item):

        if item in self.subs:
            # short circuit
            return self.subs[item]

        is_dict_list = [
            isinstance(m[item], (dict, self.__class__))
            for m in self.maps
            if item in m
        ]
        if any(is_dict_list):
            assert all(is_dict_list), (item, is_dict_list, self.maps)
            m: dict

            def my_setdefault(mapping, key, default):
                if key in mapping:
                    return mapping[key]
                else:
                    mapping[key] = default
                    return default

            sub = self.__class__(*[
                my_setdefault(m, item, {}) for m in self.maps
            ], mutable_idx=self.mutable_idx)

            self.subs[item] = sub
            return sub
        else:
            # super().__getitem__ is to expensive
            for mapping in self.maps:
                if item in mapping:
                    return mapping[item]
            raise KeyError(item)

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, self.__class__) else v
            for k, v in self.items()
        }

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            name = self.__class__.__name__
            pre, post = f'{name}(', ')'
            with p.group(len(pre), pre, post):
                for idx, m in enumerate(self.maps):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(m)


class DogmaticConfig:

    @classmethod
    def sacred_dogmatic_to_dict(cls, config):
        import sacred.config.custom_containers
        import paderbox as pb

        if isinstance(config, sacred.config.custom_containers.DogmaticDict):
            return pb.utils.nested.nested_merge(
                {
                    k: cls.sacred_dogmatic_to_dict(config[k])
                    for keys in [
                    config.keys(),
                    cls.sacred_dogmatic_to_dict(config.fallback).keys(),
                ]
                    for k in keys
                }, {
                    k: cls.sacred_dogmatic_to_dict(v)
                    for k, v in config.fixed.items()
                }
            )
        elif isinstance(config, dict):
            return {
                k: cls.sacred_dogmatic_to_dict(v)
                for k, v in config.items()
            }
        else:
            return config

    @staticmethod
    def get_signature(cls: Configurable):
        if hasattr(cls, 'get_signature'):
            return cls.get_signature()
        else:
            return Configurable.get_signature.__func__(
                cls
            )

    @classmethod
    def normalize(cls, dictionary, kwargs_normalize=True):
        """
        Normalize the nested dictionary.

        The value of cls keys are forced to have the type str
        (i.e. class_to_str).

        When a dict has the cls key it also have the kwargs key and the cls
        value as key (e.g. {'cls': 'MyClass', 'kwargs': {}, 'MyClass': {}}).

        If kwargs_normalize is True:
        Description on the example
            {'cls': 'MyClass', 'kwargs': {}, 'MyClass': {}}
            Each key value in kwargs that is missing in MyClass is set in the
            MyClass dict.

        """
        if 'cls' in dictionary:
            if not isinstance(dictionary['cls'], str):
                dictionary['cls'] = class_to_str(dictionary['cls'])
            for key in tuple(dictionary.keys()):
                if not isinstance(key, str):
                    str_key = class_to_str(key)
                    dictionary[str_key] = dictionary.pop(key)

        if 'cls' in dictionary:
            if 'kwargs' not in dictionary:
                dictionary['kwargs'] = {}

            if kwargs_normalize:
                cls_str = dictionary['cls']

                if cls_str not in dictionary:
                    dictionary[cls_str] = {}

                if 'kwargs' not in dictionary:
                    dictionary['kwargs'] = {}
                for k in set(dictionary.keys()) - {'cls', 'kwargs'}:
                    # Keep the dictionary reference and do an inplace update
                    # (fallback)
                    def nested_default(d, fallback):
                        # Similar to
                        # dict.setdefault(key[, default])
                        # but for nested
                        for k, v in fallback.items():
                            if k in d:
                                if isinstance(d[k], dict) and isinstance(v, dict):
                                    nested_default(d[k], v)
                                else:
                                    pass
                            else:
                                d[k] = v

                    nested_default(dictionary[k], dictionary['kwargs'])

                    # NestedChainMap is to expensive here
                    # dictionary[k] = NestedChainMap(
                    #     dictionary[k], dictionary['kwargs']
                    # )

        for v in dictionary.values():
            if isinstance(v, dict):
                cls.normalize(v, kwargs_normalize=kwargs_normalize)

        return dictionary

    def __init__(
            self,
            *maps,
            mutable_idx=-1
    ):
        self.mutable_idx = mutable_idx

        assert len(maps) >= 1, maps
        maps = [
            self.normalize(m, kwargs_normalize=True)
            for m in maps
        ] + [{}]

        self.data = NestedChainMap(*maps,
                                   mutable_idx=mutable_idx,
                                   )

    def get_sub_config(self, key, mutable_idx=None):
        if mutable_idx is None:
            mutable_idx = self.mutable_idx
        sub = self.data[key]

        if isinstance(sub, NestedChainMap):
            return self.__class__(
                *sub.maps, mutable_idx=mutable_idx
            )
        else:
            raise KeyError(key)

    def keys(self):
        if 'cls' in self.data:
            return tuple([
                k
                for k in self.data.keys()
                if k in ['cls', 'kwargs']
            ])
        else:
            return tuple(self.data.keys())

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self.normalize(value, kwargs_normalize=True)

        if 'cls' in self.data:
            if key == 'kwargs':
                for k in set(self.data.keys()) - {'cls', 'kwargs'}:
                    self.data[k] = value
            elif key == 'cls':
                pass
            elif key not in self.data:
                # ToDo: test this
                import copy
                for m in self.data.maps:
                    m[key] = copy.deepcopy(m['kwargs'])

                raise NotImplementedError(key, self.data)
            else:
                pass

        self.data[key] = value

    def __getitem__(self, key):
        if 'cls' in self.data:
            cls_str = self.data['cls']

            if key == 'cls':
                value = cls_str
            elif key == 'kwargs':
                # Changes 'kwargs' to cls str
                key = cls_str

                cls = import_class(cls_str)
                defaults = self.get_signature(cls)

                _ = self.get_sub_config(key)

                # Freeze the mutable_idx.
                # => changes from get_signature and update_config go to this
                #    level of the config
                kwargs = self.__class__(
                    *self.data[key].maps,
                    mutable_idx=len(self.data.maps) - 1,
                )

                for k, v in defaults.items():
                    kwargs[k] = v

                if hasattr(cls, 'update_config'):
                    cls.update_config(kwargs)

                # ToDo: assert valid kwargs

                value = self.get_sub_config(key)
            else:
                raise ValueError(key)
        elif key in self.data:
            try:
                value = self.get_sub_config(key)
            except KeyError:
                value = self.data[key]
        else:
            raise KeyError(key)
        return value

    def to_dict(self):
        if 'cls' in self.keys():
            d = {
                'cls': self['cls'],
                'kwargs': self['kwargs'].to_dict()
            }
        else:
            d = {}
            for k in self.keys():
                v = self[k]
                if isinstance(v, self.__class__):
                    v = v.to_dict()
                d[k] = v

        return d
