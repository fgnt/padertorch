"""Provide nested pytorch modules with JSON-serializable configuration.

An exemplary use-case is a compound module which consists of multiple modules
itself, e.g. a structured variational autoencoder or a farfield speech
enhancer.

You can instantiate a child of `Configurable` either with an `__init__`
resulting in an unparameterized module. If you instantiate it with
`from_config` you get a configured module.

If modules contain modules, look out for examples on how to override
`finalize_docmatic_config` (padertorch.contrib.examples.configurable).
In most cases, when you want to provide an instance as
a parameter to the `__init__` you can instead provide the parameters which were
used for that instance in your modified `finalize_docmatic_config`.

"""
import collections
import importlib
import inspect
import sys
from pathlib import Path

import paderbox as pb


class Configurable:
    """Allow subclasses to be configured automatically from JSON config files.

    Motivation: TODO: reproducability, adjustable

    Each factory is a configurable object (e.g. class constructor and
    function). The configurable object can be described with the factory object
    or the python path to the factory object and the kwargs of that function.
    To instantiate the object, the factory is called with the kwargs.
    Note: Inside the config might be again a configurable object,
    but later more about this.

    Example::

        >>> from IPython.lib.pretty import pprint
        >>> fix_doctext_import_class(locals())
        >>> import torch
        >>> def get_dense_layer(in_units, out_units):
        ...    l = torch.nn.Linear(in_units, out_units)
        ...    a = torch.nn.ReLU()
        ...    return torch.nn.Sequential(l, a)
        >>> config = {
        ...     'factory': get_dense_layer,
        ...     'in_units': 5,
        ...     'out_units': 3,
        ... }
        >>> Configurable.from_config(config)
        Sequential(
          (0): Linear(in_features=5, out_features=3, bias=True)
          (1): ReLU()
        )
        >>> str_factory = class_to_str(get_dense_layer)
        >>> str_factory
        'configurable.get_dense_layer'
        >>> config = {
        ...     'factory': 'configurable.get_dense_layer',
        ...     'in_units': 5,
        ...     'out_units': 3,
        ... }
        >>> Configurable.from_config(config)
        Sequential(
          (0): Linear(in_features=5, out_features=3, bias=True)
          (1): ReLU()
        )

    The value of the "factory" in the config is instantiated with in_units and
    out_units: `get_dense_layer(in_units=5, out_units=3)`

    When the operations get more difficult, it is common to use a class instead
    of a function. When the class inheritance from `Configurable` the class
    gets the attributes `get_config` and `from_config`. They make it easier to
    get the config and convert the config to an object.

    Example::

        >>> class DenseLayer(Configurable, torch.nn.Module):
        ...     def __init__(self, in_units, out_units=3):
        ...         super().__init__()
        ...         self.l = torch.nn.Linear(in_units, out_units)
        ...         self.a = torch.nn.ReLU()
        ...     def __call__(self, x):
        ...         return self.a(self.l(x))
        >>> config = {'factory': DenseLayer,
        ...           'in_units': 5,
        ...           'out_units': 7}
        >>> DenseLayer.from_config(config)
        DenseLayer(
          (l): Linear(in_features=5, out_features=7, bias=True)
          (a): ReLU()
        )

    Instead of manually defining each value in the config, it is recommended to
    use the `get_config` that reads the defaults from the signature (in the
    example above `out_units=3` from `DenseLayer.__init__`) and applies the
    provided updates to complete the kwargs for the instantiation.

    Example::

        >>> DenseLayer.get_config({'in_units': 5})
        {'factory': 'configurable.DenseLayer', 'in_units': 5, 'out_units': 3}
        >>> DenseLayer.get_config({'in_units': 5, 'out_units': 10})
        {'factory': 'configurable.DenseLayer', 'in_units': 5, 'out_units': 10}

    When a configurable object depends on other exchangeable configurable
    objects, it is recommended to overwrite the classmethod
    `finalize_dogmatic_config`. It gets as input the config with the defaults
    and the updates for that class. The function `finalize_dogmatic_config`
    should set all the remaining objects to fill the config that it contains
    all kwargs.

    Example::

        >>> class CustomizableDenseLayer(Configurable, torch.nn.Module):
        ...     @classmethod
        ...     def finalize_dogmatic_config(cls, config):
        ...         config['linear'] = {
        ...             'factory': torch.nn.Linear,
        ...             'out_features': 3,
        ...         }
        ...         if config['linear']['factory'] == torch.nn.Linear:
        ...             config['linear']['in_features'] = 5
        ...         config['activation'] = {'factory': torch.nn.ReLU}
        ...     def __init__(self, linear, activation):
        ...         super().__init__()
        ...         self.l = linear  # torch.nn.Linear(in_units, out_units)
        ...         self.a = activation  # torch.nn.ReLU()
        ...     def __call__(self, x):
        ...         return self.a(self.l(x))
        >>> config = CustomizableDenseLayer.get_config()
        >>> pprint(config)
        {'factory': 'configurable.CustomisableDenseLayer',
         'linear': {'factory': 'torch.nn.modules.linear.Linear',
          'in_features': 5,
          'out_features': 3,
          'bias': True},
         'activation': {'factory': 'torch.nn.modules.activation.ReLU',
          'inplace': False}}
        >>> CustomizableDenseLayer.from_config(config)
        CustomisableDenseLayer(
          (l): Linear(in_features=5, out_features=3, bias=True)
          (a): ReLU()
        )

    Note that the signature defaults from the nested configurable object
    `torch.nn.modules.linear.Linear` are also in the config.

    With the provided updates to `get_config` you can replace any value in the
    config. Note in the following example that the 'out_features' stays in the
    config, because it is outside the if in `finalize_dogmatic_config`, while
    'in_features' disappears because the config object that is given to
    `finalize_dogmatic_config` is a dogmatic dict. That means the updates have
    a higher priority than the assigned value. This behaviour is similar to
    Sacred[1].

    [1]
    https://sacred.readthedocs.io/en/latest/configuration.html#updating-config-entries.

    Example::

        >>> config = CustomizableDenseLayer.get_config(
        ...     updates={'linear': {
        ...         'factory': torch.nn.Bilinear,
        ...         'in1_features': 10,
        ...         'in2_features': 15,
        ...     }}
        ... )
        >>> pprint(config)
        {'factory': 'configurable.CustomisableDenseLayer',
         'linear': {'factory': 'torch.nn.modules.linear.Bilinear',
          'in1_features': 10,
          'in2_features': 15,
          'out_features': 3,
          'bias': True},
         'activation': {'factory': 'torch.nn.modules.activation.ReLU',
          'inplace': False}}
        >>> CustomizableDenseLayer.from_config(config)
        CustomisableDenseLayer(
          (l): Bilinear(in1_features=10, in2_features=15, out_features=3, bias=True)
          (a): ReLU()
        )

    Another use case for this behaviour are depended config entries (e.g.
    NN input size depends on selected input features).

    # TODO: This text ist outdated and needs to be reformulated
    # The values in the config are enforced to have the updated value.
    # (example below) i.e. when the update set 'activation' to be 'sigmoid'
    # the statement `config['activation'] = 'relu'` has no effect.
    # Parameters added in from_config

    Example::

        >>> class EncoderDecoder(Configurable, torch.nn.Module):
        ...     @classmethod
        ...     def finalize_dogmatic_config(cls, config):
        ...         # assume that the config only includes values that
        ...         # have defaults in the __init__ signiture
        ...         config['encoder']= {
        ...             'factory': DenseLayer,
        ...             'in_units': config['in_features'],
        ...             'out_units': 3
        ...         }
        ...         config['decoder'] = {
        ...             'factory': DenseLayer,
        ...             'in_units': config['encoder']['out_units'],
        ...             'out_units': 20
        ...         }
        ...     def __init__(self, encoder, decoder, in_features=5):
        ...         super().__init__()
        ...         self.net = torch.nn.Sequential(
        ...             encoder,
        ...             decoder
        ...         )
        ...         self.in_features = in_features
        ...     def __call__(self, x):
        ...         return self.net(x)
        >>> config = EncoderDecoder.get_config()
        >>> pprint(config)
        {'factory': 'configurable.EncoderDecoder',
         'encoder': {'factory': 'configurable.DenseLayer',
          'in_units': 5,
          'out_units': 3},
         'decoder': {'factory': 'configurable.DenseLayer',
          'in_units': 3,
          'out_units': 20},
         'in_features': 5}
        >>> EncoderDecoder.from_config(config)
        EncoderDecoder(
          (net): Sequential(
            (0): DenseLayer(
              (l): Linear(in_features=5, out_features=3, bias=True)
              (a): ReLU()
            )
            (1): DenseLayer(
              (l): Linear(in_features=3, out_features=20, bias=True)
              (a): ReLU()
            )
          )
        )

    When out_units of the encoder are updated in the config
    the in_units of the decoder are updated accordingly
    This behaviour is similar to a sacred config.

        >>> EncoderDecoder.from_config(EncoderDecoder.get_config(
        ...     updates={'encoder': {'out_units': 3}}
        ... ))
        EncoderDecoder(
          (net): Sequential(
            (0): DenseLayer(
              (l): Linear(in_features=5, out_features=3, bias=True)
              (a): ReLU()
            )
            (1): DenseLayer(
              (l): Linear(in_features=3, out_features=20, bias=True)
              (a): ReLU()
            )
          )
        )
    """

    # TODO: Drop get_signature?
    @classmethod
    def get_signature(cls):
        raise NotImplementedError('get_signature should no longer be used'
                                  'if additional defaults have to be'
                                  'specified use finalize_dogmatic_config')

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """Finalize the config. Should be overridden in subclass.

        🔥🔥 If you override this function you need to know what a        🔥🔥
        🔥🔥 dogmatic dict is. Within this method, Python behaves as if   🔥🔥
        🔥🔥 you are in a Sacred config.                                  🔥🔥

        TODO: doctext (for now see above Configurable doctext with the
              examples)
        """
        pass

    @classmethod
    def get_config(
            cls,
            updates=None,
    ):
        """Return the config as a JSON serializeable dict.

        Args:
            updates: Optional dict containing updates for the config.

        Returns:
            The serializeable config.
            If updates is not given, returns the default config.
            If updates is given, return the updated config.

        """
        if cls.__module__ == '__main__':
            # When a class is defined in the main script, it will be
            # __main__.<ModelName>, but it should be <script>.<ModelName>.
            # This fix is active when the script is called with
            # "python -m <script> ..."
            # but not when it is called with "python <script>.py ..."
            # pylint: disable=self-cls-assignment
            cls = import_class(class_to_str(cls))

        if updates is None:
            updates = {}
            config = {}
        elif isinstance(updates, _DogmaticConfig):
            raise ValueError(
                'get_config does not accept dogmatic dict as it does not need '
                'to be called within finalize_dogmatic_dict.'
            )
        else:
            config = _sacred_dogmatic_to_dict(updates)

        if 'factory' not in config:
            config['factory'] = cls
        else:
            config['factory'] = import_class(config['factory'])
            if inspect.isclass(config['factory']) \
                    and issubclass(config['factory'], Configurable):
                # When subclass of Configurable expect proper subclass
                assert issubclass(import_class(config['factory']), cls), (
                    config['factory'], cls)

        config = _DogmaticConfig.normalize(config)

        # Calculate the config and convert it to a nested dict structure
        config = _DogmaticConfig(config).to_dict()

        test_config(config, {})

        # For sacred make an inplace change to the update
        # (Earlier nessesary, now optional)
        updates.clear()
        updates.update(config)

        return updates

    @classmethod
    def from_config(
            cls,
            config,
    ) -> 'Configurable':
        """Produce a Configurable instance from a valid config."""
        # TODO: assert do not use defaults

        if isinstance(config, _DogmaticConfig):
            config = config.to_dict()  # if called in finalize_dogmatic dict
        assert 'factory' in config, (cls, config)
        if cls is not Configurable:
            assert issubclass(import_class(config['factory']), cls), \
                (config['factory'], cls)

        for key_tuple in pb.utils.nested.flatten(config, sep=None).keys():
            if 'cls' in key_tuple:
                from IPython.lib.pretty import pretty
                raise ValueError(
                    'Found the old key "cls" in the config.\n'
                    f'key path: {key_tuple}\n'
                    'Replace it with factory.\n'
                    f'{pretty(config)}'
                )

        new = config_to_instance(config)
        return new

    @classmethod
    def from_file(
            cls,
            config_path: Path,
            in_config_path: str = '',

            consider_mpi=False,
    ):
        """Instantiate the module from given config_file.

        Args:
            config_path:
            in_config_path:
            consider_mpi:
                If True and mpi is used, only read config_path and
                checkpoint_path once and broadcast the content with mpi.
                Reduces the io load.

        Returns:


        """
        config_path = Path(config_path).expanduser().resolve()

        assert config_path.is_file(), config_path

        assert config_path.is_file(), f'Expected {config_path} is file.'

        def load_config(config_path):
            if config_path.suffix == '.json':
                import json
                with config_path.open() as fp:
                    configurable_config = json.load(fp)
            elif config_path.suffix == '.yaml':
                import yaml
                with config_path.open() as fp:
                    configurable_config = yaml.safe_load(fp)
            else:
                raise ValueError(config_path)
            return configurable_config

        if consider_mpi:
            from paderbox.utils import mpi
            configurable_config = mpi.call_on_master_and_broadcast(
                load_config,
                config_path=config_path,
            )
        else:
            configurable_config = load_config(config_path=config_path)
        if config_path != '':
            for part in in_config_path.split('.'):
                configurable_config = configurable_config[part]
        return cls.from_config(configurable_config)


def test_config(config, updates):
    """Test if the config updates are valid."""
    sig = inspect.signature(import_class(config['factory']))

    # Remove default -> force completely described
    sig = sig.replace(
        parameters=[p.replace(
            default=inspect.Parameter.empty
        ) for p in sig.parameters.values()]
    )
    factory, kwargs = _split_factory_kwargs(config)
    try:
        bound_arguments: inspect.BoundArguments = sig.bind(
            **kwargs
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
                    default=kwargs.get(
                        p.name, inspect.Parameter.empty
                    ),
                ) for p in sig.parameters.values()]
            )
            raise TypeError(
                f'{factory} {ex}\n'
                f'Did you mean one of these {suggestions}?\n'
                f'Call signature: {sig_wo_anno}\n'
                f'Where\n'
                f'     kwargs.keys(): {kwargs.keys()}\n'
                f'     error msg: {ex}'
            ) from ex
        else:
            raise TypeError(
                f'The test whether the call {factory}(**kwargs) would '
                f'be successful failed.\n'
                f'Where\n'
                f'     kwargs: {kwargs}\n'
                f'     signature: {sig}\n'
                f'     updates: {updates}\n'
                f'     error msg: {ex}\n'
                f'     config: {config}'
            ) from ex

    # Guarantee that config is json serializable
    try:
        _ = pb.io.dumps_json(config)
        # _ = json.dumps(config)
    except TypeError as ex:
        from IPython.lib.pretty import pretty
        pretty_config = pretty(config)
        raise ValueError(
            'Invalid config.\n'
            'See above exception msg from json.dumps and '
            'below the sub config:\n'
            f'{pretty_config}'
        )


def fix_doctext_import_class(locals_dict):
    """Allow classes defined in a doctest to be imported.

    This is necessary because classes defined in a doctest can not not be
    imported.

    Use this function inside a doctest as
        >>> fix_doctext_import_class(locals())

    Example::
        >>> abc = 1
        >>> class Foo: pass
        >>> import_class(class_to_str(Foo))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        AttributeError: Module ... 'configurable' ... has no attribute Foo. ...

        >>> fix_doctext_import_class(locals())
        >>> import_class(class_to_str(Foo))
        <class 'configurable.Foo'>
    """
    cache = {}

    class_to_str_orig = class_to_str
    import_class_orig = import_class

    def class_to_str_fix(cls):
        nonlocal cache
        name = class_to_str_orig(cls)
        if (not isinstance(cls, str)) \
                and cls.__module__ == locals_dict['__name__']:
            cache[name] = cls
        return name

    def import_class_fix(name):
        if name in cache:
            return cache[name]
        else:
            return import_class_orig(name)

    # for code in the doctest
    locals_dict['import_class'] = import_class_fix
    locals_dict['class_to_str'] = class_to_str_fix

    # for the remaining code
    globals()['import_class'] = import_class_fix
    globals()['class_to_str'] = class_to_str_fix


def import_class(name: str):
    """Import the str and return the imported object.

    Opposite of class_to_str.

    >>> import padertorch
    >>> import_class(padertorch.Model)
    <class 'padertorch.base.Model'>
    >>> import_class('padertorch.Model')
    <class 'padertorch.base.Model'>

    """
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
    """Convert a class to an importable str.

    Opposite of import_class.

    >>> import padertorch
    >>> class_to_str(padertorch.Model)
    'padertorch.base.Model'
    >>> class_to_str('padertorch.Model')
    'padertorch.base.Model'

    TODO: fix __main__ for scripts in packages that are called with shell
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


def _split_factory_kwargs(config):
    kwargs = config.copy()
    factory = kwargs.pop('factory')
    return factory, kwargs


def config_to_instance(config):
    """Is called by `Module.from_config()`. If possible, use that directly.

    Args:
        config:

    Returns:

    """
    if isinstance(config, dict):
        if 'factory' in config:
            factory, kwargs = _split_factory_kwargs(config)
            new = import_class(factory)(
                **config_to_instance(kwargs)
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
    """A ChainMap that works on either nested dicts or DogmaticDicts.

    This class is similar to collections.ChainMap.
    Differences:
     - works on nested dictionaries
     - has a mutable_idx in the signature
        - mutable_idx == 0 => same behaviour as ChainMap
        - mutable_idx != 0 => works like a sacred.DogmaticDict
          - setting a value does not guarantee that getitem returns this value
            When a earlier mapping defines the value for that key, __getitem__
            will return that value

    Examples::
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
    >>> c['1']['1_2'] = 300  # will set the value
    >>> c['1']['1_2'] = 200  # will overwrite the value
    >>> pprint(c)
    NestedChainMap({'1': {'1_1': 1}, '2': {'2_2': 2}, '3': 3},
                   {'1': {'1_2': 200, '1_1': 100}, '2': {'2_1': 3, '2_2': 4}})
    >>> c['1']['1_2'] = 300  # will set the value
    >>> c['1']['1_2'] = 200  # will overwrite the value
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
        """Construct NestedChainMap from an arbitrary number of mappings."""
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

        is_mapping = [
            isinstance(m[item], collections.Mapping)
            for m in self.maps
            if item in m
        ]
        if any(is_mapping) and is_mapping[0]:
            if not all(is_mapping):
                for m in self.maps:
                    if item in m:
                        if not isinstance(m[item], collections.Mapping):
                            # delete the value, because it has the wrong type
                            del m[item]
            #     from IPython.lib.pretty import pretty
            #     raise Exception(
            #         f'Tried to get the value for the key "{item}" in this '
            #         f'NestedChainMap.\n'
            #         f'Expect that all values in the maps are dicts '
            #         f'or none is a dict:\n'
            #         f'{pretty(self)}'
            #     )
            m: dict

            def my_setdefault(mapping, key, default):
                if key in mapping:
                    return mapping[key]
                else:
                    mapping[key] = default
                    return default

            sub = self.__class__(*[
                my_setdefault(m, item, {})
                for m in self.maps
            ], mutable_idx=self.mutable_idx)

            self.subs[item] = sub
            return sub

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


def _sacred_dogmatic_to_dict(config):
    """Perform a mutable-safe conversion of nested sacred DogmaticDict.

    Takes a nested structure as input and sets all values that are defined
    from sacred as "fixed". Returns the nested structure with all updates
    applied from sacred.
    """
    import sacred.config.custom_containers

    if isinstance(config, sacred.config.custom_containers.DogmaticDict):
        merge_keys = [
            *config.keys(),
            *_sacred_dogmatic_to_dict(config.fallback).keys(),
        ]
        return pb.utils.nested.nested_merge(
            {k: _sacred_dogmatic_to_dict(config[k]) for k in merge_keys},
            {k: _sacred_dogmatic_to_dict(v) for k, v in config.fixed.items()}
        )
    elif isinstance(config, dict):
        return {k: _sacred_dogmatic_to_dict(v) for k, v in config.items()}
    else:
        return config


class _DogmaticConfig:
    """This class is an implementation detail of Configurable."""

    @staticmethod
    def get_signature(factory):
        """Check signature of the factory.

        If parameters have defaults, return these in a dictionary.

        Returns:

        """
        sig = inspect.signature(factory)
        defaults = {}
        param: inspect.Parameter
        for name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                defaults[name] = param.default
        return defaults

    @classmethod
    def _force_factory_type(cls, factory):
        """Convert factory from str to cls object if necessary.

        This is a placeholder until it is finally decided if the factory
        should be a str or a class in finalize_docmatic_config.

        Options:
         - class
         - str
         - Object that can be compared with str and class

        This function should later be deleted.
        """
        return import_class(factory)

    @classmethod
    def normalize(cls, dictionary):
        """Normalize the nested dictionary.

        - The value of factory keys are forced to be the object and
          not the str. (i.e. import_class).
        - pathlib.Path to str

        >>> import torch
        >>> from IPython.lib.pretty import pprint
        >>> pprint(_DogmaticConfig.normalize({
        ...     'model': {'factory': 'torch.nn.Linear'},
        ...     'storage_dir': Path('abc')
        ... }), max_width=79-8)
        {'model': {'factory': torch.nn.modules.linear.Linear},
         'storage_dir': 'abc'}
        """
        if isinstance(dictionary, collections.Mapping):
            if 'factory' in dictionary:
                dictionary['factory'] = cls._force_factory_type(
                    dictionary['factory']
                )
            dictionary = {
                k: cls.normalize(v)
                for k, v in dictionary.items()
            }
        elif isinstance(dictionary, (tuple, list)):
            dictionary = [
                cls.normalize(v)
                for v in dictionary
            ]
        elif isinstance(dictionary, Path):
            dictionary = str(dictionary)
        else:
            pass

        return dictionary

    def __init__(
            self,
            *maps,
            mutable_idx=-1
    ):
        """
        Takes maps as input (usually dicts) and store them as a NestedChainMap
        with one further dict at the end.
        The mutable_idx is the index for that dict that this class writes
        everything that is set with __setitem__. That means if a dict with a
        lower index already defines a value for that key, the new value is
        ignored.

        For __getitem__ see __getitem__ doctext.

        Args:
            *maps:
            mutable_idx:

        """
        assert len(maps) >= 1, maps
        maps = list(maps)+ [{}]

        self.data = NestedChainMap(
            *maps,
            mutable_idx=mutable_idx,
        )

        if 'factory' in self.data:
            self._check_redundant_keys(
                'padertorch.Configurable.get_config(updates=...) got an '
                f'unexpected keyword argument in updates for '
                f'{self.data["factory"]}.\n'
                'See details below.\n'
            )

    def get_sub_config(self, key, mutable_idx=None):
        if mutable_idx is None:
            mutable_idx = self.data.mutable_idx
        sub = self.data[key]

        if isinstance(sub, NestedChainMap):
            return self.__class__(
                *sub.maps, mutable_idx=mutable_idx
            )
        else:
            raise KeyError(key)

    def _key_candidates(self):
        if 'factory' in self.data:
            factory = import_class(self.data['factory'])
            parameters = inspect.signature(factory).parameters.values()
            p: inspect.Parameter

            parameter_names = tuple(['factory']) + tuple([
                p.name
                for p in parameters
                if p.kind in [
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ]
            ])

            if inspect.Parameter.VAR_KEYWORD in [p.kind for p in parameters]:
                parameter_names += tuple(self.data.keys())

                # Removing duplicates in lists
                # https://stackoverflow.com/a/7961390/5766934
                parameter_names = tuple(
                    collections.OrderedDict.fromkeys(parameter_names)
                )
            return parameter_names
        else:
            return tuple(self.data.keys())

    def keys(self):
        return tuple(self.data.keys())

    def _check_redundant_keys(self, msg):
        assert 'factory' in self.data
        factory = import_class(self.data['factory'])
        parameters = inspect.signature(factory).parameters.values()
        p: inspect.Parameter

        if inspect.Parameter.VAR_KEYWORD in [p.kind for p in parameters]:
            pass
        else:
            parameter_names = set([
                p.name
                for p in parameters
                if p.kind in [
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ]
            ]) | {'factory'}

            redundant_keys = set(self.data.keys()) - parameter_names

            if len(redundant_keys) != 0:
                from IPython.lib.pretty import pretty
                raise Exception(
                    f'{msg}\n'
                    f'Too many keywords for the factory {factory}.\n'
                    f'Redundant keys: {redundant_keys}\n'
                    f'Signature: {inspect.signature(factory)}\n'
                    f'Current config with fallbacks:\n{pretty(self.data)}'
                )

    def __contains__(self, item):
        raise NotImplementedError(
            f'{self.__class__.__name__}.__contains__\n'
            f'Use `key in {self.__class__.__name__}.keys()`\n instead of\n'
            f'`key in {self.__class__.__name__}`'
        )

    def __setitem__(self, key, value):
        self.data[key] = self.normalize(value)

        if 'factory' in self.data.keys():
            self._check_redundant_keys(
                'Tried to set an unexpected keyword argument for '
                f'{self.data["factory"]} in finalize_dogmatic_config.\n'
                'See details below and stacktrace above.\n'
            )

    def update(self, dictionary: dict, **kwargs):
        dictionary.update()
        for key, value in dictionary.items():
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def setdefault(self, key, default):
        """
        If key is in the _DogmaticConfig, return its value.
        If not, insert key with a value of default and return default.
        """
        if key not in self.data.keys():
            self[key] = default
        return self[key]

    def _update_factory_kwargs(self):
        assert 'factory' in self.data, self.data

        # Force factory to be the class/function
        factory = import_class(self.data['factory'])

        # Freeze the mutable_idx (i.e. all updates to the config of
        # this level)
        mutable_idx_old = self.data.mutable_idx
        self.data.mutable_idx = len(self.data.maps) - 1

        # Get the defaults from the factory signature
        defaults = self.get_signature(factory)
        for k, v in defaults.items():
            self[k] = v

        if hasattr(factory, 'finalize_dogmatic_config'):
            try:
                factory.finalize_dogmatic_config(config=self)
            except TypeError as e:
                if 'finalize_dogmatic_config() missing 1 ' \
                   'required positional argument' in str(e):
                    raise TypeError(
                        f'{factory.__name__}.{e} \n'
                        f'finalize_dogmatic_config has to be'
                        f' a classmethod') from e
                else:
                    raise

        delta = set(self.data.keys()) - set(self._key_candidates())

        if len(delta) > 1:
            # (delta, self.data.keys(), parameter_names)
            from IPython.lib.pretty import pretty
            raise Exception(
                f'Too many keywords for the factory {factory}.\n'
                f'Delta: {delta}\n'
                f'signature: {inspect.signature(factory)}\n'
                f'current config with fallbacks:\n{pretty(self.data)}'
            )

        self.data.mutable_idx = mutable_idx_old

    def __getitem__(self, key):
        """Return the value for the key.

        When the value is not a dict, directly return the value.
        When the value is a dict and does not contain "factory" as key,
        return a _DogmaticConfig instance for that dict
        (i.e. take the sub dict of each dict is self.data.maps).
        When the dict contains the key "factory", freeze the mutable_idx of
        this _DogmaticConfig instance and update the kwargs
        (i.e. get the defaults from the signature and call
        finalize_dogmatic_config if it exists.)

        """
        if 'factory' in self.data \
                and key != 'factory' \
                and self.data.mutable_idx != (len(self.data.maps) - 1):
            self._update_factory_kwargs()

        if 'cls' in self._key_candidates():
            from IPython.lib.pretty import pretty
            factory = self.data['cls']
            factory_str = class_to_str(factory)
            raise Exception(
                f'Got the old key "cls" (value: {factory_str}).\n'
                f'Use the new key: factory\n'
                f'Signature: {inspect.signature(factory)}\n'
                f'Current config with fallbacks:\n{pretty(self.data)}'
            )

        if key in self.data:
            try:
                value = self.get_sub_config(key)
            except KeyError:
                value = self.data[key]
        else:
            raise KeyError(key)

        return value

    def to_dict(self):
        """Export the Configurable object to a dict."""
        result_dict = {}
        if 'factory' in self._key_candidates():
            self._update_factory_kwargs()
        for k in self._key_candidates():
            try:
                v = self[k]
            except KeyError as ex:
                from IPython.lib.pretty import pretty
                if 'factory' in self._key_candidates() and k != 'factory':
                    # KeyError has a bad __repr__, use Exception
                    missing_keys = set(self._key_candidates()) - set(self.data.keys())
                    raise Exception(
                        f'KeyError: {k}\n'
                        f'signature: {inspect.signature(self["factory"])}\n'
                        f'missing keys: {missing_keys}\n'
                        f'self:\n{pretty(self)}'
                    ) from ex
                else:
                    # KeyError has a bad __repr__, use Exception
                    # Can this happen?
                    raise Exception(
                        f'{k}\n'
                        f'signature: {inspect.signature(self["factory"])}'
                        f'self:\n{pretty(self)}'
                    ) from ex

            if k == 'factory':
                v = class_to_str(v)
            if isinstance(v, self.__class__):
                v = v.to_dict()
            result_dict[k] = v

        return result_dict

    def __str__(self):
        # Keep the str representer simple (only one dict)
        return f'{self.__class__.__name__}({self.data.to_dict()})'

    def __repr__(self):
        # Keep the repr representer verbose (show all dicts)
        maps = ', '.join([repr(m) for m in self.data.maps])
        return (f'{self.__class__.__name__}({maps},'
                f' mutable_idx={self.data.mutable_idx})')

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text(f'{self.__class__.__name__}(...)')
        else:
            name = self.__class__.__name__
            pre, post = f'{name}(', ')'
            with pp.group(len(pre), pre, post):
                for idx, m in enumerate(self.data.maps):
                    if idx:
                        pp.text(',')
                        pp.breakable()
                    pp.pretty(m)
                pp.text(',')
                pp.breakable()
                pp.text(f'mutable_idx={self.data.mutable_idx})')
