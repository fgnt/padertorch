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
import sys
import builtins
import os
import collections
import functools
import importlib
import inspect
import dataclasses
from pathlib import Path
import copy

import paderbox as pb

# pylint: disable=import-outside-toplevel


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
        'padertorch.configurable.get_dense_layer'
        >>> config = {
        ...     'factory': 'padertorch.configurable.get_dense_layer',
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
        {'factory': 'padertorch.configurable.DenseLayer', 'in_units': 5, 'out_units': 3}
        >>> DenseLayer.get_config({'in_units': 5, 'out_units': 10})
        {'factory': 'padertorch.configurable.DenseLayer', 'in_units': 5, 'out_units': 10}

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
        {'factory': 'padertorch.configurable.CustomizableDenseLayer',
         'linear': {'factory': 'torch.nn.modules.linear.Linear',
          'in_features': 5,
          'out_features': 3,
          'bias': True,
          'device': None,
          'dtype': None},
         'activation': {'factory': 'torch.nn.modules.activation.ReLU',
          'inplace': False}}
        >>> CustomizableDenseLayer.from_config(config)
        CustomizableDenseLayer(
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
        {'factory': 'padertorch.configurable.CustomizableDenseLayer',
         'linear': {'factory': 'torch.nn.modules.linear.Bilinear',
          'in1_features': 10,
          'in2_features': 15,
          'out_features': 3,
          'bias': True,
          'device': None,
          'dtype': None},
         'activation': {'factory': 'torch.nn.modules.activation.ReLU',
          'inplace': False}}
        >>> CustomizableDenseLayer.from_config(config)
        CustomizableDenseLayer(
          (l): Bilinear(in1_features=10, in2_features=15, out_features=3, bias=True)
          (a): ReLU()
        )

    Another use case for this behaviour are depended config entries (e.g.
    NN input size depends on selected input features).

    Some modules need unitilized classes or functions as an input.
    Factory already supports functions but cannot make them json
    serializable.
    Therefore, one can use the partial key instead of the factory key.
    All defined kwargs will overwrite the defaults without calling the function
    or initializing the class using partial.
    This is essentially a `functools.partial` call.
    One usecase is SpeechBrain which requires the activity to be not
    initialized at the class input.

        >>> class SBDenseLayer(Configurable, torch.nn.Module):
        ...     @classmethod
        ...     def finalize_dogmatic_config(cls, config):
        ...         config['linear'] = {
        ...             'factory': torch.nn.Linear,
        ...             'out_features': 3,
        ...         }
        ...         if config['linear']['factory'] == torch.nn.Linear:
        ...             config['linear']['in_features'] = 5
        ...         config['activation'] = {'partial': torch.nn.ReLU,}
        ...         config['linear_2'] = {'partial': torch.nn.Linear,
        ...                               'in_features': 3}
        ...     def __init__(self, linear, linear_2, activation):
        ...         super().__init__()
        ...         self.l = linear  # torch.nn.Linear(in_units, out_units)
        ...         self.l2 = linear_2(out_features=10)
        ...         self.a = activation()  # torch.nn.ReLU()
        ...     def __call__(self, x):
        ...         return self.l2(self.a(self.l(x)))
        >>> config = SBDenseLayer.get_config()
        >>> pprint(config)
        {'factory': 'padertorch.configurable.SBDenseLayer',
         'linear': {'factory': 'torch.nn.modules.linear.Linear',
          'in_features': 5,
          'out_features': 3,
          'bias': True,
          'device': None,
          'dtype': None},
         'linear_2': {'partial': 'torch.nn.modules.linear.Linear',
          'in_features': 3,
          'bias': True,
          'device': None,
          'dtype': None},
         'activation': {'partial': 'torch.nn.modules.activation.ReLU',
          'inplace': False}}
        >>> SBDenseLayer.from_config(config)
        SBDenseLayer(
          (l): Linear(in_features=5, out_features=3, bias=True)
          (l2): Linear(in_features=3, out_features=10, bias=True)
          (a): ReLU()
        )

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
        {'factory': 'padertorch.configurable.EncoderDecoder',
         'encoder': {'factory': 'padertorch.configurable.DenseLayer',
          'in_units': 5,
          'out_units': 3},
         'decoder': {'factory': 'padertorch.configurable.DenseLayer',
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
        """This function is an outdated concept and thus deprecated."""
        raise DeprecationWarning('get_signature should no longer be used'
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
    def new(
            cls,
            updates=None,
    ):
        """Produce a Configurable instance.

        The updates are used to create a config and this config is then used to
        create the instance.
        """
        new: cls = cls.from_config(cls.get_config(updates))
        return new

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

        if 'factory' in config:
            config['factory'] = import_class(config['factory'])
            if inspect.isclass(config['factory']) \
                    and issubclass(config['factory'], Configurable):
                # When subclass of Configurable expect proper subclass
                assert issubclass(config['factory'], cls), (
                    config['factory'], cls)
        # If get_config has to be called on a partial object,
        # you can use this
        # elif 'partial' in config:
        #     config['partial'] = import_class(config['partial'])
        #     assert callable(config['partial']), config['partial']
        else:
            config['factory'] = cls

        config = _DogmaticConfig.normalize(config)

        # Calculate the config and convert it to a nested dict structure
        config = _DogmaticConfig(config).to_dict()

        _test_config(config, {})

        # For sacred make an inplace change to the update
        # (Earlier nessesary, now optional)
        updates.clear()
        updates.update(config)

        return updates

    @classmethod
    def from_config(
            cls,
            config,
    ):
        """Produce a Configurable instance from a valid config."""
        # TODO: assert do not use defaults


        if isinstance(config, _DogmaticConfig):
            config = config.to_dict()  # if called in finalize_dogmatic dict
        assert 'factory' in config, (cls, config)
        if cls is not Configurable:

            if cls.__module__ == '__main__':
                # When a class is defined in the main script, it will be
                # __main__.<ModelName>, but it should be <script>.<ModelName>.
                # This fix is active when the script is called with
                # "python -m <script> ..."
                # but not when it is called with "python <script>.py ..."
                # pylint: disable=self-cls-assignment
                cls = import_class(class_to_str(cls))

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

        new: cls = config_to_instance(config)
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
            in_config_path: e.g. 'trainer.model'
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
            import dlp_mpi
            if dlp_mpi.IS_MASTER:
                configurable_config = load_config(config_path=config_path)
            else:
                configurable_config = None
            configurable_config = dlp_mpi.bcast(configurable_config)
        else:
            configurable_config = load_config(config_path=config_path)
        if config_path != '':
            for part in in_config_path.split('.'):
                configurable_config = configurable_config[part]
        return cls.from_config(configurable_config)


def _test_config(config, updates):
    """Test if the config updates are valid.

    >>> _test_config({'factory': 'dict', 'a': 'abc'}, None)
    >>> _test_config({'factory': 'list'}, None)
    >>> _test_config({'factory': 'set'}, None)
    >>> _test_config({'factory': 'tuple'}, None)

    """
    # Rename this function, when it is nessesary to make it public.
    # The name test_config without an leading `_` confuses pytest.
    cls = import_class(config['factory'])

    sig = _get_signature(
        cls,
        drop_positional_only=True,  # not supported
    )

    # Remove default -> force completely described
    sig = sig.replace(
        parameters=[
            p.replace(default=inspect.Parameter.empty)
            for p in sig.parameters.values()
        ]
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


def dataclass_to_config(cls, depth=0, force_valid_config=True):
    """ Create a config from a dataclass. Follows fields and consides functools
    partial to create the config.

    This is a utility that is used in Configurable, but it can also be used as
    standalone function.

    Args:
        cls:
        depth: Helper for RecursionError.
        force_valid_config:

    Returns:


    >>> fix_doctext_import_class(locals())
    >>> import dataclasses, functools
    >>> from paderbox.utils.pretty import pprint
    >>> def bar(e=5, f=6): pass
    >>> @dataclasses.dataclass
    ... class Foo:
    ...     arg: int = 1
    ...     l: None = dataclasses.field(default_factory=list)
    ...     d: None = dataclasses.field(default_factory=functools.partial(dict, key=2))
    >>> @dataclasses.dataclass
    ... class A:
    ...     p: None = dataclasses.field(default_factory=functools.partial(Foo, arg=3))
    ...     f: None = dataclasses.field(default_factory=Foo)
    ...     c: None = 4
    ...     g: None = bar
    >>> config = dataclass_to_config(A)
    >>> pprint(config)  # doctest: +ELLIPSIS
    {'factory': ...configurable.A,
     'p': {'factory': ...configurable.Foo,
      'arg': 3,
      'l': {'factory': list},
      'd': {'factory': dict, 'key': 2}},
     'f': {'factory': ...configurable.Foo,
      'arg': 1,
      'l': {'factory': list},
      'd': {'factory': dict, 'key': 2}},
     'c': 4,
     'g': {'partial': <function ...configurable.bar(e=5, f=6)>}}
    >>> Configurable.from_config(config)  # doctest: +ELLIPSIS
    A(p=Foo(arg=3, l=[], d={'key': 2}), f=Foo(arg=1, l=[], d={'key': 2}), c=4, g=<function bar at 0x...>)

    >>> pprint(Configurable.get_config({'factory': A}))
    {'factory': 'padertorch.configurable.A',
     'p': {'factory': 'padertorch.configurable.Foo',
      'arg': 3,
      'l': {'factory': 'list'},
      'd': {'factory': 'dict', 'key': 2}},
     'f': {'factory': 'padertorch.configurable.Foo',
      'arg': 1,
      'l': {'factory': 'list'},
      'd': {'factory': 'dict', 'key': 2}},
     'c': 4,
     'g': {'partial': 'padertorch.configurable.bar', 'e': 5, 'f': 6}}

    >>> @dataclasses.dataclass
    ... class B:
    ...     no_default: None
    ...     missing_in_init: None = dataclasses.field(init=False, default_factory=Foo)
    >>> config = dataclass_to_config(B)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    RuntimeError: ('no_default', Field(name='no_default',type=None,default=<..._MISSING_TYPE...>,default_factory=<..._MISSING_TYPE...>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),..._field_type=_FIELD), <class '...configurable.B'>)
    >>> config = dataclass_to_config(B, force_valid_config=False)
    >>> pprint(config)  # doctest: +ELLIPSIS
    {'factory': ...configurable.B}
    """
    import dataclasses
    import functools

    if depth > 20:
        raise RecursionError(cls)

    def is_dataclass(obj):
        return dataclasses.is_dataclass(obj) and isinstance(obj, type)

    config = {'factory': cls}
    for k, f in cls.__dataclass_fields__.items():
        assert f.default is dataclasses.MISSING or f.default_factory is dataclasses.MISSING, f

        if not f.init:
            # Skip fields that are marked as not part of the __init__
            pass
        elif f.default is not dataclasses.MISSING:
            if callable(f.default):
                config[k] = {'partial': f.default}
            else:
                config[k] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            if isinstance(f.default_factory, functools.partial):
                assert f.default_factory.args == (), (f.default_factory.args, f)
                assert 'factory' not in f.default_factory.keywords, (f.default_factory.keywords, f)
                if is_dataclass(f.default_factory.func):
                    config[k] = dataclass_to_config(
                        f.default_factory.func, depth=depth + 1,
                        force_valid_config=force_valid_config)
                    config[k].update(**f.default_factory.keywords)
                else:
                    config[k] = {
                        'factory': f.default_factory.func,
                        **f.default_factory.keywords,
                    }

                if f.default_factory.args != ():
                    raise NotImplementedError(
                        'Found functools.partial with a positional'
                        'arguments. This is not yet supported for a'
                        'config.\n'
                        f'f: {f}\n'
                        f'f.default_factory: {f.default_factory}\n'
                        f'f.default_factory.args: {f.default_factory.args}\n'
                        f'f.default_factory.keywords: {f.default_factory.keywords}'
                    )
            else:
                if is_dataclass(f.default_factory):
                    config[k] = dataclass_to_config(
                        f.default_factory, depth=depth + 1,
                        force_valid_config=force_valid_config,)
                else:
                    config[k] = {'factory': f.default_factory}
        else:
            if force_valid_config:
                raise RuntimeError(k, f, cls)
    return config


def fix_doctext_import_class(locals_dict):
    """Allow classes defined in a doctest to be imported.

    This is necessary because classes defined in a doctest can not not be
    imported.

    Use this function inside a doctest as
        >>> fix_doctext_import_class(locals())  # doctest: +SKIP

    Example::
        >>> abc = 1
        >>> class Foo: pass
        >>> def foo(): pass
        >>> import_class(class_to_str(Foo))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ImportError: Could not import 'Foo' from '...configurable',
        because module '...configurable' has no attribute 'Foo'
        <BLANKLINE>
        Make sure that
         1. This is the class you want to import.
         2. You activated the right environment.
         3. The module exists and has been installed with pip.
         4. You can import the module (and class) in ipython.
        <BLANKLINE>
        >>> fix_doctext_import_class(locals())
        >>> class_to_str(Foo)  # doctest: +ELLIPSIS
        '...configurable.Foo'
        >>> class_to_str(foo)  # doctest: +ELLIPSIS
        '...configurable.foo'
        >>> Foo  # doctest: +ELLIPSIS
        <class '...configurable.Foo'>
        >>> import_class(class_to_str(foo)) == foo
        True
        >>> import_class(class_to_str(Foo))  # doctest: +ELLIPSIS
        <class '...configurable.Foo'>
    """
    cache = {}
    cls_cache = {}

    # Fix the case when this function is called multiple times.
    # This is necessary when multiple doctest are executed (e.g. pytest).
    # Then this fix will be called multiple times.
    class_to_str_orig = getattr(class_to_str, 'orig', class_to_str)
    import_class_orig = getattr(import_class, 'orig', import_class)

    def is_doctest_callable(obj):
        if inspect.isfunction(obj):
            return obj.__code__.co_filename.startswith('<doctest')
        elif inspect.isclass(obj):
            try:
                return False
            except OSError:  # OSError: could not find class definition
                return True
        else:
            raise TypeError(obj)

    def class_to_str_fix(cls):
        nonlocal cache
        nonlocal cls_cache
        if cls in cls_cache:
            return cls_cache[cls]
        name = class_to_str_orig(cls, fix_module=True)
        if '__name__' not in locals_dict:
            # This function was called in another doctest.
            # Remove this function from globals.
            globals()['import_class'] = import_class_orig
            globals()['class_to_str'] = class_to_str_orig

        elif (not isinstance(cls, str)) \
                and locals_dict['__name__'].endswith(cls.__module__):
            if is_doctest_callable(cls):
                cls.__module__ = _get_correct_module_str_for_callable(cls)
            cache[name] = cls
            # cls_cache[cls] = name
        return name

    def import_class_fix(name):
        if name in cache:
            return cache[name]
        else:
            return import_class_orig(name)

    class_to_str_fix.orig = class_to_str_orig
    import_class_fix.orig = import_class_orig

    # for code in the doctest
    locals_dict['import_class'] = import_class_fix
    locals_dict['class_to_str'] = class_to_str_fix

    # for the remaining code
    globals()['import_class'] = import_class_fix
    globals()['class_to_str'] = class_to_str_fix


def import_class(name: [str, callable]):
    """Import the str and return the imported object.

    Opposite of class_to_str.

    >>> import padertorch
    >>> import_class(padertorch.Model)
    <class 'padertorch.base.Model'>
    >>> import_class('padertorch.base.Model')
    <class 'padertorch.base.Model'>
    >>> import_class(padertorch.Model.from_file)  # doctest: +ELLIPSIS
    <bound method ...Configurable.from_file of <class 'padertorch.base.Model'>>
    >>> import_class('padertorch.Model.from_file')  # doctest: +ELLIPSIS
    <bound method ...Configurable.from_file of <class 'padertorch.base.Model'>>
    >>> import_class('dict')
    <class 'dict'>
    >>> import_class('padertorch.Model.typo')
    Traceback (most recent call last):
    ...
    ImportError: Could not import 'Model.typo' from 'padertorch',
    because type object 'Model' has no attribute 'typo'
    <BLANKLINE>
    Make sure that
     1. This is the class you want to import.
     2. You activated the right environment.
     3. The module exists and has been installed with pip.
     4. You can import the module (and class) in ipython.
    <BLANKLINE>
    >>> import_class('padertorch.base.typo')
    Traceback (most recent call last):
    ...
    ImportError: Could not import 'typo' from 'padertorch.base',
    because module 'padertorch.base' has no attribute 'typo'
    <BLANKLINE>
    Make sure that
     1. This is the class you want to import.
     2. You activated the right environment.
     3. The module exists and has been installed with pip.
     4. You can import the module (and class) in ipython.
    <BLANKLINE>
    >>> import_class('typo.in.pkg.name')
    Traceback (most recent call last):
    ...
    ImportError: Could not import 'typo.in.pkg.name'.
    <BLANKLINE>
    Make sure that
     1. This is the class you want to import.
     2. You activated the right environment.
     3. The module exists and has been installed with pip.
     4. You can import the module (and class) in ipython.
    <BLANKLINE>

    """
    if not isinstance(name, str):
        if not callable(name):
            raise TypeError(
                'expects string or callable but got', type(name), name)
        return name

    splitted = name.split('.')

    for i in reversed(range(1, len(splitted))):
        module_name = '.'.join(splitted[:i])
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError:
            continue
    else:
        if hasattr(builtins, name):
            return getattr(builtins, name)
        else:
            module_name = '__main__'  # Fallback
            module = importlib.import_module(module_name)
            i = 0

    qualname = splitted[i:]
    cls = module
    for part in qualname:
        try:
            cls = getattr(cls, part)
        except AttributeError as e:
            qualname = '.'.join(qualname)
            if name.startswith(module_name):
                raise ImportError(
                    f'Could not import {qualname!r} from {module_name!r},\n'
                    f'because {e}\n\n'
                    f'Make sure that\n'
                    f' 1. This is the class you want to import.\n'
                    f' 2. You activated the right environment.\n'
                    f' 3. The module exists and has been installed with pip.\n'
                    f' 4. You can import the module (and class) in ipython.\n'
                ) from None
            else:
                raise ImportError(
                    f'Could not import {name!r}.\n\n'
                    f'Make sure that\n'
                    f' 1. This is the class you want to import.\n'
                    f' 2. You activated the right environment.\n'
                    f' 3. The module exists and has been installed with pip.\n'
                    f' 4. You can import the module (and class) in ipython.\n'
                )

    return cls


def get_module_name_from_file(file):
    """
    >>> get_module_name_from_file(__file__)
    'padertorch.configurable'
    """

    # coppied from inspect.getabsfile
    file = os.path.normcase(os.path.abspath(file))

    file, module_path = os.path.split(file)
    module_path = os.path.splitext(module_path)[0]
    while file:
        # See setuptools.PackageFinder._looks_like_package
        if not os.path.isfile(os.path.join(file, '__init__.py')):
            break
        file, part = os.path.split(file)
        module_path = part + '.' + module_path
    if '.' in module_path:
        return module_path
    else:
        return '__main__'


def resolve_main_python_path() -> str:
    """
    Can only resolve, if you run scripts with `python -m`.

    Added a special fallback:
        __init__.py files mark package directories.
        Use this knowledge to find the root of the package.
        See setuptools.PackageFinder._looks_like_package

    """
    python_path = getattr(sys.modules['__main__'].__spec__, 'name', '__main__')

    # In an interactive interpreter, sys.modules['__main__'] doesn't have a
    # __file__ attribute, so just keep __main__ as python_path then. This
    # obviously doesn't allow to load a config created in an interactive
    # interpreter from outside.
    if python_path == '__main__' and hasattr(
            sys.modules['__main__'], '__file__'):
        python_path = get_module_name_from_file(
            sys.modules['__main__'].__file__)

    return python_path


def _get_correct_module_str_for_callable(callable_obj):
    """

    When a script is called with `python path/to/script.py` the module path is
    wrong. This functions determines the module path as when the
    script was called with `python -m path.to.script`

    >>> _get_correct_module_str_for_callable(_get_correct_module_str_for_callable)
    'padertorch.configurable'
    >>> import torch.nn
    >>> _get_correct_module_str_for_callable(torch.nn.modules.linear.Linear)
    'torch.nn.modules.linear'

    >>> def foo(): pass
    >>> class Foo: pass

    >>> _get_correct_module_str_for_callable(foo)
    'padertorch.configurable'
    >>> _get_correct_module_str_for_callable(Foo)
    'padertorch.configurable'

    """

    if inspect.isclass(callable_obj):
        # classes have no __globals__
        file = inspect.getabsfile(callable_obj)
    elif inspect.isfunction(callable_obj):
        # inspect.getabsfile yields `/path/to/func<doctest func>`
        file = callable_obj.__globals__['__file__']
    else:
        raise TypeError(callable_obj)

    try:
        candidates = [
            f for f in Path(file).parents
            if (f / '__init__.py').exists()
        ]
        p = candidates[-1]

    except IndexError:
        raise Exception(file)
    module = '.'.join(
        Path(file).relative_to(p.parent).with_suffix('').parts)
    return module


def class_to_str(cls, fix_module=False):
    """Convert a class to an importable str.

    Opposite of import_class.

    >>> import padertorch
    >>> class_to_str(padertorch.Model)
    'padertorch.base.Model'
    >>> class_to_str('padertorch.Model')
    'padertorch.base.Model'

    # Pycharm starts doctests with the absolute file, so the package is not
    # correctly recognized.
    >>> class_to_str(class_to_str)  # doctest: +SKIP
    'configurable.class_to_str'

    # With the option fix_module the module will be correctly identified.
    # Usually this is not nessesary. But we need it to fix some doctests.
    >>> class_to_str(class_to_str, fix_module=True)
    'padertorch.configurable.class_to_str'

    >>> import torch.nn
    >>> class_to_str(torch.nn.Linear)
    'torch.nn.modules.linear.Linear'
    >>> class_to_str(torch.nn.Linear, fix_module=True)
    'torch.nn.modules.linear.Linear'

    >>> class_to_str(dict, fix_module=True)
    'dict'
    >>> class_to_str(list, fix_module=True)
    'list'

    >>> class_to_str(padertorch.Model.get_config)
    'padertorch.base.Model.get_config'

    TODO: fix __main__ for scripts in packages that are called with shell
          path (path/to/script.py) and not python path (path.to.script).
    """
    if isinstance(cls, str):
        cls = import_class(cls)

    try:
        # https://stackoverflow.com/a/59924144/5766934
        # `cls.__self__` is only defined for bound methods
        module = cls.__self__.__module__
    except AttributeError:
        module = cls.__module__

    if fix_module \
            and '.' not in module \
            and module not in ['builtins']:
        module = _get_correct_module_str_for_callable(cls)

    if module == '__main__':
        # Try to figure out the module.
        # Could be done, when the script is started with "python -m ..."
        module = resolve_main_python_path()

    def fixed_qualname(cls):
        try:
            # https://stackoverflow.com/a/59924144/5766934
            # inherited method has wrong `__qualname__`
            # `cls.__self__` is only defined for bound methods
            return f'{cls.__self__.__qualname__}.{cls.__name__}'
        except AttributeError:
            return cls.__qualname__

    if module not in ['__main__', 'builtins']:
        return f'{module}.{fixed_qualname(cls)}'
    else:
        return f'{fixed_qualname(cls)}'


def recursive_class_to_str(config, sort=False):
    """
    Ensures that factory and partial values are strings.

    The config that is returned from a configurable already takes care, that
    all factory and partial values are strings. But when sacred overwrites a
    factory values with a class and not the str, the config will contain a
    class instead of the corresponding string.

    Args:
        config:

    Returns:
        config where each factory value is a str and each pathlib.Path
        is converted to str.


    >>> import torch.nn
    >>> cfg = {'factory': torch.nn.Linear, 'in_features': 1, 'out_features': 2}
    >>> recursive_class_to_str(cfg, sort=True)
    {'factory': 'torch.nn.modules.linear.Linear', 'in_features': 1, 'out_features': 2}
    >>> cfg = {'factory': torch.nn.Linear, 'out_features': 2, 'in_features': 1}
    >>> recursive_class_to_str(cfg, sort=True)
    {'factory': 'torch.nn.modules.linear.Linear', 'in_features': 1, 'out_features': 2}
    >>> cfg = {'out_features': 2, 'in_features': 1, 'factory': torch.nn.Linear}
    >>> recursive_class_to_str(cfg, sort=True)
    {'factory': 'torch.nn.modules.linear.Linear', 'in_features': 1, 'out_features': 2}
    >>> cfg = {'partial': torch.nn.LeakyReLU, 'negative_slope': 0.01, 'inplace': False}
    >>> recursive_class_to_str(cfg, sort=True)
    {'partial': 'torch.nn.modules.activation.LeakyReLU', 'negative_slope': 0.01, 'inplace': False}
    >>> cfg = {'partial': torch.nn.LeakyReLU, 'inplace': False, 'negative_slope': 0.01}
    >>> recursive_class_to_str(cfg, sort=True)
    {'partial': 'torch.nn.modules.activation.LeakyReLU', 'negative_slope': 0.01, 'inplace': False}
    >>> cfg = {'inplace': False, 'negative_slope': 0.01, 'partial': torch.nn.LeakyReLU}
    >>> recursive_class_to_str(cfg, sort=True)
    {'partial': 'torch.nn.modules.activation.LeakyReLU', 'negative_slope': 0.01, 'inplace': False}
    >>> recursive_class_to_str(Path('/pathlib/Path/object'), sort=True)
    '/pathlib/Path/object'

    """
    # ToDo: Support tuple and list?
    if isinstance(config, dict):
        d = config.__class__()
        special_key = _get_special_key(config)

        if sort and special_key:
            # Force the special key to be the first key
            d[special_key] = None  # will be set later
            imported = import_class(config[special_key])
            arg_names = _get_signature(
                imported,
                drop_positional_only=True,  # not supported
            ).parameters.keys()
            # This ensure that the keys are in the same order as the signature
            for k in arg_names:
                if k in config:
                    d[k] = None  # will be set later

        for k, v in config.items():
            if special_key and k == special_key:
                d[k] = class_to_str(v)
            else:
                d[k] = recursive_class_to_str(v)
        return d
    elif isinstance(config, (tuple, list)):
        return config.__class__([
            recursive_class_to_str(l) for l in config
        ])
    elif isinstance(config, Path):
        return str(config)
    else:
        return config


def _split_factory_kwargs(config, key='factory'):
    kwargs = config.copy()
    factory = kwargs.pop(key)
    return factory, kwargs


def _get_special_key(config):
    # These special keys are used in the config to indicate a class or
    # function. 'factory' is used to specify initialized classes or
    # function outputs as an input. 'partial' is used if an input is a
    # non-initialized class or a functions
    for key in ['factory', 'partial']:
        if key in config.keys():
            return key
    return None


def _check_factory_signature_and_kwargs(factory, kwargs, strict, special_key):
    """
    Buildins are can be problematic, becuase they may have no signature
    >>> config_to_instance({'factory': 'dict', 'A': 3})
    {'A': 3}
    >>> config_to_instance({'factory': 'list'})
    []
    >>> _check_factory_signature_and_kwargs(list, {}, True, 'factory')
    """
    sig = _get_signature(
        factory,
        drop_positional_only=True,  # not supported
    )
    # Remove annotation, sometimes they are to verbose and in python
    # 3.7 they changed the `__str__` function, when an annotation is
    # known (e.g. '(inplace:bool)' -> '(inplace: bool)').
    # This breaks doctests across python versions.
    sig = sig.replace(
        parameters=[
            p.replace(annotation=inspect.Parameter.empty)
            for p in sig.parameters.values()
        ]
    )

    # Define key specific check_func since factory requires all keys of the
    # signature to be specified in kwargs whereas partial only requires the
    # keys in kwargs to be in the signature.
    if special_key == 'factory':
        check_func = sig.bind
    elif special_key == 'partial':
        check_func = sig.bind_partial
    else:
        raise ValueError(special_key)
    try:
        # With sig.bind we ensure, that the "bind" here raises the
        # exception. Using the factory(**kwargs) may raise TypeError
        # with another cause. The overhead doesn't matter here.
        bound_arguments: inspect.BoundArguments = check_func(**kwargs)
    except TypeError as e:
        raise TypeError(
            f'{e}\n'
            f'Tried to instantiate/call {factory} with\n'
            f'`{class_to_str(factory)}(**{kwargs})`.\n'
            f'Signature: {sig}'
        ) from e

    if strict:
        sig = sig.replace(
            parameters=[p.replace(
                default=inspect.Parameter.empty
            ) for p in sig.parameters.values()]
        )
        try:
            bound_arguments: inspect.BoundArguments = sig.bind(**kwargs)
        except TypeError as e:
            raise TypeError(
                f'{e}\n'
                f'Tried to instantiate/call {factory} with\n'
                f'`{class_to_str(factory)}(**{kwargs})` '
                f'in strict mode.\n'
                f'Strict means ignore defaults from the signature.\n'
                f'Signature: {sig}'
            ) from e


def config_to_instance(config, strict=False):
    """Is called by `Module.from_config()`. If possible, use that directly.

    Args:
        config:
        strict:
            If True, checks, that the instansiations doesn't use the default
            values of the signature of the factory. That means, the config must
            contains all arguments for the factory.

            Usecase: While doing experiment, you add new arguments to your
            factory and these don't reflect the old behaviour.
            The strict arguments allows you to detect when you use an old
            config that has to be adjusted.

    Returns:

    >>> import torch.nn
    >>> config = {
    ...     'factory': 'torch.nn.modules.activation.ReLU',
    ...     'inplace': False}
    >>> config_to_instance(config)
    ReLU()
    >>> config_to_instance(config, strict=True)
    ReLU()
    >>> config = {
    ...     'factory': 'torch.nn.modules.activation.ReLU'}
    >>> config_to_instance(config)
    ReLU()
    >>> config_to_instance(config, strict=True)
    Traceback (most recent call last):
    ...
    TypeError: missing a required argument: 'inplace'
    Tried to instantiate/call <class 'torch.nn.modules.activation.ReLU'> with
    `torch.nn.modules.activation.ReLU(**{})` in strict mode.
    Strict means ignore defaults from the signature.
    Signature: (inplace)
    >>> config = {
    ...     'factory': 'torch.nn.modules.activation.ReLU',
    ...     'inplace_typo': False}
    >>> config_to_instance(config)
    Traceback (most recent call last):
    ...
    TypeError: got an unexpected keyword argument 'inplace_typo'
    Tried to instantiate/call <class 'torch.nn.modules.activation.ReLU'> with
    `torch.nn.modules.activation.ReLU(**{'inplace_typo': False})`.
    Signature: (inplace=False)
    >>> config_to_instance(config, strict=True)
    Traceback (most recent call last):
    ...
    TypeError: got an unexpected keyword argument 'inplace_typo'
    Tried to instantiate/call <class 'torch.nn.modules.activation.ReLU'> with
    `torch.nn.modules.activation.ReLU(**{'inplace_typo': False})`.
    Signature: (inplace=False)
    >>> config = {
    ...     'partial': 'torch.nn.modules.activation.ReLU',
    ...     'inplace': False}
    >>> config_to_instance(config)
    functools.partial(<class 'torch.nn.modules.activation.ReLU'>, inplace=False)
    >>> config_to_instance(config, strict=True)
    functools.partial(<class 'torch.nn.modules.activation.ReLU'>, inplace=False)
    >>> config = {
    ...     'partial': 'torch.nn.modules.activation.ReLU'}
    >>> config_to_instance(config)
    <class 'torch.nn.modules.activation.ReLU'>
    >>> config = {
    ...     'partial': 'torch.nn.Linear'}
    >>> config_to_instance(config)
    <class 'torch.nn.modules.linear.Linear'>
    """

    if isinstance(config, _DogmaticConfig):
        config = config.to_dict()  # if called in finalize_dogmatic dict

    if isinstance(config, dict):
        special_key = _get_special_key(config)
        if special_key:
            factory, kwargs = _split_factory_kwargs(config, key=special_key)
            try:
                factory = import_class(factory)
            except TypeError as err:
                raise TypeError(f'The special key {special_key} expects a '
                                f'string or a callable but got',
                                type(factory), factory) from err
            kwargs = config_to_instance(kwargs, strict)

            _check_factory_signature_and_kwargs(factory, kwargs,
                                                strict, special_key)

            if special_key == 'factory':
                new = factory(**kwargs)
            elif special_key == 'partial':
                if len(kwargs) > 0:
                    new = functools.partial(factory, **kwargs)
                else:
                    new = factory
            else:
                Exception('This cannot happen')
            try:
                new.config = config
            except AttributeError:
                pass
            return new
        else:
            d = copy.copy(config)  # config.__class__() not possible in sacred>=0.8 because of ReadOnlyDict.
            for k, v in config.items():
                d[k] = config_to_instance(v, strict)
            return d
    elif isinstance(config, (tuple, list)):
        return config.__class__([
            config_to_instance(l, strict) for l in config
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
            isinstance(m[item], collections.abc.Mapping)
            for m in self.maps
            if item in m
        ]
        if any(is_mapping) and is_mapping[0]:
            if not all(is_mapping):
                for m in self.maps:
                    if item in m:
                        if not isinstance(m[item], collections.abc.Mapping):
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


def _get_signature(cls, drop_positional_only=False, drop_type_annotations=False):
    """

    >>> _get_signature(dict)
    <Signature (**kwargs)>
    >>> _get_signature(list)
    <Signature (iterable=(), /)>
    >>> _get_signature(tuple)
    <Signature (iterable=(), /)>
    >>> _get_signature(set)
    <Signature (iterable=(), /)>

    >>> inspect.signature(dict)
    Traceback (most recent call last):
    ...
    ValueError: no signature found for builtin type <class 'dict'>
    >>> inspect.signature(set)
    Traceback (most recent call last):
    ...
    ValueError: no signature found for builtin type <class 'set'>


    >>> _get_signature(Configurable.from_file)
    <Signature (config_path: pathlib.Path, in_config_path: str = '', consider_mpi=False)>
    >>> _get_signature(Configurable.from_file, drop_type_annotations=True)
    <Signature (config_path, in_config_path='', consider_mpi=False)>

    """
    if cls in [
        set,  # py38: set missing signature
        tuple,  # py36: tuple missing signature (available in py37)
        list,  # py36: list missing signature (available in py37)
    ]:
        sig = inspect.Signature(
            parameters=[inspect.Parameter(
                'iterable', inspect.Parameter.POSITIONAL_ONLY,
                default=(),
            )]
        )
    elif cls.__init__ in [dict.__init__]:
        # Dict has no correct signature, hence return the signature, that is
        # needed here.
        sig = inspect.Signature(
            parameters=[inspect.Parameter(
                'kwargs', inspect.Parameter.VAR_KEYWORD,
            )]
        )
    else:
        sig = inspect.signature(cls)

    if drop_positional_only:
        sig = sig.replace(
            parameters=[
                p
                for p in sig.parameters.values()
                if p.kind != inspect.Parameter.POSITIONAL_ONLY
            ]
        )
    if drop_type_annotations:
        sig = sig.replace(
            parameters=[
                p.replace(annotation=p.empty)
                for p in sig.parameters.values()
            ],
            return_annotation=sig.empty
        )

    return sig


class _DogmaticConfig:
    """This class is an implementation detail of Configurable."""

    @staticmethod
    def get_signature(factory):
        """Check signature of the factory.

        If parameters have defaults, return these in a dictionary.

        Returns:

        >>> _DogmaticConfig.get_signature(Configurable.get_config)
        {'updates': None}
        >>> _DogmaticConfig.get_signature(list)  # Wrong signature
        {}
        >>> _DogmaticConfig.get_signature(dict)  # Has no signature
        {}
        """
        if factory in [tuple, list, set, dict]:
            return {}
        try:
            sig = inspect.signature(factory)
        except ValueError:
            if factory.__init__ in [tuple.__init__, list.__init__, set.__init__, dict.__init__]:
                # Buildin type is in MRO and __init__ is not overwritten. e.g.
                # ValueError: no signature found for builtin type <class 'paderbox.utils.mapping.Dispatcher'>
                return {}
            else:
                raise

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
        >>> pprint(_DogmaticConfig.normalize({
        ...     'model': {'partial': 'torch.nn.Linear'},
        ...     'storage_dir': Path('abc')
        ... }), max_width=79-8)
        {'model': {'partial': torch.nn.modules.linear.Linear},
         'storage_dir': 'abc'}
        >>> pprint(_DogmaticConfig.normalize({
        ...     'model': {'factory': 'torch.nn.Linear',
        ...               'partial': 'torch.nn.Linear'},
        ...     'storage_dir': Path('abc')
        ... }), max_width=79-8)
        {'model': {'factory': torch.nn.modules.linear.Linear,
          'partial': 'torch.nn.Linear'},
         'storage_dir': 'abc'}
        """
        if isinstance(dictionary, collections.abc.Mapping):
            special_key = _get_special_key(dictionary)
            if special_key:
                dictionary[special_key] = cls._force_factory_type(
                    dictionary[special_key]
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

        if self.special_key:
            self._check_redundant_keys(
                'padertorch.Configurable.get_config(updates=...) got an '
                f'unexpected keyword argument in updates for '
                f'{self.data[self.special_key]}.\n'
                'See details below.\n'
            )

    @property
    def special_key(self):
        return _get_special_key(self.data)

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
        if self.special_key:
            factory = import_class(self.data[self.special_key])
            parameters = _get_signature(
                factory,
                drop_positional_only=True,  # not supported
            ).parameters.values()
            p: inspect.Parameter

            parameter_names = tuple([self.special_key]) + tuple([
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
        assert self.special_key, f'Missing factory or partial in {self.data}'
        imported = import_class(self.data[self.special_key])
        parameters = _get_signature(
            imported,
            drop_positional_only=True,  # not supported
        ).parameters.values()
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
            ]) | {self.special_key}

            redundant_keys = set(self.data.keys()) - parameter_names

            if len(redundant_keys) != 0:
                from IPython.lib.pretty import pretty
                raise Exception(
                    f'{msg}\n'
                    f'Too many keywords for the factory {imported}.\n'
                    f'Redundant keys: {redundant_keys}\n'
                    f'Signature: {_get_signature(imported, drop_type_annotations=True)}\n'
                    f'Current config with fallbacks:\n{pretty(self.data)}'
                )

    def __contains__(self, item):
        return item in self.data

    def __setitem__(self, key, value):
        self.data[key] = self.normalize(value)

        if self.special_key:
            self._check_redundant_keys(
                'Tried to set an unexpected keyword argument for '
                f'{self.data[self.special_key]} in finalize_dogmatic_config.\n'
                'See details below and stacktrace above.\n',
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
        assert self.special_key, f'Missing factory or partial in {self.data}'

        # Force factory to be the class/function
        factory = import_class(self.data[self.special_key])

        # Freeze the mutable_idx (i.e. all updates to the config of
        # this level)
        mutable_idx_old = self.data.mutable_idx
        self.data.mutable_idx = len(self.data.maps) - 1

        if dataclasses.is_dataclass(factory) and isinstance(factory, type):
            # dataclasses.is_dataclass returns True for instance and class.
            # The isinstance(factory, type) makes it True for only class
            defaults = dataclass_to_config(factory, force_valid_config=False)
        else:
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
            except RecursionError as e:
                raise AssertionError(
                    f'Did you tried to call `{class_to_str(factory)}.from_config(config)` in '
                    f'`{class_to_str(factory)}.finalize_dogmatic_config`?\n'
                    'This is theoretically impossible.\n'
                    'You can try `cls.from_config({**config})`, this may not '
                    'result in an RecursionError, but ignores the code below '
                    'in the finalize_dogmatic_config.'
                ) from e

        delta = set(self.data.keys()) - set(self._key_candidates())

        if len(delta) > 1:
            # (delta, self.data.keys(), parameter_names)
            from IPython.lib.pretty import pretty
            raise Exception(
                f'Too many keywords for the factory {factory}.\n'
                f'Delta: {delta}\n'
                f'signature: {_get_signature(factory)}\n'
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
        if self.special_key and key != self.special_key \
                    and self.data.mutable_idx != (len(self.data.maps) - 1):
            self._update_factory_kwargs()

        if 'cls' in self._key_candidates():
            from IPython.lib.pretty import pretty
            factory = self.data['cls']
            factory_str = class_to_str(factory)
            raise Exception(
                f'Got the old key "cls" (value: {factory_str}).\n'
                f'Use the new key: factory\n'
                f'Signature: {_get_signature(factory)}\n'
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

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self):
        """Export the Configurable object to a dict."""
        result_dict = {}
        if self.special_key:
            self._update_factory_kwargs()

        for k in self._key_candidates():
            try:
                v = self[k]
            except KeyError as ex:
                from IPython.lib.pretty import pretty
                if self.special_key == 'factory' \
                        and self.special_key in self._key_candidates() and \
                        k != self.special_key:
                    # KeyError has a bad __repr__, use Exception
                    missing_keys = set(self._key_candidates()) - set(self.data.keys())
                    raise Exception(
                        f'KeyError: {k}\n'
                        f'signature: {_get_signature(self[self.special_key])}\n'
                        f'missing keys: {missing_keys}\n'
                        f'self:\n{pretty(self)}'
                    ) from ex
                elif self.special_key == 'partial' \
                        and self.special_key in self._key_candidates() and \
                        k != self.special_key:
                    continue
                else:
                    # KeyError has a bad __repr__, use Exception
                    # Can this happen?
                    raise Exception(
                        f'{k}\n'
                        f'signature: {_get_signature(self[self.special_key])}'
                        f'self:\n{pretty(self)}'
                    ) from ex

            if self.special_key and k == self.special_key:
                v = class_to_str(v)
            if isinstance(v, self.__class__):
                v = v.to_dict()

            assert not hasattr(v, 'to_dict'), (k, v, result_dict)
            result_dict[k] = v

        if 'factory' in result_dict:
            assert isinstance(result_dict['factory'], str), result_dict
            _test_config(result_dict, {})

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
