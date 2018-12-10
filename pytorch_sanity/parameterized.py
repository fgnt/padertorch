"""

ToDo: Example
Example usage: see example in ...

"""
import abc
import sys
import json
import inspect
import importlib


class Parameterized(abc.ABC):
    _config = None

    @property
    def config(self):
        if self._config is None:
            p_name = f'{Parameterized.__module__}.{Parameterized.__qualname__}'
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
            p_name = f'{Parameterized.__module__}.{Parameterized.__qualname__}'
            name = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
            raise Exception(
                f'The config property of a {p_name} object\n'
                f'can only be set once.\n'
                f'You tried to set it for an instance of {name}.'
            )

    @classmethod
    def get_signature(cls):
        sig = inspect.signature(cls.__init__)
        sig = sig.replace(
            parameters=list(sig.parameters.values())[1:]  # drop self
        )
        defaults = {}
        p: inspect.Parameter
        for name, p in sig.parameters.items():
            if p.default is not inspect.Parameter.empty:
                defaults[name] = p.default
        return defaults

    @classmethod
    def get_config(
            cls,
            updates=None,
            config=None,
    ):
        """
        Provides configuration to allow Instantiation with
        module = Module.from_config(Module.get_config())
        :param update_dict: dict with values to be modified w.r.t. defaults.
        Sub-configurations are updated accordingly if top-level-keys are
        changed. An Exception is raised if update_dict has unused entries.
        :return: config

        """
        if config is None:
            config = {
                'cls': class_to_str(cls)
            }
        elif 'cls' not in config:
            config['cls'] = class_to_str(cls)
        else:
            config['cls'] = class_to_str(config['cls'])

        # This assert is for sacred that may change values in the config dict.
        assert issubclass(import_class(config['cls']), cls), (
            config['cls'], cls
        )

        config['kwargs'] = {
            **recursive_class_to_str(cls.get_signature()),
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
        except ConfigUpdateException as e:
            raise Exception(
                f'{cls.__module__}.{cls.__qualname__}: '
                f'Updates that are not used anywhere: {e}'
            )

        # Test if the kwargs are valid
        sig = inspect.signature(import_class(config['cls']).__init__)

        # Remove default -> force completely described
        sig = sig.replace(
            parameters=[p.replace(
                default=inspect.Parameter.empty
            ) for p in sig.parameters.values()]
        )
        try:
            bound_arguments: inspect.BoundArguments = sig.bind(
                self=None, **config['kwargs']
            )
            bound_arguments.apply_defaults()
        except TypeError as e:
            unexpected_keyword = 'got an unexpected keyword argument '
            if unexpected_keyword in str(e):
                func_name, keyword = str(e).split(unexpected_keyword)
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
                        default=config['kwargs'].get(p.name, inspect.Parameter.empty),
                    ) for p in sig.parameters.values()]
                )
                raise TypeError(
                    f'{config["cls"]} {e}\n'
                    f'Did you mean one of these {suggestions}?\n'
                    f'Call signature: {sig_wo_anno}'
                ) from e
            else:
                raise Exception(
                    f'The test, if the call {config["cls"]}(**kwargs) would be '
                    f'succesfull, failed.\n'
                    f'Where\n'
                    f'     kwargs: {config["kwargs"]}\n'
                    f'     signature: {sig}\n'
                    f'     updates: {updates}\n'
                ) from e

        return config

    @classmethod
    def from_config(
            cls,
            config,
    ) -> 'Parameterized':
        # assert do not use defaults
        assert 'cls' in config, (cls, config)
        assert issubclass(import_class(config['cls']), cls), (config['cls'], cls)
        new = config_to_instance(config)
        # new.config = config
        return new


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
    return getattr(module, splitted[-1])


def class_to_str(cls):
    """
    >>> import pytorch_sanity
    >>> class_to_str(pytorch_sanity.Model)
    'pytorch_sanity.base.Model'
    >>> class_to_str('pytorch_sanity.Model')
    'pytorch_sanity.base.Model'
    """
    if isinstance(cls, str):
        cls = import_class(cls)
    module = cls.__module__
    if module != '__main__':
        return f'{module}.{cls.__qualname__}'
    else:
        return f'{cls.__qualname__}'


def recursive_class_to_str(dictionary):
    """
    >>> from pytorch_sanity import Model
    >>> recursive_class_to_str([{'cls': 'pytorch_sanity.Model'}])
    [{'cls': 'pytorch_sanity.base.Model'}]
    >>> recursive_class_to_str([{'cls': Model, Model: {}}])
    [{'cls': 'pytorch_sanity.base.Model', 'pytorch_sanity.base.Model': {}}]
    """
    if isinstance(dictionary, dict):
        if 'cls' in dictionary:
            return {
                class_to_str(k) if not isinstance(k, str) else k
                :
                class_to_str(v) if k == 'cls' else recursive_class_to_str(v)
                for k, v in dictionary.items()
            }
        else:
            return dictionary.__class__({
                k: recursive_class_to_str(v)
                for k, v in dictionary.items()
            })
    elif isinstance(dictionary, (tuple, list)):
        return dictionary.__class__([
            recursive_class_to_str(l)
            for l in dictionary
        ])
    else:
        return dictionary


class ConfigUpdateException(Exception):
    pass


def update_config(config, updates=None):
    """

    :param config: config dict
    :param updates: updates dict. Note that update entries which are not valid
    for the current config are ignored.
    :return:
    """
    if 'cls' in config or 'cls' in updates:
        if 'cls' in updates:
            config['cls'] = class_to_str(updates['cls'])
        sub_updates = {
            **updates.get('kwargs', dict()),
            **updates.get(config['cls'], dict()),
        }

        cls = import_class(config['cls'])
        try:
            # inplace
            cls.get_config(
                updates=sub_updates,
                config=config,
            )
        except AttributeError as e:
            raise TypeError(
                f'Expected a subclass of {Parameterized}, but got {cls}.\n'
                f'config: {config}\n'
                f'method resolution order: {inspect.getmro(cls)}'
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

        if updates:
            from IPython.lib.pretty import pretty
            raise ConfigUpdateException(
                '\n\n'
                'updates:\n'
                f'{pretty(updates)}\n\n'
                'config:\n'
                f'{pretty(config)}'
            )


def config_to_instance(config):
    if isinstance(config, dict):
        if 'cls' in config:
            assert 'kwargs' in config, config
            assert len(config) == 2, config
            new = import_class(config['cls'])(
                **config_to_instance(config['kwargs'])
            )
            new.config = config
            return new
        else:
            return {k: config_to_instance(v) for k, v in config.items()}
    elif isinstance(config, (tuple, list)):
        return config.__class__([
            config_to_instance(l) for l in config
        ])
    else:
        return config
