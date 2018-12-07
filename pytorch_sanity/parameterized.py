"""

ToDo: Example
Example usage: see example in ...

"""
import abc
import json
import inspect
import importlib


class Parameterized(abc.ABC):
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
    ):
        """
        Provides configuration to allow Instantiation with
        module = Module.from_config(Module.get_config())
        :param update_dict: dict with values to be modified w.r.t. defaults.
        Sub-configurations are updated accordingly if top-level-keys are
        changed. An Exception is raised if update_dict has unused entries.
        :return: config

        """
        defaults = recursive_class_to_str(cls.get_signature())
        # if config is None:
        # config = defaults  # cls._get_defaults()
        # assert defaults is not None, (defaults, config, cls)
        if updates is None:
            updates = dict()
        try:
            update_config(defaults, updates)
        except ConfigUpdateException as e:
            raise Exception(
                f'{cls.__module__}.{cls.__qualname__}: '
                f'Updates that are not used anywhere: {e}'
            )
        return defaults

    @classmethod
    def from_config(
            cls,
            config,
    ):
        # assert do not use defaults
        kwargs = config_to_kwargs(config)
        new = cls(**kwargs)
        new.config = config
        return new


def import_class(name: str):
    if not isinstance(name, str):
        return name
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
    blacklist = list()
    for key in sorted(list(config.keys())):
        if key in blacklist:
            continue
        if isinstance(config[key], dict):
            update_config(
                config[key],
                updates.pop(key) if updates and key in updates else dict(),
            )
        elif updates and key in updates:
            config[key] = updates.pop(key)
        if key.endswith('cls'):
            kwargs_key = f'{key[:-len("cls")]}kwargs'
            kwargs_config = config.get(kwargs_key, {})
            kwargs_specific_config = config.get(config[key], {})
            if kwargs_config is None:
                kwargs_config = {}
            if kwargs_specific_config is None:
                kwargs_specific_config = {}

            config[kwargs_key] = import_class(config[key]).get_config(
                {
                    **kwargs_config,
                    **kwargs_specific_config,
                    **updates.pop(kwargs_key, dict()),
                    **updates.pop(config[key], dict()),
                }
            )
            blacklist.append(kwargs_key)

            if key == 'cls':
                for cfg_key in list(config.keys()):
                    if cfg_key not in [key, kwargs_key]:
                        del config[cfg_key]
                for cfg_key in list(updates.keys()):
                    if cfg_key not in [key, kwargs_key]:
                        del updates[cfg_key]
                break

    if updates:
        from IPython.lib.pretty import pretty
        raise ConfigUpdateException(
            '\n\n'
            'updates:\n'
            f'{pretty(updates)}\n\n'
            'config:\n'
            f'{pretty(config)}'
        )


def config_to_kwargs(config):
    if isinstance(config, dict):
        if 'cls' in config:
            assert 'kwargs' in config, config
            assert len(config) == 2, config
            return import_class(config['cls'])(**config['kwargs'])
        else:
            return {k: config_to_kwargs(v) for k, v in config.items()}
    elif isinstance(config, (tuple, list)):
        return config.__class__([
            config_to_kwargs(l) for l in config
        ])
    else:
        return config
