import os
import io
from pathlib import Path

from paderbox.io.new_subdir import get_new_subdir


def get_new_storage_dir(
        experiment_name: str,
        *,
        id_naming: [str, callable] = 'index',
        mkdir: bool = True,
        prefix: str = None,
        suffix: str = None,
        consider_mpi: bool = False,
        dry_run:bool = False,
):
    """Determine a new non-existent storage_dir located in
        `os.environ['STORAGE_ROOT'] / experiment_name`

    This is a wrapper around `paderbox.io.new_subdir.get_new_subdir`.

    Features:
     - With mkdir: Thread and process save.
     - Different conventions for ID naming possible, default running index.
     - MPI aware: Get the folder on one worker and distribute to others.

    Args:
        experiment_name:
            The sub folder name, that is used in STORAGE_ROOT.
        id_naming:
            The id naming that is used for the folder name.
             - str: 'index':
                The largest index in basedir + 1.
                e.g.: '1', '2', ...
             - str: 'time': A timestamp with the format %Y-%m-%d-%H-%M-%S
                e.g. '2020-08-13-17-02-57'
             - callable: Each call should generate a new name.
        mkdir:
            Creates the dir and makes the program process/thread safe.
            Note this option ensures that you don't get a
            conflict between two concurrent calls of get_new_folder.
            Example:
                You launch several times your programs and each should get
                another folder (e.g. hyperparameter search). When inspecting
                basedir maybe some recognize they can use '2' as sub folder.
                This option ensures, that only one program gets the '2' and the
                remaining programs search for another free id.
        prefix:
            Optional prefix for the id. e.g.: '2' -> '{prefix}_2'
        suffix:
            Optional suffix for the id. e.g.: '2' -> '2_{suffix}'
        consider_mpi:
            If True, only search on one mpi process for the folder and
            distribute the folder name.
            When using mpi (and `consider_mpi is False`) the following can/will
            happen
             - When mkdir is True every process will get another folder.
               i.e. each process has a folder just for this process.
             - Warning: Never use mpi, when `mkdir is False` and
               `consider_mpi is False`. Depending on some random factors
               (e.g. python startup time) all workers could get the same
               folder, but mostly some get the same folder and some different.
               You never want this.
        dry_run:
            When true, disables mkdir and prints the folder name.

    Returns:
        pathlib.Path of the new subdir

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     os.environ['STORAGE_ROOT'] = tmp_dir  # simulate enviroment variable for doctest
    ...     print(get_new_storage_dir('fance_nn_experiment').relative_to(tmp_dir))
    ...     print(get_new_storage_dir('fance_nn_experiment').relative_to(tmp_dir))
    fance_nn_experiment/1
    fance_nn_experiment/2

    """
    basedir = Path(os.environ['STORAGE_ROOT']) / experiment_name
    del experiment_name
    return get_new_subdir(**locals())


def dump_config(
        config,
        path,
):
    """
    This functions dumps a config as json or yaml.

     - Convert factory values to str if necessary
     - Move factory to be the first key in the dict
     - Sort the order of the kwargs based on the signature of the factory.

    """
    path = Path(path)
    path.write_text(dumps_config(config, path.suffix))


def dumps_config(
        config,
        format='.yaml'  # '.json' or '.yaml'
):
    """
    >>> print(dumps_config({'a': 1, 'b': 2}, '.yaml'))
    a: 1
    b: 2
    <BLANKLINE>
    >>> print(dumps_config({'b': 2, 'a': 1}, '.yaml'))
    b: 2
    a: 1
    <BLANKLINE>
    >>> print(dumps_config({'a': 1, 'b': 2}, '.json'))
    {
      "a": 1,
      "b": 2
    }
    >>> print(dumps_config({'b': 2, 'a': 1}, '.json'))
    {
      "b": 2,
      "a": 1
    }

    >>> import torch
    >>> print(dumps_config({'factory': torch.nn.ReLU, 'b': 2}, '.yaml'))
    factory: torch.nn.modules.activation.ReLU
    b: 2
    <BLANKLINE>
    >>> print(dumps_config({'b': 2, 'factory': torch.nn.ReLU}, '.yaml'))
    factory: torch.nn.modules.activation.ReLU
    b: 2
    <BLANKLINE>

    """
    import padertorch as pt
    config = pt.configurable.recursive_class_to_str(config, sort=True)
    if format == ".json":
        from paderbox.io import dumps_json
        return dumps_json(config, sort_keys=False)
    elif format == ".yaml":
        from paderbox.io.yaml_module import dumps_yaml
        return dumps_yaml(config, sort_keys=False)
    else:
        raise NotImplementedError(format)


def load_config(
        path
):
    import paderbox as pb
    return pb.io.load(path)


def loads_config(
        content,
        format=None,  # '.yaml' or '.json'
):
    """
    >>> print(loads_config(dumps_config({'a': 1, 'b': 2}, '.yaml')))
    {'a': 1, 'b': 2}
    >>> print(loads_config(dumps_config({'b': 2, 'a': 1}, '.yaml')))
    {'b': 2, 'a': 1}
    >>> print(loads_config(dumps_config({'a': 1, 'b': 2}, '.json')))
    {'a': 1, 'b': 2}
    >>> print(loads_config(dumps_config({'b': 2, 'a': 1}, '.json')))
    {'b': 2, 'a': 1}
    """
    if format is None:
        if content.strip()[0] == '{':
            format = '.json'
        else:
            format = '.yaml'

    if format == ".json":
        from paderbox.io import loads_json
        return loads_json(content)
    elif format == ".yaml":
        from paderbox.io import loads_yaml
        return loads_yaml(content)
    else:
        raise NotImplementedError(format)