import os
from pathlib import Path

from paderbox.io.new_subdir import get_new_subdir


def get_new_storage_dir(
        experiment_name,
        *,
        id_naming='index',
        mkdir=True,
        prefix=None,
        suffix=None,
        consider_mpi=False,
        dry_run=False,
):
    """
    This is a wrapper around `paderbox.io.new_subdir.get_new_subdir`.
    
    Different:
        Use as basedir: `os.environ['STORAGE_ROOT'] / experiment_name`

    >>> os.environ['STORAGE_ROOT'] = '/tmp'  # simulate enviroment variable for doctest
    >>> get_new_storage_dir('fance_nn_experiment')
    PosixPath('/tmp/fance_nn_experiment')
    """
    basedir = Path(os.environ['STORAGE_ROOT']) / experiment_name
    del experiment_name
    return get_new_subdir(**locals())
