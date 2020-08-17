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
        Use as basedir:
            `os.environ['STORAGE_ROOT'] / experiment_name`
        and return:
            `os.environ['STORAGE_ROOT'] / experiment_name / ID`

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
