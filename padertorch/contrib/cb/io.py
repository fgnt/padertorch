import os
import re
import fnmatch
from pathlib import Path

import paderbox as pb
import padertorch as pt

def get_new_folder(
        basedir,
        try_id=None,
        dry_run=False,
        consider_mpi=False,
        chdir=False,
        mkdir=True,
):
    """

    Args:
        basedir:
        try_id:
        mkdir: Enables thread safety
        dry_run: Per default also creates the directory to be thread safe.

    Returns:

    """
    if consider_mpi:
        if pb.utils.mpi.IS_MASTER:
            pass
        else:
            new_folder = None
            new_folder = pb.utils.mpi.COMM.bcast(new_folder)
            return new_folder

    suggested_id = try_id
    basedir = Path(basedir).expanduser().resolve()

    if Path('/net') in basedir.parents:
        # If nt filesystem, assert not in /net/home
        assert Path('/net/home') not in basedir.parents, basedir

    for i in range(200):
        if suggested_id is None:
            dir_nrs = [
                int(d)
                for d in os.listdir(str(basedir))
                if (basedir / d).is_dir() and d.isdigit()
            ]
            _id = max(dir_nrs + [0]) + 1
        else:
            if (basedir / f'{suggested_id}').exists():
                dir_nrs = [
                    int(re.sub(f'{suggested_id}_?', '', str(d)))
                    for d in os.listdir(str(basedir))
                    if (basedir / d).is_dir()
                    if fnmatch.fnmatch(d, f'{suggested_id}_*')
                    if re.sub(f'{suggested_id}_?', '', str(d)).isdigit()
                ]
                _id = max(dir_nrs + [1]) + 1
                _id = f'{suggested_id}_{_id}'
            else:
                _id = f'{suggested_id}'

        simu_dir = basedir / str(_id)

        try:
            if dry_run:
                print(f'dry_run: "os.mkdir({simu_dir})"')
            elif mkdir is False:
                pass
            elif mkdir is True:
                simu_dir.mkdir()
            else:
                raise ValueError(mkdir)

            if consider_mpi:
                assert pb.utils.mpi.IS_MASTER, pb.utils.mpi.RANK
                simu_dir = pb.utils.mpi.COMM.bcast(simu_dir)

            if chdir:
                os.chdir(simu_dir)

            return simu_dir
        except FileExistsError:
            # Catch race conditions
            if i > 100:
                # After some tries,
                # expect that something other went wrong
                raise


def write_makefile_and_config(storage_dir, _config, _run, backend='yaml'):
    """
    Writes a Makefile and a config file in the storage_dir to resume the
    Experiment.

    Args:
        storage_dir:
        _config:
        _run:
        backend:

    Returns:

    """
    if backend == 'json':
        config_path = Path(storage_dir) / "config.json"
        pb.io.dump_json(_config, config_path)
    elif backend == 'yaml':
        import yaml
        config_path = Path(storage_dir) / "config.yaml"
        config_path.write_text(yaml.dump(_config))
    else:
        raise ValueError(backend)

    makefile_path = Path(storage_dir) / "Makefile"
    makefile_path.write_text(
        "resume:\n"
        f"\tpython -m {pt.configurable.resolve_main_python_path()} "
        f"resume with {config_path.name}\n"
    )
