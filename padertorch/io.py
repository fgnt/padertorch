import os
import re
import fnmatch
import datetime
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
        force_suffix=False,
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
        import dlp_mpi
        if dlp_mpi.IS_MASTER:
            pass
        else:
            new_folder = None
            new_folder = dlp_mpi.bcast(new_folder)
            return new_folder

    suggested_id = try_id
    basedir = Path(basedir).expanduser().resolve()
    if not basedir.exists():
        basedir.mkdir(parents=True)

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
            if force_suffix or (basedir / f'{suggested_id}').exists():
                dir_nrs = [
                    int(re.sub(f'{suggested_id}_?', '', str(d)))
                    for d in os.listdir(str(basedir))
                    if (basedir / d).is_dir()
                    if fnmatch.fnmatch(d, f'{suggested_id}_*')
                    if re.sub(f'{suggested_id}_?', '', str(d)).isdigit()
                ]
                if force_suffix:
                    dir_nrs += [0]
                else:
                    dir_nrs += [1]

                _id = max(dir_nrs) + 1
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
                import dlp_mpi
                assert dlp_mpi.IS_MASTER, dlp_mpi.RANK
                simu_dir = dlp_mpi.bcast(simu_dir)

            if chdir:
                os.chdir(simu_dir)

            return simu_dir
        except FileExistsError:
            # Catch race conditions
            if i > 100:
                # After some tries,
                # expect that something other went wrong
                raise
