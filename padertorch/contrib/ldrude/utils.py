from pathlib import Path
import os
import re
import fnmatch
from natsort import natsorted
import paderbox as pb
import padertorch as pt
import sacred.commands
import functools
from sacred.observers.file_storage import FileStorageObserver


def get_new_folder(
        basedir,
        try_id=None,
        dry_run=False,
        mkdir=True,
        consider_mpi=False,
):
    """

    The core source code if copied from the FileStorageObserver in sacred.

    Get a sub folder from basedir with sacred style.
    Assume integer folder names and select as return folder the last folder
    integer plus one.

    Args:
        basedir:
        try_id: Suggestion for the folder name. Can be used as prefix.
            try_id=prefix with return a folder like: prefix, prefix_2, ...
        dry_run: Per default also creates the directory to be thread safe.
        mkdir: With mkdir this function is thread and process safe.
        consider_mpi: If True only the master selects a folder and syncs the
            folder with the slaves.

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
    basedir = Path(basedir).expanduser()

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

            return simu_dir
        except FileExistsError:
            # Catch race conditions
            if i > 100:
                # After some tries,
                # expect that something other went wrong
                raise


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_last_child(path, glob_string='*', condition_fn=None):
    candidates = [
        path
        for path in natsorted(path.glob(glob_string), key=lambda p: p.name)
        if condition_fn is None or condition_fn(path)
    ]
    if len(candidates) > 0:
        return candidates[-1]
    else:
        return


def get_last_sacred_dir(model_path):
    """Pick the most recent subfolder created by sacred in e.g. a model dir."""
    return get_last_child(
        model_path,
        glob_string='*',
        condition_fn=lambda p: represents_int(p.name)
    )


def has_checkpoints(model_dir):
    checkpoint_dir = model_dir / 'checkpoints'
    return checkpoint_dir.is_dir() and len(checkpoint_dir.glob('ckpt_*')) > 0


def write_makefile_and_config_json(storage_dir, _config, _run):
    config_path = Path(storage_dir) / "config.json"
    pb.io.dump_json(_config, config_path)

    makefile_path = Path(storage_dir) / "Makefile"
    makefile_path.write_text(
        "resume:\n"
        f"\tpython -m {pt.configurable.resolve_main_python_path()} "
        "resume with config.json\n"
    )


def decorator_append_file_storage_observer_with_lazy_basedir(
        experiment: sacred.Experiment,

        *,
        consider_mpi=False,
):
    """
    ToDo: test this

    """
    def wrapper(func):
        captured_func = experiment.capture(func)

        class FileStorageObserverLazyBasedir(FileStorageObserver):
            @property
            @functools.lru_cache()
            def basedir(self):
                basedir = captured_func()
                if basedir is None:
                    raise ValueError(
                        f'Captured function {func} returned None.\n'
                        'Expect that it returns the basedir for sacred.'
                    )
                if not os.path.exists(basedir):
                    os.makedirs(basedir)

                return Path(basedir).expanduser().resolve()

            @basedir.setter
            def basedir(self, value):
                assert value is None, value

            @property
            @functools.lru_cache()
            def resource_dir(self):
                return os.path.join(self.basedir, '_resources')

            @resource_dir.setter
            def resource_dir(self, value):
                assert value is None, value

            @property
            @functools.lru_cache()
            def source_dir(self):
                return os.path.join(self.basedir, '_sources')

            @source_dir.setter
            def source_dir(self, value):
                assert value is None, value

            def __hash__(self):
                # Necessary for functools.lru_cache
                return id(self)

        observer = FileStorageObserverLazyBasedir(
            basedir=None,
            resource_dir=None,
            source_dir=None,
            template=None,
        )

        if consider_mpi:
            from paderbox.utils import mpi
            if mpi.IS_MASTER:
                experiment.observers.append(observer)
        else:
            experiment.observers.append(observer)

        return captured_func

    return wrapper
