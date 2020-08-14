import os
import re
import time
import fnmatch
import datetime
import urllib.request
from pathlib import Path


def get_new_folder(
        basedir,
        *,
        id_type='index',
        mkdir=True,
        try_id=None,
        force_suffix=False,
        consider_mpi=False,
        dry_run=False,
):
    """

    Args:
        basedir:
            The new folder will be inside this directory
        id_type:
            The id type that is used for the folder name.
             - index:
                The largest index in basedir + 1.
                e.g.: '1', '2', ...
             - time: A timestamp with the format %Y-%m-%d-%H-%M-%S
                e.g. '2020-08-13-17-02-57'
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
        try_id:
            When not None use this as prefix for the new folder name.
            First try f'{try_id}' before continuing with f'{try_id}_2'.
        force_suffix:
            If True, use f'{try_id}_1' instead of f'{try_id}' as first try when
            try_id is not None.
        consider_mpi:
            If True, only search on one mpi process for the folder and
            distribute the folder name.
            When using mpi with this option disables the following can/will
            happen
             - When mkdir is True every process will get another folder.
               i.e. each process has a folder just for this process.
             - When mkdir is False it is likely that all get the same folder,
               but this is not guarantied. Never use mpi with consider_mpi and
               mkdir disabled.
        dry_run:
            Similar to mkdir. When true, disable mkdir and print the folder
            name.

    Returns:
        pathlib.Path of the new folder


    >>> get_new_folder('/', dry_run=True)  # root folder usually contain no digits
    dry_run: "os.mkdir(/1)"
    PosixPath('/1')

    >>> import numpy as np
    >>> np.random.seed(0)  # This is for doctest. Never use it in practise.
    >>> get_new_folder('/', id_type=NameGenerator(), dry_run=True)
    dry_run: "os.mkdir(/helpful_tomato_finch)"
    PosixPath('/helpful_tomato_finch')
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
        if dry_run:
            print(f'dry_run: "os.makedirs({basedir})"')
            # ToDo: Make this working.
            #       Will fail when calling os.listdir
        else:
            basedir.mkdir(parents=True)

    if Path('/net') in basedir.parents:
        # If nt filesystem, assert not in /net/home
        assert Path('/net/home') not in basedir.parents, basedir

    for i in range(200):
        if id_type == 'index':
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
        elif id_type == 'time':
            if i != 0:
                time.sleep(1)
            _id = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        elif callable(id_type):
            _id = id_type()
        else:
            raise ValueError(id_type)

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

            return simu_dir
        except FileExistsError:
            # Catch race conditions
            if i > 100:
                # After some tries,
                # expect that something other went wrong
                raise


def get_basedir(
        experiment_name
):
    """
    >>> os.environ['STORAGE_ROOT'] = '/tmp'  # simulate enviroment variable for doctest
    >>> get_basedir('fance_nn_experiment')
    PosixPath('/tmp/fance_nn_experiment')
    """
    return Path(os.environ['STORAGE_ROOT']) / experiment_name


def _get_list_from_unique_names_generator(name_type, overwrite_cache=False):
    """
    Download a list from
    https://github.com/andreasonny83/unique-names-generator
    and cache the result in `~/.cache/padertorch/unique_names_generator`.

    >>> _get_list_from_unique_names_generator('adjectives')[:3]
    ['average', 'big', 'colossal']
    >>> _get_list_from_unique_names_generator('colors')[:3]
    ['amaranth', 'amber', 'amethyst']
    >>> _get_list_from_unique_names_generator('animals')[:3]
    ['canidae', 'felidae', 'cat']
    >>> _get_list_from_unique_names_generator('names')[:3]
    ['Aaren', 'Aarika', 'Abagael']
    >>> _get_list_from_unique_names_generator('countries')[:3]
    ['Afghanistan', 'Åland Islands', 'Albania']
    >>> _get_list_from_unique_names_generator('star-wars')[:3]
    ['Luke Skywalker', 'C-3PO', 'R2-D2']

    """
    # adjectives.ts
    # animals.ts
    # colors.ts
    # countries.ts
    # names.ts
    # star-wars.ts
    # index.ts    -> Does not work (Is not dictionary)
    # numbers.ts  -> Does not work (Cannot be parsed)
    from appdirs import user_cache_dir
    import paderbox as pb

    file = (
            Path(user_cache_dir('padertorch'))
            / 'unique_names_generator' / f'{name_type}.json'
    )
    if overwrite_cache or (not file.exists()):
        url = (
            f'https://raw.githubusercontent.com/andreasonny83/'
            f'unique-names-generator/master/src/dictionaries/{name_type}.ts'
        )

        try:
            resource = urllib.request.urlopen(url)
        except Exception as e:
            # ToDo: use the following api to list the names
            # https://api.github.com/repos/andreasonny83/unique-names-generator/git/trees/master?recursive=1
            raise ValueError(
                f'Tried to download {name_type!r}.\nCould not open\n{url}\n'
                'See in\n'
                'https://github.com/andreasonny83/unique-names-generator/tree/master/src/dictionaries\n'
                'for valid names.'
            )
        # https://stackoverflow.com/a/19156107/5766934
        data = resource.read().decode(resource.headers.get_content_charset())

        data = re.findall("'(.*?)'", data)

        pb.io.dump(data, file)
    else:
        data = pb.io.load(file)
    return data


class NameGenerator:
    """
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> ng = NameGenerator()
    >>> ng()
    'helpful_tomato_finch'
    >>> ng.possibilities()  # With 28 million a collision is unlikely
    28406196
    >>> ng = NameGenerator(['adjectives', 'animals'])
    >>> ng()
    'colourful_tern'
    
    """
    def __init__(
            self,
            lists=('adjectives', 'colors', 'animals'),
            separator='_',
            rng=None,
            replace=True,
            # style='default',  # 'capital', 'upperCase', 'lowerCase'
    ):
        self.lists = [
            _get_list_from_unique_names_generator(l)
            if isinstance(l, str) else l
            for l in lists
        ]
        self.separator = separator
        if rng is None:
            import numpy as np
            rng = np.random
        self.rng = rng
        self.replace = replace
        self.seen = set()

    def __call__(self):
        for _ in range(1000):
            name_parts = [
                self.rng.choice(d)
                for d in self.lists
            ]
            name = self.separator.join(name_parts)

            if self.replace:
                break
            elif name in self.seen:
                continue
            else:
                self.seen.add(name)
                break
        else:
            raise RuntimeError("Couldn't find a new name.")

        return name

    def possibilities(self):
        import numpy as np
        return np.prod([[len(d) for d in self.lists]])
