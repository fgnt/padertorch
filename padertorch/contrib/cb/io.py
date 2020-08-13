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


class Makefile:
    def __init__(self, data: dict=None):
        """
        >>> m = Makefile()
        >>> m.add_last_ccs_tail_cmd()
        >>> m.add_sacred_resume_cmd()
        >>> m.add_ccsalloc()
        >>> m.add_restart_cmd()
        >>> m.add_raw('evalfolder := $(patsubst checkpoints/%.pth,eval/%,$(wildcard checkpoints/*.pth))')
        >>> print(m.text.replace('\\t', '    '))
        tail:
            $(eval trace_file := $(shell ls *.trace | sort | tail -n 1))
            $(eval err_file := $(subst .trace,.err,$(trace_file)))
            $(eval out_file := $(subst .trace,.out,$(trace_file)))
            tail -F $(trace_file) $(err_file) $(out_file)
        <BLANKLINE>
        resume:
            python -m pycharm.docrunner with resume with config.yaml
        <BLANKLINE>
        ccsalloc:
            ccsalloc --stdout=log_%reqid.out --stderr=log_%reqid.err --tracefile=log_%reqid.trace --group=hpc-prf-nt1 --res=rset=40:mem=2G:ncpus=1 -t 2h --name=docrunner ompi -- make run
        <BLANKLINE>
        run:
            python -m pycharm.docrunner with config.yaml
        <BLANKLINE>
        restart:
            python -m pycharm.docrunner /home/cbj/python/pytorch_sanity/padertorch/contrib/cb/io.py::Makefile::__init__
        <BLANKLINE>
        evalfolder := $(patsubst checkpoints/%.pth,eval/%,$(wildcard checkpoints/*.pth))

        """
        self.globals = []

        if data is None:
            self.data = {}
        else:
            self.data = {**data}

    def __setitem__(self, alias, value):
        assert isinstance(alias, str), (type(alias), alias)
        self.data[alias] = value

    # def module_name(self):
    #     return pt.configurable.resolve_main_python_path()

    def add_raw(self, value):
        self.data[object()] = value

    @property
    def commands(self):
        return tuple([
            k.split(':')[0]
            for k in self.data.keys()
        ])

    def __contains__(self, item):
        return item in self.commands

    @property
    def text(self):
        text = []

        if len(self.globals) > 0:
            text.append('\n'.join(self.globals))

        for k, v in self.data.items():
            if type(k) == object:
                text.append(f'{v}')
            else:
                if isinstance(v, (tuple, list)):
                    v = '\n'.join(v)
                v = textwrap.indent(v, '\t')

                if ':' not in k:
                    k = f'{k}:'
                text.append(f'{k}\n{v}')

        return '\n\n'.join(text) + '\n'

    def add_last_ccs_tail_cmd(self, alias='tail', prefix=''):
        """
        >>> m = Makefile()
        >>> m.add_last_ccs_tail_cmd()
        >>> print(m.text.replace('\\t', '    '))
        tail:
            $(eval trace_file := $(shell ls *.trace | sort | tail -n 1))
            $(eval err_file := $(subst .trace,.err,$(trace_file)))
            $(eval out_file := $(subst .trace,.out,$(trace_file)))
            tail -F $(trace_file) $(err_file) $(out_file)
        <BLANKLINE>
        >>> m = Makefile()
        >>> m.add_last_ccs_tail_cmd(prefix='log/')
        >>> print(m.text.replace('\\t', '    '))
        tail:
            $(eval trace_file := $(shell ls log/*.trace | sort | tail -n 1))
            $(eval err_file := $(subst .trace,.err,$(trace_file)))
            $(eval out_file := $(subst .trace,.out,$(trace_file)))
            tail -F $(trace_file) $(err_file) $(out_file)
        <BLANKLINE>
        """

        # When replacing `*.trace` with another glob (e.g. `eval/*/*.trace`)
        # and the folders that match will have the wrong order, the `sort`
        # command must be replaced (e.g. `sort -t "/" -k 3`).
        # The argument
        # '$(eval trace_file := $(shell ls eval/*/*.trace | sort -t "/" -k 3 | tail -n 1))',

        self[alias] = [
            f'$(eval trace_file := $(shell ls {prefix}*.trace | sort | tail -n 1))',
            '$(eval err_file := $(subst .trace,.err,$(trace_file)))',
            '$(eval out_file := $(subst .trace,.out,$(trace_file)))',
            'tail -F $(trace_file) $(err_file) $(out_file)',
        ]

    def add_raw_ccsalloc(
            self,
            alias,
            cmd,
            name=None,
            *,
            rset='1:mem=2G:ncpus=1',
            time='2h',
            group='hpc-prf-nt1',
            ccsalloc_options=tuple(),

            stdout='log_%reqid.out',
            stderr='log_%reqid.err',
            tracefile='log_%reqid.trace',
    ):
        if name is None:
            name = alias

        if isinstance(cmd, str):
            cmd = (cmd,)

        self[alias] = ' '.join([
            'ccsalloc',
            f'--stdout={stdout}',
            f'--stderr={stderr}',
            f'--tracefile={tracefile}',
            *ccsalloc_options,
            f'--group={group}',
            f'--res=rset={rset}',
            '-t', f'{time}',
            f'--name={name}',
            *cmd,
        ])

    def add_ompi_ccsalloc(
            self,
            alias,
            cmd,
            name=None,
            *,
            rset='40:mem=2G:ncpus=1',
            time='2h',
            group='hpc-prf-nt1',
            ccsalloc_options=tuple(),
    ):
        if name is None:
            name = alias

        if isinstance(cmd, str):
            cmd = (cmd,)

        self[alias] = ' '.join([
            'ccsalloc',
            '--stdout=log_%reqid.out',
            '--stderr=log_%reqid.err',
            '--tracefile=log_%reqid.trace',
            *ccsalloc_options,
            f'--group={group}',
            f'--res=rset={rset}',
            '-t', f'{time}',
            f'--name={name}',
            'ompi',
            '--',
            *cmd,
        ])

    def add_ccsalloc(
            self,
            alias='ccsalloc',
            group='hpc-prf-nt1',
            rset='40:mem=2G:ncpus=1',
            time='2h',
            name=None,
            ccsalloc_options=tuple(),
            python_cmd='run',
            config_name='config.yaml',
    ):
        """
        >>> m = Makefile()
        >>> m.add_ccsalloc(ccsalloc_options=['--notifyjob=XCPU,60m'])
        >>> print(m.text.replace('\\t', '    '))
        ccsalloc:
            ccsalloc --stdout=log_%reqid.out --stderr=log_%reqid.err --tracefile=log_%reqid.trace --notifyjob=XCPU,60m --group=hpc-prf-nt1 --res=rset=40:mem=2G:ncpus=1 -t 2h --name=docrunner ompi -- make run
        <BLANKLINE>
        run:
            python -m pycharm.docrunner with config.yaml
        """
        # '--notifyjob=XCPU,60m',

        if name is None:
            name = Path(
                pt.configurable.resolve_main_python_path()
            ).suffix.replace('.', '')
        self[alias] = ' '.join([
            'ccsalloc',
            '--stdout=log_%reqid.out',
            '--stderr=log_%reqid.err',
            '--tracefile=log_%reqid.trace',
            *ccsalloc_options,
            f'--group={group}',
            f'--res=rset={rset}',
            '-t', f'{time}',
            f'--name={name}',
            'ompi',
            '--',
            'make',
            f'{python_cmd}',
        ])

        if python_cmd not in self:
            module_name = pt.configurable.resolve_main_python_path()
            self[python_cmd] = ' '.join([
                'python',
                '-m',
                f'{module_name}',
                'with',
                f'{config_name}',
            ])

    def add_sacred_resume_cmd(
            self,
            alias='resume',
            sacred_cmd='resume',
            config_name='config.yaml',
    ):
        """
        >>> m = Makefile()
        >>> m.add_sacred_resume_cmd()
        >>> print(m.text.replace('\\t', '    '))
        resume:
            python -m pycharm.docrunner resume with config.yaml
        <BLANKLINE>
        >>> m.add_sacred_resume_cmd('makefile', 'makefile')
        >>> print(m.text.replace('\\t', '    '))
        resume:
            python -m pycharm.docrunner resume with config.yaml
        <BLANKLINE>
        makefile:
            python -m pycharm.docrunner makefile with config.yaml
        <BLANKLINE>
        """
        module_name = pt.configurable.resolve_main_python_path()
        self[alias] = [
            f'python -m {module_name} {sacred_cmd} with {config_name}'
        ]

    def add_restart_cmd(self, alias='restart'):
        """
        >>> m = Makefile()
        >>> m.add_restart_cmd()
        >>> print(m.text.replace('\\t', '    '))
        restart:
            python -m pycharm.docrunner .../padertorch/contrib/cb/io.py::Makefile::add_restart_cmd
        """
        import shlex, sys
        # Maybe add an option to select "python -m module.path" or
        # "python module/path.py"

        restart_cmd = (
                f'python -m {pt.configurable.resolve_main_python_path()} '
                + ' '.join(map(shlex.quote, sys.argv[1:]))
        )
        self[alias] = restart_cmd


import contextlib
import textwrap


@contextlib.contextmanager
def makefile(
        folder,
        when_exist: ['fail', 'backup', 'append', 'overwrite'] = 'fail',
):
    file = Path(folder) / 'Makefile'
    append = False

    backup = False
    if when_exist == 'backup':
        if file.exists():
            backup = True
    elif when_exist == 'append':
        append = True
    elif when_exist == 'overwrite':
        append = True
    elif when_exist == 'fail':
        if file.exists():
            raise FileExistsError(
                f'Remove the Makefile {file} befor write a new one.\n'
                'Alternatively set when_exist to "backup", "append" or'
                '"overwrite"'
            )
    else:
        raise ValueError(when_exist)

    m = Makefile()
    yield m

    if backup:
        now = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        file.rename(Path(folder) / f'Makefile_{now}')

    with file.open(
        mode={
            True: 'a',
            False: 'w',
        }[append]
    ) as fd:
        fd.write(m.text)


def write_makefile_and_config(
        storage_dir,
        _config,
        _run,
        backend='yaml',
        write_config=True,
        write_makefile=True,
):
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
    from padertorch.configurable import recursive_class_to_str
    if backend == 'json':
        config_path = Path(storage_dir) / "config.json"
        if write_config:
            pb.io.dump_json(recursive_class_to_str(_config), config_path)
    elif backend == 'yaml':
        config_path = Path(storage_dir) / "config.yaml"
        if write_config:
            pb.io.dump_yaml(_config, config_path)
    else:
        raise ValueError(backend)

    if write_makefile:
        makefile_path = Path(storage_dir) / "Makefile"

        module_name = pt.configurable.resolve_main_python_path()

        makefile_path.write_text(
            "resume:\n"
            f"\tpython -m {module_name} "
            f"resume with {config_path.name}\n"
        )


if __name__ == '__main__':
    import sys
    print(sys.modules['__main__'])
    print(pt.configurable.resolve_main_python_path())
