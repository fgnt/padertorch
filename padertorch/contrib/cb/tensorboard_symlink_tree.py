"""
Create a symlink tree for all specified files in the current folder.

    python -m padertorch.contrib.cb.tensorboard_symlink_tree ../*/*tfevents* --max_age=1days

Usecase:

Tensorboard does a recursive search for all tfevent files.
In many cases this works good and is better than this workaround.

When you have a slow recursive search, this script can be used as a workaround.
This can be caused by a slow filesystem (usually remote) and to many files
inside the tensorboard (e.g. a Kaldi experiment folder).

The problem of tensorboard in this case is, that it does not support either
multiple tfevent in the command line interface (only one is supported) or a
customisation for the search pattern of the event files (e.g. limited depth
search).

This workaround mirrors the folder tree, but only for the files that are the
input of this file. In the commandline you can use bash wildcards like `*`:

    python -m padertorch.contrib.cb.tensorboard_symlink_tree ../*/*tfevents*

This command creates a symlinks to all tfevent that match the pattern
`../*/*tfevents*` in the current folder.
Sadly, this command has to be executed each time, you create a new experiment.
Because of this I created a Makefile in that folder:

    .../tensorboard$ cat Makefile
    symlink_tree1day:
        find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
        python -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents* --max_age=1days

    symlink_tree:
        find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
        python -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents*

    tensorboard:
        date && $(cd .../tensorboard && ulimit -v 10000000 && tensorboard --bind_all -v 1 --logdir=. --port=...) && date || date

"""

import os
from pathlib import Path
import datetime

import paderbox as pb


def main(*files, prefix=None, max_age=None):
    if prefix is None:
        prefix = os.path.commonpath(files)
    print('Common Prefix', prefix)
    print('Create')

    files = [Path(f) for f in files]

    if max_age is not None:
        # Panda import is slow, but pd.Timedelta
        # accepts many styles for time
        # (e.g. '1day')
        import pandas as pd
        max_age = pd.Timedelta(max_age)
        now = pd.Timestamp('now')

        files = sorted(files, key=lambda file: file.stat().st_mtime)

    for file in files:
        link_name = file.relative_to(prefix)
        if max_age is not None:
            last_modified = file.stat().st_mtime
            last_modified = datetime.datetime.fromtimestamp(last_modified)

            if max_age > now - last_modified:
                # Create symlink if it doesn't exist.
                pass
            else:
                if not link_name.is_symlink():
                    print(f'Skip {file}, it is {now - last_modified} > {max_age} old.')
                continue

        link_name.parent.mkdir(exist_ok=True)
        source = os.path.relpath(file, link_name.parent)
        if not link_name.exists():
            print(f'\t{link_name} -> {source}')

        # Create symlink if it does not exist,
        # or check that the symlink point to the
        # same file.
        pb.io.symlink(source, link_name)
    print('Finish')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
