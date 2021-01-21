"""
Create a symlink tree for all specified files in the current folder.

    python -m padertorch.contrib.cb.tensorboard_symlink_tree ../*/*tfevents*

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
    symlink_tree:
        python -m padertorch.contrib.cb.tensorboard_symlink_tree ../*/*tfevents*

    tensorboard:
        date && $(cd .../tensorboard && ulimit -v 10000000 && tensorboard --bind_all -v 1 --logdir=. --port=...) && date || date

"""

import os
from pathlib import Path

import paderbox as pb


def main(*files, prefix=None):
    if prefix is None:
        prefix = os.path.commonpath(files)
    print('Common Prefix', prefix)
    for file in files:
        file = Path(file)
        link_name = file.relative_to(prefix)
        link_name.parent.mkdir(exist_ok=True)
        pb.io.symlink(os.path.relpath(file, link_name.parent), link_name)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
