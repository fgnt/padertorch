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
