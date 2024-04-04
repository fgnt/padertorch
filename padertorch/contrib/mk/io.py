import os
from pathlib import Path
from typing import List


# https://stackoverflow.com/a/59803793/16085876
def run_fast_scandir(dir: Path, ext: List[str]):
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(Path(f.path))


    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files
