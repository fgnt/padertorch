
# PyPi upload

Package a Python Package/ version bump See: https://packaging.python.org/tutorials/packaging-projects/

1. Update `setup.py` to new version number
2. Commit this change
3. Tag and upload

## pypirc

Example `~/.pypirc` (see https://packaging.python.org/en/latest/specifications/pypirc/)
```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = <PyPI token>

[testpypi]
username = __token__
password = <TestPyPI token>
```

## Install dependencies:
```bash
pip install --upgrade setuptools
pip install --upgrade wheel
pip install --upgrade twine
# pip install --upgrade bleach html5lib  # some versions do not work
pip install --upgrade bump2version
```

`bump2version` takes care to increase the version number, create the commit and tag.


## Publish

```bash
export SETUP_PY_IGNORE_GIT_DEPENDENCIES=1
bump2version --verbose --tag patch  # major, minor or patch
python setup.py sdist  # bdist_wheel  # It is difficult to get bdist_wheel working with binary files
git push origin --tags
# Wait for the github action to build the windows wheels, ToDO: Fix wheels.
twine upload --repository testpypi dist/*  # 
twine upload dist/*
git push
```
