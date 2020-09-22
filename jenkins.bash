#!/usr/bin/env bash

# This file is only required for internal testing
# cd dirname "$(readlink -f "$0")"

git clone https://github.com/fgnt/paderbox

# include common stuff (installation of toolbox, paths, traps, nice level...)
source paderbox/jenkins_common.bash

# Cuda
source paderbox/bash/cuda.bash

pip install --user -e .[test]

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation

# nosetests --with-xunit --with-coverage --cover-package=padertorch -v -w "tests" # --processes=-1
pytest -v "tests/" "padertorch/"
# Use as many processes as you have cores: --processes=-1
# Acording to https://gist.github.com/hangtwenty/1aeb36ee85f4bdce0899
# `--cov-report term` solves the problem that doctests are not included
# in coverage

# Export coverage
python -m coverage xml --include="padertorch*"

# Pylint tests
pylint --rcfile="paderbox/pylint.cfg" -f parseable padertorch > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes padertorch

# copy html code to lighttpd webserver
# rsync -a --delete-after /var/lib/jenkins/jobs/python_toolbox/workspace/toolbox/doc/build/html/ /var/www/doku/html/python_toolbox/
