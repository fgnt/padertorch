#!/usr/bin/env bash

# This file is only required for internal testing
# cd dirname "$(readlink -f "$0")"

git clone https://github.com/fgnt/paderbox

# include common stuff (installation of toolbox, paths, traps, nice level...)
source toolbox/jenkins_common.bash

# Cuda
source toolbox/bash/cuda.bash

pip install --user -e .

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation

# nosetests --with-xunit --with-coverage --cover-package=padertorch -v -w "tests" # --processes=-1
pytest --junitxml='test_results.xml' --cov=padertorch  \
  --doctest-modules --doctest-continue-on-failure --cov-report term -v "tests/" # --processes=-1
# Use as many processes as you have cores: --processes=-1
# Acording to https://gist.github.com/hangtwenty/1aeb36ee85f4bdce0899
# `--cov-report term` solves the problem that doctests are not included
# in coverage

# Export coverage
python -m coverage xml --include="padertorch*"

# Pylint tests
pylint --rcfile="toolbox/pylint.cfg" -f parseable paderbox > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes padertorch

# copy html code to lighttpd webserver
# rsync -a --delete-after /var/lib/jenkins/jobs/python_toolbox/workspace/toolbox/doc/build/html/ /var/www/doku/html/python_toolbox/
