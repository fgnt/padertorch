#!/usr/bin/env bash

# cd dirname "$(readlink -f "$0")"

git clone git@ntgit.upb.de:python/toolbox

# include common stuff (installation of toolbox, paths, traps, nice level...)
source toolbox/jenkins_common.bash

pip install --user -e .

# Unittets
# It seems, that jenkins currentliy does not work with matlab: Error: Segmentation violation

nosetests --with-xunit --with-coverage --cover-package=padertorch -v -w "tests" # --processes=-1
# Use as many processes as you have cores: --processes=-1

# Export coverage
python -m coverage xml --include="padertorch*"

# Pylint tests
pylint --rcfile="toolbox/pylint.cfg" -f parseable paderbox > pylint.txt
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes padertorch

# copy html code to lighttpd webserver
# rsync -a --delete-after /var/lib/jenkins/jobs/python_toolbox/workspace/toolbox/doc/build/html/ /var/www/doku/html/python_toolbox/
