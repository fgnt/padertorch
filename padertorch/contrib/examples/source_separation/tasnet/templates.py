MAKEFILE_TEMPLATE_TRAIN = """SHELL := /bin/bash
MODEL_PATH := $(shell pwd)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

train:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--notifyuser=awe \\
\t\t--res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\
\t\t--time=100h \\
\t\t--join \\
\t\t--stdout=stdout \\
\t\t--tracefile=%x.%reqid.trace \\
\t\t-N train_{experiment_name} \\
\t\tpython -m {main_python_path} with config.json

evaluate:
\tpython -m {eval_python_path} init with model_path=$(MODEL_PATH)"""

MAKEFILE_TEMPLATE_EVAL = """SHELL := /bin/bash

evaluate:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--notifyuser=awe \\
\t\t--res=rset=100:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\
\t\t--time=1h \\
\t\t--join \\
\t\t--stdout=stdout \\
\t\t--tracefile=trace_%reqid.trace \\
\t\t-N evaluate_{nickname} \\
\t\tompi \\
\t\t-x STORAGE \\
\t\t-x NT_MERL_MIXTURES_DIR \\
\t\t-x NT_DATABASE_JSONS_DIR \\
\t\t-x KALDI_ROOT \\
\t\t-x LD_PRELOAD \\
\t\t-x CONDA_EXE \\
\t\t-x CONDA_PREFIX \\
\t\t-x CONDA_PYTHON_EXE \\
\t\t-x CONDA_DEFAULT_ENV \\
\t\t-x PATH \\
\t\t-- \\
\t\tpython -m {main_python_path} with config.json
"""