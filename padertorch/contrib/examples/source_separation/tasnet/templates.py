MAKEFILE_TEMPLATE_TRAIN = """SHELL := /bin/bash
MODEL_PATH := $(shell pwd)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

train:
    python -m {main_python_path} with config.json

ccsalloc:
    ccsalloc \\
        --notifyuser=awe \\
        --res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\
        --time=100h \\
        --join \\
        --stdout=stdout \\
        --tracefile=%x.%reqid.trace \\
        -N train_{experiment_name} \\
        python -m {main_python_path} with config.json

evaluate:
    python -m {eval_python_path} init with model_path=$(MODEL_PATH)"""

MAKEFILE_TEMPLATE_EVAL = """SHELL := /bin/bash

evaluate:
    python -m {main_python_path} with config.json

ccsalloc:
    ccsalloc \\
        --notifyuser=awe \\
        --res=rset=100:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\
        --time=1h \\
        --join \\
        --stdout=stdout \\
        --tracefile=trace_%reqid.trace \\
        -N evaluate_{nickname} \\
        ompi \\
        -x STORAGE \\
        -x NT_MERL_MIXTURES_DIR \\
        -x NT_DATABASE_JSONS_DIR \\
        -x KALDI_ROOT \\
        -x LD_PRELOAD \\
        -x CONDA_EXE \\
        -x CONDA_PREFIX \\
        -x CONDA_PYTHON_EXE \\
        -x CONDA_DEFAULT_ENV \\
        -x PATH \\
        -- \\
        python -m {main_python_path} with config.json
"""