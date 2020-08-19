MAKEFILE_TEMPLATE_TRAIN = """SHELL := /bin/bash
MODEL_PATH := "{model_path}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

train:
\tpython -m {main_python_path} with config.json

finetune:
\tpython -m {main_python_path} init_with_new_storage_dir with config.json trainer.model.finetune=True load_model_from=$(MODEL_PATH)/checkpoints/ckpt_latest.pth batch_size=1"

ccsalloc:
\tccsalloc \\
\t\t--res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\
\t\t--time=100h \\
\t\t--stdout=%x.%reqid.trace \\
\t\t--stderr=%x.%reqid.trace \\
\t\t--tracefile=%x.%reqid.trace \\
\t\t-N train_{experiment_name} \\
\t\tpython -m {main_python_path} with config.json

evaluate:
\tpython -m {eval_python_path} init with model_path=$(MODEL_PATH)

evaluate_oracle_num_spk:
\tpython -m {eval_python_path} init with model_path=$(MODEL_PATH) oracle_num_spk=True
"""

MAKEFILE_TEMPLATE_EVAL = """SHELL := /bin/bash

evaluate:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--res=rset=200:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\
\t\t--time=1h \\
\t\t--stdout=%x.%reqid.trace \\
\t\t--stderr=%x.%reqid.trace \\
\t\t--tracefile=trace_%reqid.trace \\
\t\t-N evaluate_{experiment_name} \\
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
