MAKEFILE_TEMPLATE_TRAIN = """
SHELL := /bin/bash

train:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--res=rset=1:ncpus=4:gtx1080=1:ompthreads=1 \\
\t\t--time=100h \\
\t\t--stdout=%x.%reqid.out \\
\t\t--stderr=%x.%reqid.err \\
\t\t--tracefile=%x.%reqid.trace \\
\t\t-N train_{nickname} \\
\t\tpython -m {main_python_path} with config.json
"""

MAKEFILE_TEMPLATE_EVAL = """
SHELL := /bin/bash

evaluate:
\tpython -m {main_python_path} with config.json

ccsalloc:
\tccsalloc \\
\t\t--res=rset=200:mpiprocs=1:ncpus=1:mem=4g:vmem=6g \\
\t\t--time=1h \\
\t\t--stdout=%x.%reqid.out \\
\t\t--stderr=%x.%reqid.err \\
\t\t--tracefile=%x.%reqid.trace \\
\t\t-N evaluate_{nickname} \\
\t\tompi ${{OMPI_PARAMS}}\\
\t\t-- \\
\t\tpython -m {main_python_path} with config.json
"""
