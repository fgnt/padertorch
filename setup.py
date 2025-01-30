"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
import os
# Allow editable install into user site directory.
# See https://github.com/pypa/pip/issues/7953.
import site
import sys
site.ENABLE_USER_SITE = '--user' in sys.argv[1:]

# To use a consistent encoding
from codecs import open
# Always prefer setuptools over distutils
from distutils.core import setup
from os import path

# import numpy
from setuptools import find_packages
# from Cython.Build import cythonize

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# testing dependencies
test = [
    'pytest',
    'pytest-cov',
    'mock',
    'coverage',
    'pylint',
    'sacred',
    'appdirs',
    'protobuf3_to_dict',
    'torchvision',
    'matplotlib',  # padertorch.summary.tbx_utils use mpl for colorize
    'pb_bss @ git+http://github.com/fgnt/pb_bss',
    'torch_complex',  # https://github.com/kamo-naoyuki/pytorch_complex
    'pyyaml',
    'humanize',
    'codecarbon',
]

if os.environ.get('SETUP_PY_IGNORE_GIT_DEPENDENCIES', False):
    # Remove git dependencies for sdist, because they are not supported on
    # pypi.
    # Can't have direct dependency: pb_bss@ git+http://github.com/fgnt/pb_bss ; extra == "test".
    # https://packaging.python.org/specifications/core-metadata for more information.
    test = [
        d
        for d in test
        if '@ git+http://' not in d
    ]

setup(
    name='padertorch',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='A collection of common functionality to simplify the design, '
    'training and evaluation of machine learning models based on pytorch '
    'with an emphasis on speech processing.',
    long_description=long_description,

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # The project's main homepage.
    url='https://github.com/fgnt/padertorch/',

    # Author details
    author='Department of Communications Engineering, Paderborn University',
    author_email='sek@nt.upb.de',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.10',
    ],

    # What does your project relate to?
    keywords='pytorch, audio, speech',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'torch',
        'tensorboardX',
        'einops',
        'tqdm',
        'natsort',
        'lazy_dataset',
        'IPython',
        'paderbox',
    ],

    # Installation problems in a clean, new environment:
    # 1. `cython` and `scipy` must be installed manually before using
    # `pip install`
    # 2. `pyzmq` has to be installed manually, otherwise `pymatbridge` will
    # complain

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'test': test,
        'all': test,
    },

    # ext_modules=cythonize(
    #     [],
    #     annotate=True,
    # ),
    # include_dirs=[numpy.get_include()],
)
