import padertorch as pt
import numpy as np
import pytest


def foo(b=1, c=2):
    pass


def bar(a, b=3, d=4):
    pass


class A(pt.configurable.Configurable):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['e'] = {
            'factory': foo,
            'b': 5
        }
        if config['e']['factory'] == foo:
            config['e']['c'] = 6
        if config['e']['factory'] == bar:
            config['e']['d'] = 7
        return config

    def __init__(self, e, f=0):
        pass


class Test:
    def test_(self):
        config = A.get_config()
        expect = {
            'factory': 'tests.test_configurable.A',
            'f': 0,
            'e': {
                'factory': 'tests.test_configurable.foo',
                'b': 5,
                'c': 6
            }
        }
        np.testing.assert_equal(config, expect)

        with np.testing.assert_raises_regex(Exception, "missing keys: {'a'}"):
            config = A.get_config({'e': {'factory': bar}})

        config = A.get_config({'e': {'factory': bar, 'a': 10}})
        expect = {
            'factory': 'tests.test_configurable.A',
            'f': 0,
            'e': {
                'factory': 'tests.test_configurable.bar',
                'b': 5,
                'd': 7,
                'a': 10
            }
        }
        np.testing.assert_equal(config, expect)

        config = A.get_config({'e': {'factory': bar, 'a': 10}})
        expect = {
            'factory': 'tests.test_configurable.A',
            'f': 0,
            'e': {
                'factory': 'tests.test_configurable.bar',
                'b': 5,
                'd': 7,
                'a': 10
            }
        }
        np.testing.assert_equal(config, expect)


class B(pt.Configurable):

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['a'] = 1
        config['b'] = 2  # Should raise an Exception

    def __init__(self, a):
        pass


def test_wrong_finalize_dogmatic_config():
    # Doctest may raise an error. This is expected.
    """
    >>> B.get_config()
    Traceback (most recent call last):
    ...
    Exception: Tried to set an unexpected keyword argument for <class 'tests.test_configurable.B'> in finalize_dogmatic_config.
    See details below and stacktrace above.
    <BLANKLINE>
    Too many keywords for the factory <class 'tests.test_configurable.B'>.
    Redundant keys: {'b'}
    Signature: (a)
    Current config with fallbacks:
    NestedChainMap({'factory': tests.test_configurable.B}, {'a': 1, 'b': 2})
    >>> B.get_config(updates={'C': 3})
    Traceback (most recent call last):
    ...
    Exception: padertorch.Configurable.get_config(updates=...) got an unexpected keyword argument in updates for <class 'tests.test_configurable.B'>.
    See details below.
    <BLANKLINE>
    Too many keywords for the factory <class 'tests.test_configurable.B'>.
    Redundant keys: {'C'}
    Signature: (a)
    Current config with fallbacks:
    NestedChainMap({'C': 3, 'factory': tests.test_configurable.B}, {})
    """

    import doctest
    # doctest.run_docstring_examples does not raise an Exception
    # doctest.run_docstring_examples(
    #     test_wrong_finalize_dogmatic_config, globals(),
    #     optionflags=doctest.ELLIPSIS,
    # )

    f = test_wrong_finalize_dogmatic_config
    globs = globals()
    verbose = False
    name = "NoName",
    compileflags = None
    optionflags = doctest.ELLIPSIS

    finder = doctest.DocTestFinder(verbose=verbose, recurse=False)
    runner = doctest.DocTestRunner(verbose=verbose, optionflags=optionflags)
    for test in finder.find(f, name, globs=globs):
        test_results: doctest.TestResults = runner.run(
            test, compileflags=compileflags)

        assert not test_results.failed, (test_results, 'See above stdout')
