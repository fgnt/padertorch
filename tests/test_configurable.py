import padertorch as pts
import numpy as np


def foo(b=1, c=2):
    pass


def bar(a, b=3, d=4):
    pass


class A(pts.configurable.Configurable):
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

        # with np.testing.assert_raises_regex(TypeError, "missing keys: {'a'}"):
        #     config = A.get_config({'e': {'factory': bar}})

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