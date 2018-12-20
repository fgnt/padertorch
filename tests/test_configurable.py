import padertorch as pts
import numpy as np


def foo(b=1, c=2):
    pass


def bar(a, b=3, d=4):
    pass


class A(pts.configurable.Configurable):
    @classmethod
    def get_signature(cls):
        defaults = super().get_signature()
        defaults['e'] = {
            'cls': foo,
            'kwargs': {'b': 5},
            foo: {'c': 6},
            bar: {'d': 7},
        }
        return defaults

    def __init__(self, e, f=0):
        pass


class Test:
    def test_(self):
        config = A.get_config()
        expect = {
            'cls': 'tests.test_configurable.A',
            'kwargs': {
                'f': 0,
                'e': {
                    'cls': 'tests.test_configurable.foo',
                    'kwargs': {
                        'b': 5,
                        'c': 6
                    }
                }
            }
        }
        np.testing.assert_equal(config, expect)

        with np.testing.assert_raises_regex(TypeError, "missing a required argument: 'a'"):
            config = A.get_config({'e': {'cls': bar}})

        config = A.get_config({'e': {'cls': bar, 'kwargs': {'a': 10}}})
        expect = {
            'cls': 'tests.test_configurable.A',
            'kwargs': {
                'f': 0,
                'e': {
                    'cls': 'tests.test_configurable.bar',
                    'kwargs': {
                        'b': 5,
                        'd': 7,
                        'a': 10
                    }
                }
            }
        }
        np.testing.assert_equal(config, expect)
