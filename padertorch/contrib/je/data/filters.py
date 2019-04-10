import abc

from padertorch.configurable import Configurable
from padertorch.utils import to_list


class Filter(Configurable, abc.ABC):
    """
    Base class for callable transformations. Not intended to be instantiated.
    """
    @abc.abstractmethod
    def __call__(self, example, training=False):
        raise NotImplementedError


class DiscardLabelsFilter(Filter):
    def __init__(self, key, names):
        self.key = key
        self.names = to_list(names)

    def __call__(self, example, training=False):
        return not any([name in to_list(example[self.key]) for name in self.names])


class RestrictLabelsFilter(Filter):
    def __init__(self, key, names):
        self.key = key
        self.names = to_list(names)

    def __call__(self, example, training=False):
        return any([name in to_list(example[self.key]) for name in self.names])
