from dataclasses import dataclass, asdict, field
from padertorch.base import Module
from padertorch.configurable import Configurable


__all__ = [
    'Parameterized',
    'BuildingBlock',
    'dict_func'
]

def dict_func(in_dict):
    return field(default_factory=lambda: in_dict)

class Parameterized(Configurable):
    @dataclass
    class Options:
        pass

    def __init__(self, **kwargs):
        super().__init__()
        if 'opts' in kwargs:
            self.opts = kwargs['opts']
            assert hasattr(self.opts, '__dataclass_fields__')
        else:
            self.opts = self.Options(**kwargs)

    def __repr__(self):
        return f'{type(self).__name__}:\n{str(self.opts)}'

    @classmethod
    def get_signature(cls):
        opts_dict = super().get_signature()
        opts_dict.update(asdict(cls.Options()))
        return opts_dict


class BuildingBlock(Parameterized, Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        super(Parameterized).__init__()
        self.build()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def build(self, *args, **kwargs):
        pass
