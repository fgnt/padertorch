import enum
import types

import numpy as np
import torch

_empty_dict = types.MappingProxyType({})


class ExampleToDevice:
    """

    >>> from paderbox.utils.pretty import pprint
    >>> example = {'a': 1, 'b': [np.zeros([2, 2]), np.zeros(3, np.complex128)]}
    >>> pprint(ExampleToDevice()(example, 'cpu'))
    {'a': 1,
     'b': [tensor([[0., 0.],
            [0., 0.]], dtype=torch.float64),
      array([0.+0.j, 0.+0.j, 0.+0.j])]}

    >>> import torch_complex
    >>> pprint(ExampleToDevice(
    ...     category_register={
    ...     ExampleToDevice.category.ComplexNumpy:
    ...       lambda x, d: torch_complex.ComplexTensor(x, device=d)}
    ... )(example, 'cpu'))
    {'a': 1,
     'b': [tensor([[0., 0.],
            [0., 0.]], dtype=torch.float64),
      ComplexTensor(
          real=tensor([0., 0., 0.], dtype=torch.float64),
          imag=tensor([0., 0., 0.], dtype=torch.float64),
      )]}

    >>> pprint(KamoExampleToDevice()(example, 'cpu'))
    {'a': 1,
     'b': [tensor([[0., 0.],
            [0., 0.]], dtype=torch.float64),
      ComplexTensor(
          real=tensor([0., 0., 0.], dtype=torch.float64),
          imag=tensor([0., 0., 0.], dtype=torch.float64),
      )]}


    """
    def __init__(self, type_register={}, category_register={}): # noqa
        self.type_register = {
            dict: self.mapping,
            tuple: self.sequence,
            list: self.sequence,
            **type_register,
        }

        self.category_register = {
            self.category.Tensor: self.tensor,
            self.category.Numpy: self.numpy,
            self.category.ComplexNumpy: lambda x, d: x,
            self.category.Dataclass: self.dataclass,
            self.category.DoNothing: lambda x, d: x,
            **category_register,
        }

    class category(enum.Enum):
        Tensor = enum.auto()
        Numpy = enum.auto()
        ComplexNumpy = enum.auto()
        Dataclass = enum.auto()
        DoNothing = enum.auto()

    @classmethod
    def get_category(cls, example):
        if isinstance(example, np.ndarray):
            if example.dtype.kind == 'c':
                return cls.category.ComplexNumpy
            else:
                return cls.category.Numpy
        elif torch.is_tensor(example):
            return cls.category.Tensor
        elif hasattr(example, '__dataclass_fields__'):
            return cls.category.Dataclass
        else:
            return cls.category.DoNothing

    def __call__(self, example, device=None):
        try:
            func = self.type_register[type(example)]
        except KeyError:
            category = self.get_category(example)
            try:
                func = self.category_register[category]
            except KeyError:
                raise TypeError(
                    f"The example {example!r} is categorized as '{category}'.\n"
                    f'This category has no registered function.'
                ) from None
        return func(example, device)

    def mapping(self, example, device):
        return example.__class__({
            key: self(value, device=device)
            for key, value in example.items()
        })

    def sequence(self, example, device):
        return example.__class__([
            self(element, device=device)
            for element in example
        ])

    def numpy(self, example, device):
        return self(torch.from_numpy(example), device=device)

    def tensor(self, example, device):
        return example.to(device=device)

    def dataclass(self, example, device):
        return example.__class__(**{
            f: self(getattr(example, f), device=device)
            for f in example.__dataclass_fields__
        })


class KamoExampleToDevice(ExampleToDevice):
    """
    Converts complex numpy to torch_complex.ComplexTensor
    """
    def __init__(self, type_register={}, category_register={}):  # noqa
        super().__init__(
            type_register,
            {
                self.category.ComplexNumpy: self.complex_numpy,
                **category_register
            },
        )

    def complex_numpy(self, example, device):
        import torch_complex
        return torch_complex.ComplexTensor(example, device=device)


class ExampleToDeviceNativeComplex(ExampleToDevice):
    def __init__(self, type_register={}, category_register={}):  # noqa
        super().__init__(
            type_register,
            {
                self.category.ComplexNumpy: self.numpy,
                **category_register
            },
        )


def add_batch_dim_to_dataset(dataset, batch_size, length_key='num_samples'):
    import padertorch as pt

    if batch_size is None:
        return dataset

    assert isinstance(batch_size, int), (type(batch_size), batch_size)
    dataset = dataset.batch(batch_size)
    if batch_size > 1:
        dataset = dataset.map(pt.data.batch.Sorter(length_key))
    else:
        assert batch_size == 1, batch_size

    dataset = dataset.map(pt.data.utils.collate_fn)

    return dataset
