from pathlib import Path
import numpy as np

import cached_property

from lazy_dataset.database import Database

class MnistDatabase(Database):
    """
    >>> db = MnistDatabase()
    >>> db.get_dataset('train')
      DictDataset(len=60000)
    MapDataset(_pickle.loads)
    >>> db.get_dataset('test')
      DictDataset(len=10000)
    MapDataset(_pickle.loads)
    >>> db.get_dataset('test')[0]['image'].shape
    (784,)
    """

    def __repr__(self):
        return f'{type(self).__name__}()'

    @cached_property.cached_property
    def data(self):
        import torchvision.datasets as datasets
        from appdirs import user_cache_dir

        path = Path(user_cache_dir('padertorch')) / 'data'

        path.parent.mkdir(parents=True, exist_ok=True)
        train_set = datasets.MNIST(
            root=str(path), train=True, download=True, transform=None)
        test_set = datasets.MNIST(
            root=str(path), train=False, download=True, transform=None)

        def create_dict(image, digit):
            if not isinstance(digit, int):
                digit = digit.cpu().numpy().astype(np.int32)
            return dict(image=(np.array(image.getdata()) / 256).astype(np.float32),
                        digit=digit)

        out_dict = {
            'datasets':{
                'train': {f'example_{idx}': create_dict(*train_set[idx])
                          for idx in range(len(train_set))},
                'test': {f'example_{idx}': create_dict(*test_set[idx])
                         for idx in range(len(test_set))}
            }
        }
        del train_set
        del test_set
        return out_dict