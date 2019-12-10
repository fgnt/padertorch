from pathlib import Path

import cached_property
import numpy as np

from lazy_dataset.database import DictDatabase

class MnistDatabase(DictDatabase):
    """
    >>> db = MnistDatabase()
    >>> db.get_dataset('train')
      ExamplesIterator(name=train, len=60000)
    MapIterator(<built-in function loads>)
    >>> db.get_dataset('test')
      ExamplesIterator(name=test, len=10000)
    MapIterator(<built-in function loads>)
    >>> db.get_dataset('test')[0]['image'].shape
    (28, 28)
    """
    def __init__(self):
        # Do not call super
        pass

    def __repr__(self):
        return f'{type(self).__name__}()'

    @cached_property.cached_property
    def database_dict(self):
        from paderbox.database.mnist.create_json import download, construct_json
        from appdirs import user_cache_dir

        path = Path(user_cache_dir('paderbox')) / 'fgnt_mnist.npz'

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = download()
            np.savez_compressed(path, **data)

        data = np.load(str(path))
        data_dict = dict(data)
        data.close()

        json = construct_json(data_dict)
        return json