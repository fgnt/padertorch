import gzip
import struct
import tempfile
from pathlib import Path

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cached_property

import numpy as np

from lazy_dataset.database import Database


def load_data(src, num_samples, verbose):
    # https://notebooks.azure.com/cntk/projects/tutorials/html/CNTK_103A_MNIST_DataLoader.ipynb
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        if verbose:
            print(f'Downloading {src}')
        gzfname, h = urlretrieve(src, tmpdir / 'delete.me')
        if verbose:
            print('Done.')
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception(
                    f'Invalid file: unexpected magic number. {n[0]}'
                )
            # Read number of entries.
            n = struct.unpack(">I", gz.read(4))[0]
            if n != num_samples:
                raise Exception(
                    f'Invalid file: expected {num_samples} entries.'
                )
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception(
                    'Invalid file: expected 28 rows/cols per image.\n'
                    f'rows:{crow}, cols:{ccol}'
                )
            # Read data.
            res = np.frombuffer(
                gz.read(num_samples * crow * ccol), dtype=np.uint8
            )
    return res.reshape((num_samples, crow, ccol))


def load_labels(src, num_samples, verbose):
    # https://notebooks.azure.com/cntk/projects/tutorials/html/CNTK_103A_MNIST_DataLoader.ipynb
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        if verbose:
            print(f'Downloading {src}')
        gzfname, h = urlretrieve(src, tmpdir / 'delete.me')
        if verbose:
            print('Done.')
        with gzip.open(gzfname) as gz:
            n = struct.unpack("I", gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception(
                    f'Invalid file: unexpected magic number. {n[0]}'
                )
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != num_samples:
                raise Exception(
                    f'Invalid file: expected {num_samples} rows.'
                )
            # Read labels.
            res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        return res


def download(verbose=True):
    """

    >>> from paderbox.notebook import pprint
    >>> pprint(download())
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Done.
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Done.
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Done.
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Done.
    {'train_data': array(shape=(60000, 28, 28), dtype=uint8),
     'train_digits': array(shape=(60000,), dtype=uint8),
     'test_data': array(shape=(10000, 28, 28), dtype=uint8),
     'test_digits': array(shape=(10000,), dtype=uint8)}

    """
    # URLs for the train image and label data
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    # URLs for the test image and label data
    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    return {
        'train_data': load_data(url_train_image, num_train_samples, verbose),
        'train_digits': load_labels(url_train_labels, num_train_samples, verbose),
        'test_data': load_data(url_test_image, num_test_samples, verbose),
        'test_digits': load_labels(url_test_labels, num_test_samples, verbose)
    }


def construct_json(data):

    train_images = (data['train_data'] / 256).astype(np.float32)
    train_digits = data['train_digits'].astype(np.int32)

    test_images = (data['test_data'] / 256).astype(np.float32)
    test_digits = data['test_digits'].astype(np.int32)

    train_examples = {
        f'example_{idx}':
            {'image': image, 'digit': int(digit)}
        for idx, (image, digit) in enumerate(zip(train_images, train_digits))
    }
    test_examples = {
        f'example_{idx}':
            {'image': image, 'digit': int(digit)}
        for idx, (image, digit) in enumerate(zip(test_images, test_digits))
    }
    return {
        'datasets': {
            'train': train_examples,
            'test': test_examples,
        }
    }


class MnistDatabase(Database):
    """
    >>> db = MnistDatabase()
    >>> db.get_dataset('train')
      DictDataset(name='train', len=60000)
    MapDataset(_pickle.loads)
    >>> db.get_dataset('test')
      DictDataset(name='test', len=10000)
    MapDataset(_pickle.loads)
    >>> db.get_dataset('test')[0]['image'].shape
    (28, 28)
    >>> db.get_dataset('test')[0]['digit']
    7
    >>> type(db.get_dataset('test')[0]['digit'])
    <class 'int'>
    """

    def __repr__(self):
        return f'{type(self).__name__}()'

    @cached_property.cached_property
    def data(self):
        from appdirs import user_cache_dir

        path = Path(user_cache_dir('padercontrib')) / 'fgnt_mnist.npz'
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = download(verbose=False)
            np.savez_compressed(path, **data)

        data = np.load(str(path))
        data_dict = dict(data)
        data.close()

        json = construct_json(data_dict)
        return json
