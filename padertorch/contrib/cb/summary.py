
import collections

import padertorch as pt

# loss: torch.Tenso r =None,
# losses: dict =None,
# scalars: dict =None,
# histograms: dict =None,
# audios: dict =None,
# images: dict =None,


class ReviewSummary(collections.Mapping):
    """
    >>> ReviewSummary()
    """

    _keys = set(pt.train.hooks.SummaryHook.empty_summary_dict().keys()) | {
        'loss', 'losses'
    }

    def __init__(self, prefix='', _data=None):
        if _data is None:
            _data = {}
        self.data = _data
        self.prefix = prefix

    def add_to_loss(self, value):
        if 'loss' in self.data:
            self.data['loss'] = self.data['loss'] + value
        else:
            self.data['loss'] = value

    def add_scalar(self, name, value):
        # Save the mean of all added values
        value = pt.utils.to_numpy(value, detach=True)
        self.data.setdefault(
            'scalars',
            {}
        ).setdefault(
            f'{self.prefix}{name}',
            []
        ).append(value)

    def add_image(self, name, image):
        # Save the last added value
        image = pt.utils.to_numpy(image, detach=True)
        if image.ndim != 3:
            raise AssertionError(
                'Did you forgot to call "pt.summary.*_to_image"?\n'
                f'Expect ndim == 3, got shape {image.shape}.'
            )
        self.data.setdefault(
            'images',
            {}
        )[f'{self.prefix}{name}'] = image

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, key):
        assert key in self._keys, (key, self._keys)
        return self.data[key]

    def __setitem__(self, key, value):
        assert key in self._keys, (key, self._keys)
        self.data[key] = value

    def get(self, item, default):
        if item in self:
            return self.data[item]
        else:
            return default

    def setdefault(self, key, default):
        self.data.setdefault(key, default)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)