
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
    >>> review_summary = ReviewSummary()
    >>> review_summary
    ReviewSummary(prefix='', _data={})
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

    def add_scalar(self, name, *value):
        # Save the mean of all added values
        value = pt.data.batch.example_to_numpy(value, detach=True)
        self.data.setdefault(
            'scalars',
            {}
        ).setdefault(
            f'{self.prefix}{name}',
            []
        ).extend(value)

    def add_text(self, name, text):
        assert isinstance(text, str), (type(text), text)
        self.data.setdefault(
            'texts',
            {}
        )[f'{self.prefix}{name}'] = text

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

    def add_stft_image(self, name, signal, batch_first=False, color='viridis'):
        image = pt.summary.stft_to_image(signal, batch_first=batch_first, color=color)
        self.add_image(name, image)

    def add_spectrogram_image(self, name, signal, batch_first=False, color='viridis'):
        image = pt.summary.spectrogram_to_image(signal, batch_first=batch_first, color=color)
        self.add_image(name, image)

    def add_mask_image(self, name, mask, batch_first=False):
        image = pt.summary.mask_to_image(mask, batch_first=batch_first)
        self.add_image(name, image)

    def add_histogram(self, name, values):
        value = pt.utils.to_numpy(values, detach=True)
        self.data.setdefault(
            'histograms',
            {}
        ).setdefault(
            f'{self.prefix}{name}',
            []
        ).append(value)

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

    def __repr__(self):
        return f'{self.__class__.__name__}(prefix={self.prefix!r}, _data={dict(self)!r})'

    def _repr_pretty_(self, p, cycle):
        """
        >>> review_summary = ReviewSummary()
        >>> review_summary.add_to_loss(1)
        >>> review_summary.add_scalar('abc', 2)
        >>> review_summary
        ReviewSummary(prefix='', _data={'loss': 1, 'scalars': {'abc': [2]}})
        >>> from IPython.lib.pretty import pprint
        >>> pprint(review_summary)
        ReviewSummary(prefix='', _data={'loss': 1, 'scalars': {'abc': [2]}})
        >>> pprint(review_summary, max_width=79-18)
        ReviewSummary(
            prefix='',
            _data={'loss': 1, 'scalars': {'abc': [2]}}
        )
        >>> pprint(review_summary, max_width=79-40)
        ReviewSummary(
            prefix='',
            _data={'loss': 1,
                   'scalars': {'abc': [2]}}
        )
        """
        if cycle:
            p.text(f'{self.__class__.__name__}(...)')
        else:
            txt = f'{self.__class__.__name__}('
            with p.group(4, txt, ''):
                p.breakable(sep='')
                p.text('prefix=')
                p.pretty(self.prefix)
                p.text(',')
                p.breakable()
                txt = '_data='
                with p.group(len(txt), txt, ''):
                    p.pretty(dict(self))
            p.breakable('')
            p.text(')')
