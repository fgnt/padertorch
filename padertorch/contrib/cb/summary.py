
import collections

import torch

import einops
import cached_property

import padertorch as pt

# loss: torch.Tenso r =None,
# losses: dict =None,
# scalars: dict =None,
# histograms: dict =None,
# audios: dict =None,
# images: dict =None,


class ReviewSummary(collections.abc.Mapping):
    """
    >>> review_summary = ReviewSummary()
    >>> review_summary
    ReviewSummary(prefix='', _data={})
    """

    _keys = set(pt.train.hooks.SummaryHook.empty_summary_dict().keys()) | {
        'loss', 'losses'
    }

    def __init__(self, prefix='', _data=None, sampling_rate=None, visible_dB=60):
        if _data is None:
            _data = {}
        self.data = _data
        self.prefix = prefix
        self.sampling_rate = sampling_rate
        self.visible_dB = visible_dB

    def add_to_loss(self, value):
        assert torch.isfinite(value), value
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

    def add_audio(self, name, signal, sampling_rate=None, batch_first=None,
                  normalize=True):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        assert sampling_rate is not None, sampling_rate
        audio = pt.summary.audio(
            signal=signal, sampling_rate=sampling_rate,
            batch_first=batch_first, normalize=normalize
        )
        self.data.setdefault(
            'audios',
            {}
        )[f'{self.prefix}{name}'] = audio

    def add_buffer(self, name, data):
        data = pt.data.batch.example_to_numpy(data, detach=True)
        self.data.setdefault(
            'buffers',
            {}
        ).setdefault(
            f'{self.prefix}{name}',
            []
        ).append(data)

    def add_text(self, name, text):
        assert isinstance(text, str), (type(text), text)
        self.data.setdefault(
            'texts',
            {}
        )[f'{self.prefix}{name}'] = text

    def _rearrange(self, array, rearrange):
        if rearrange is not None:
            return einops.rearrange(array, rearrange)
        else:
            return array

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

    def add_stft_image(
            self, name, signal,
            *, batch_first=None, color='viridis', rearrange=None):
        signal = self._rearrange(signal, rearrange)
        image = pt.summary.stft_to_image(signal, batch_first=batch_first, color=color, visible_dB=self.visible_dB)
        self.add_image(name, image)

    def add_spectrogram_image(
            self, name, signal,
            *, batch_first=None, color='viridis', rearrange=None):
        signal = self._rearrange(signal, rearrange)
        image = pt.summary.spectrogram_to_image(signal, batch_first=batch_first, color=color, visible_dB=self.visible_dB)
        self.add_image(name, image)

    def add_mask_image(self, name, mask, *, batch_first=None, color='viridis', rearrange=None):
        mask = self._rearrange(mask, rearrange)
        image = pt.summary.mask_to_image(mask, batch_first=batch_first, color=color)
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

    def pop(self, *args, **kwargs):
        """pop(key[, default])"""
        return self.data.pop(*args, **kwargs)

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

    class _Plotter:
        def __init__(self, review: 'ReviewSummary'):
            self.review = review

        def image(
                self, key, origin='lower', **kwargs
        ):
            import numpy as np
            import matplotlib.pyplot as plt
            kwargs = {
                'origin': origin,
                **kwargs,
            }
            if key not in self.review['images']:
                from paderbox.utils.mapping import DispatchError
                raise DispatchError(key, self.review['images'].keys())

            X = np.einsum('chw->hwc', self.review['images'][key])

            if origin == 'lower':
                X = X[::-1]
            else:
                assert origin == 'upper'

            # ToDo: Where is AxesImage defined?
            ax: 'plt.AxesImage' = plt.imshow(
                X,
                **kwargs,
            )
            # ax.set_title(key)
            plt.title(key)
            plt.grid(False)
            return ax

        def images(
                self,
                columns=1,
                font_scale=1.0,
                line_width=3,
                figure_size=(8.0, 6.0),
        ):
            from paderbox.visualization import axes_context
            from paderbox.visualization.context_manager import _AxesHandler
            with axes_context(
                    columns=columns,
                    font_scale=font_scale,
                    line_width=line_width,
                    figure_size=figure_size,
            ) as axes:
                axes: _AxesHandler
                for k in self.review['images']:
                    axes.new.grid(False)  # set gca

                    self.image(k)

    @cached_property.cached_property
    def plot(self):
        return self._Plotter(self)

    def play(self, key=None):
        if key is None:
            for k in self['audios'].keys():
                self.play(k)
        elif key in self['audios']:
            from paderbox.io.play import play
            data, sample_rate = self['audios'][key]
            play(data, sample_rate=sample_rate, name=key)
        else:
            from paderbox.utils.mapping import DispatchError
            raise DispatchError(key, self['audios'].keys())
