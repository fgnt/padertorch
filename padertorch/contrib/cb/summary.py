

# loss: torch.Tenso r =None,
# losses: dict =None,
# scalars: dict =None,
# histograms: dict =None,
# audios: dict =None,
# images: dict =None,


class ReviewSummary:
    def __init__(self, prefix='', _data=None):
        if _data is None:
            _data = {}
        self.data = _data
        self.prefix = prefix

    def add_to_loss(self, value):
        if 'loss' in self.data:
            self.data['loss'] += value
        else:
            self.data['loss'] = value

    def add_scalar(self, name, value):
        # Save the mean of all added values
        self.data.setdefault(
            'scalars',
            {}
        ).setdefault(
            f'{self.prefix}{name}',
            []
        ).append(value)

    def add_image(self, name, value):
        # Save the last added value
        self.data.setdefault(
            'images',
            {}
        )[f'{self.prefix}{name}'] = value

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        if item == 'scalars':
            return {
                k: sum(v) / len(v)
                for k, v in self.data[item].items()
            }
        elif item == 'images':
            return self.data[item]
        else:
            raise KeyError()

    def __setitem__(self, key, value):
        self.data[key] = value

    def get(self, item, default):
        if item in self:
            return self.data[item]
        else:
            return default

    def setdefault(self, key, default):
        self.data.setdefault(key, default)
