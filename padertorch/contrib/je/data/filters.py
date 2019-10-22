from padertorch.utils import to_list


class DiscardLabelsFilter:
    def __init__(self, key, names):
        self.key = key
        self.names = to_list(names)

    def __call__(self, example):
        return not any([name in to_list(example[self.key]) for name in self.names])


class RestrictLabelsFilter:
    def __init__(self, key, names):
        self.key = key
        self.names = to_list(names)

    def __call__(self, example):
        return any([name in to_list(example[self.key]) for name in self.names])
