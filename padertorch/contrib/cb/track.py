"""

How I use this code:

from padertorch.contrib.cb.track import track, tracker_list, ShapeTracker, ParameterTracker, TimeTracker, GPUMemTracker, OBackwardMemTracker
with track(trainer.model, tracker_list(
        ShapeTracker,
        ParameterTracker,
        TimeTracker,
        GPUMemTracker,
        OBackwardMemTracker,
)) as trackers:
    trainer.test_run(...)
Path(log_file).write_text(str(trackers))

"""


import contextlib
import dataclasses
import typing
import weakref
import time
import re
import itertools

import torch

__all__ = [
    'track',
    'Tracker',
    'tracker_list',
    # Examples:
    'ShapeTracker',
    'DTypeTracker',
    'DeviceTracker',
    'ParameterTracker',
    'TimeTracker',
    'CPUMemTracker',
    'GPUMemTracker',
    'IOPMemTracker',
    'IOPNumTracker',
]


class Tracker:
    def __init__(self, count, depth, leaf, shared_dict, module):
        """

        Args:
            count: Running index starts with zero
            depth: The call depth
            leaf: Whether this tracker works on a leaf.
                  it might be a real leaf or a module that should be considerd
                  as leaf.
            shared_dict: A dict that is shared between all Trackers.
        """
        self.count = count
        self.depth = depth
        self.leaf = leaf
        self.shared_dict = shared_dict
        self.module = module

    def pre(self, module, input) -> None:
        pass

    def post(self, module, input, output) -> None:
        pass

    @property
    def prefix(self):
        name = ' ' * self.depth * 2 + self.module.__class__.__name__ + ':'
        return f'{self.count:3} {name:20}'

    @property
    def data(self):
        """

        Returns:
            dict:
                header: Column header
                align: left ('<'), right('>') or center ('^')
                data: Data in column
            Example:
                {
                    'header': ['input', '', 'output'],
                    'align': '^^^',
                    'data': [self.input_shape, '->', 'self.output_shape'],
                }

        """
        raise NotImplementedError()

    def __repr__(self):
        return f'{self.prefix} {" ".join([str(d) for d in self.data["data"]])}'


class track:
    def __init__(
            self,
            net: torch.nn.Module,
            tracker_factory: typing.Callable[[int, int, bool, dict], Tracker],
            leaf_types=tuple(),
    ):
        """

        Args:
            net:
            tracker_factory:
            leaf_types:

        Returns:

        >>> import psutil, os
        >>> from torch.nn import Sequential, ReLU, Linear
        >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
        >>> net
        Sequential(
          (0): Linear(in_features=3, out_features=1000, bias=True)
          (1): ReLU()
          (2): Sequential(
            (0): Linear(in_features=1000, out_features=2, bias=True)
            (1): ReLU()
          )
        )

        >>> with track(net, ShapeTracker) as trackers:
        ...     _ = net(torch.randn(7, 3))
        >>> for t in trackers:
        ...     print(t)
          0 Sequential:          ([7, 3],) -> [7, 2]
          1   Linear:            ([7, 3],) -> [7, 1000]
          2   ReLU:              ([7, 1000],) -> [7, 1000]
          3   Sequential:        ([7, 1000],) -> [7, 2]
          4     Linear:          ([7, 1000],) -> [7, 2]
          5     ReLU:            ([7, 2],) -> [7, 2]
        >>> print(trackers)
                                    input         output
          0 Sequential:           ([7, 3],)   ->  [7, 2]
          1   Linear:             ([7, 3],)   -> [7, 1000]
          2   ReLU:              ([7, 1000],) -> [7, 1000]
          3   Sequential:        ([7, 1000],) ->  [7, 2]
          4     Linear:          ([7, 1000],) ->  [7, 2]
          5     ReLU:             ([7, 2],)   ->  [7, 2]

        >>> with track(net, ParameterTracker) as trackers:
        ...     _ = net(torch.randn(7, 3))
        >>> print(trackers)
                                 #Params
          0 Sequential:                0
          1   Linear:              4_000
          2   ReLU:                    0
          3   Sequential:              0
          4     Linear:            2_002
          5     ReLU:                  0

        >>> sum([t.num_params for t in trackers])
        6002

        """
        self.net = net
        if isinstance(tracker_factory, (tuple, list)):
            tracker_factory = tracker_list(*tracker_factory)
        self.tracker_factory = tracker_factory
        self.leaf_types = leaf_types

        self.shared_dict = {}
        self.all_trackers = []
        self.tracker_stack = []
        self.hooks = []

    def __enter__(self):
        self.apply_filtered(self.net, self.register_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()

    def __getitem__(self, item):
        return self.all_trackers[item]

    def register_hook(self, module, leaf):
        def pre_hook(module, input):
            tracker = self.tracker_factory(
                len(self.all_trackers),
                len(self.tracker_stack),
                leaf,
                self.shared_dict,
                module,
            )
            tracker.pre(module, input)
            self.tracker_stack.append(tracker)
            self.all_trackers.append(tracker)

        def hook(module, input, output):
            tracker = self.tracker_stack.pop()
            tracker.post(module, input, output)

        self.hooks.append(module.register_forward_pre_hook(pre_hook))
        self.hooks.append(module.register_forward_hook(hook))

    def apply_filtered(self, module, fn):
        is_leaf = True
        for child in module.children():
            is_leaf = False
            if child.__class__ in self.leaf_types:
                fn(child, leaf=True)
            else:
                self.apply_filtered(child, fn)
        fn(module, leaf=is_leaf)
        return module

    @staticmethod
    def as_tabular_data(data, align: '"<^>"' = "<", sep=' ', max_cell_width=40):
        """
        >>> import string
        >>> print(track.as_tabular_data([['ab', 'abcde'], ['abcd', 'abc']]))
        ab   abcde
        abcd abc
        >>> print(track.as_tabular_data([
        ...     [string.ascii_letters, string.ascii_uppercase],
        ...     [string.digits, string.ascii_lowercase]], max_cell_width=20))
        abcdefghijklmnopqrst ABCDEFGHIJKLMNOPQRST
        uvwxyzABCDEFGHIJKLMN UVWXYZ
        OPQRSTUVWXYZ
        0123456789           abcdefghijklmnopqrst
                             uvwxyz
        """
        column_width = {}

        data = [[str(entry) for entry in row] for row in data]

        # Add linebreaks
        rows = []
        r = re.compile(f'(.{{{max_cell_width}}})')
        for row in data:
            if max([len(e) for e in row]) > max_cell_width:
                rows.extend(itertools.zip_longest(*[
                    [part for part in r.split(cell) if part]
                    for cell in row
                ], fillvalue=''))
            else:
                rows.append(row)
        data = rows

        for row in data:
            for i, entry in enumerate(row):
                column_width[i] = max(len(str(entry)), column_width.get(i, -1))

        rows = []
        for row in data:
            line = []
            for i, entry in enumerate(row):
                try:
                    a = align[i]
                except IndexError:
                    a = align[-1]
                line.append(f'{entry:{a}{column_width[i]}}')
            rows.append(sep.join(line).rstrip(' '))
        return '\n'.join(rows)

    def __repr__(self):
        assert len(self.all_trackers), self.all_trackers
        rows = [
            [' ', *self.all_trackers[0].data['header']]
        ]
        for t in self.all_trackers:
            rows.append([t.prefix, *t.data['data']])
        return self.as_tabular_data(rows, '<' + self.all_trackers[0].data['align'])


def tracker_list(*tracker_factories):
    """

    Args:
        *tracker_factories:

    Returns:

    >>> from torch.nn import Sequential, ELU, Linear
    >>> net = Sequential(Linear(3, 1000), ELU(), Sequential(Linear(1000, 2), ELU()))
    >>> with track(net, tracker_list(ShapeTracker, ParameterTracker)) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> print(trackers)
                                input         output   #Params
      0 Sequential:           ([7, 3],)   ->  [7, 2]         0
      1   Linear:             ([7, 3],)   -> [7, 1000]   4_000
      2   ELU:               ([7, 1000],) -> [7, 1000]       0
      3   Sequential:        ([7, 1000],) ->  [7, 2]         0
      4     Linear:          ([7, 1000],) ->  [7, 2]     2_002
      5     ELU:              ([7, 2],)   ->  [7, 2]         0

    Manual print:
    >>> for ts in zip(*trackers):
    ...     for t in ts:
    ...         print(t)
    ...     print()
      0 Sequential:          ([7, 3],) -> [7, 2]
      1   Linear:            ([7, 3],) -> [7, 1000]
      2   ELU:               ([7, 1000],) -> [7, 1000]
      3   Sequential:        ([7, 1000],) -> [7, 2]
      4     Linear:          ([7, 1000],) -> [7, 2]
      5     ELU:             ([7, 2],) -> [7, 2]
    <BLANKLINE>
      0 Sequential:          0
      1   Linear:            4_000
      2   ELU:               0
      3   Sequential:        0
      4     Linear:          2_002
      5     ELU:             0
    <BLANKLINE>

    You can run the following in a ipynb with GPU

    import torch
    from torch.nn import Sequential, ELU, Linear
    from padertorch.contrib.cb.track import track, GPUMemTracker, ShapeTracker, tracker_list, ParameterTracker

    net = Sequential(Linear(3, 1000), ELU(), Sequential(Linear(1000, 2), ELU()))
    net.to(torch.device(0))

    with track(net, tracker_list(ShapeTracker, ParameterTracker, GPUMemTracker)) as trackers:
        _ = net(torch.randn(7, 3).to(torch.device(0)))

    for ts in zip(*trackers):
        for t in ts:
            print(t)
        print()

    0 Sequential          : ([7, 3],) -> [7, 2]
    1   Linear            : ([7, 3],) -> [7, 1000]
    2   ELU               : ([7, 1000],) -> [7, 1000]
    3   Sequential        : ([7, 1000],) -> [7, 2]
    4     Linear          : ([7, 1000],) -> [7, 2]
    5     ELU             : ([7, 2],) -> [7, 2]

    0 Sequential          : 0
    1   Linear            : 4000
    2   ELU               : 0
    3   Sequential        : 0
    4     Linear          : 2002
    5     ELU             : 0

    0 Sequential          : 57344 B
    1   Linear            : 28160 B
    2   ELU               : 28160 B
    3   Sequential        : 1024 B
    4     Linear          : 512 B
    5     ELU             : 512 B

    """
    class TrackerList(Tracker):
        def __init__(self, count, depth, leaf, shared_dict, module):
            super().__init__(count, depth, leaf, shared_dict, module)
            self.instances = [
                tf(count, depth, leaf, shared_dict.setdefault(i, {}), module)
                for i, tf in enumerate(tracker_factories)
            ]

        def pre(self, module, input):
            for i in self.instances:
                i.pre(module, input)

        def post(self, module, input, output):
            for i in self.instances:
                i.post(module, input, output)

        @property
        def data(self):
            data = {
                'header': [],
                'align': '',
                'data': [],
            }
            for i in self.instances:
                i_data = i.data
                assert len(i_data['header']) == len(i_data['data']), (len(i_data['header']), len(i_data['align']), len(i_data['data']))
                assert len(i_data['header']) == len(i_data['align']), (len(i_data['header']), len(i_data['align']), len(data))
                for k in data.keys():
                    data[k] += i_data[k]
            return data

        def __getitem__(self, item):
            return self.instances[item]

    return TrackerList


###############################################################################
# Examples                                                                    #
###############################################################################


class ShapeTracker(Tracker):

    def get_shape(self, obj):
        if isinstance(obj, tuple):
            return tuple([self.get_shape(e) for e in obj])
        elif isinstance(obj, list):
            return [self.get_shape(e) for e in obj]
        elif isinstance(obj, dict):
            return {
                k: shape
                for k, v in obj.items()
                for shape in [self.get_shape(v)]
                if shape is not None
            }
        else:
            try:
                return list(obj.shape)
            except AttributeError:
                return '?'

    def pre(self, module, input):
        self.input_shape = self.get_shape(input)

    def post(self, module, input, output):
        self.output_shape = self.get_shape(output)

    @property
    def data(self):
        return {
            'header': ['input', '', 'output'],
            'align': '^^^',
            'data': [self.input_shape, '->', self.output_shape],
        }


class DTypeTracker(ShapeTracker):
    def get_shape(self, obj):
        if isinstance(obj, (tuple, list, dict)):
            return super().get_shape(obj)
        else:
            try:
                return obj.dtype
            except AttributeError:
                return 'unknown'


class DeviceTracker(ShapeTracker):
    def get_shape(self, obj):
        if isinstance(obj, (tuple, list, dict)):
            return super().get_shape(obj)
        else:
            try:
                return str(obj.device)
                # str: 'cuda:0'
                # repr: device(type='cuda', index=0)
                # The input will always use repr, hence convert to str.
            except AttributeError:
                return 'unknown'


class ParameterTracker(Tracker):
    def pre(self, module, input):
        pass

    def post(self, module, input, output):
        self.num_params = sum([
            p.numel() for p in module.parameters(recurse=self.leaf)])

    @property
    def data(self):
        return {
            'header': ['#Params'],
            'align': '>',
            'data': [f'{self.num_params:_}'],
        }

def get_ParameterTracker(
        name='#Params',
        num_or_bytes='num',
        include_require_grad=True,
        include_not_require_grad=True,
):
    class ParameterTracker(Tracker):
        def pre(self, module, input):
            pass

        def post(self, module, input, output):
            self.num_params = sum([
                p.numel() for p in module.parameters(recurse=self.leaf)])

        @property
        def data(self):
            return {
                'header': ['#Params'],
                'align': '>',
                'data': [f'{self.num_params:_}'],
            }


class TimeTracker(Tracker):
    """

    >>> from torch.nn import Sequential, ReLU, Linear
    >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
    >>> with track(net, TimeTracker) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> print(trackers)  # doctest: +SKIP
                                               Time
      0 Sequential:           0.0005062560085207224
      1   Linear:            0.00027760001830756664
      2   ReLU:               8.528295438736677e-05
      3   Sequential:          9.90619882941246e-05
      4     Linear:           5.891895852982998e-05
      5     ReLU:            1.8481980077922344e-05
    """
    timestamp = time.perf_counter  # time.process_time

    def pre(self, module, input):
        self.start = self.timestamp()

    def post(self, module, input, output):
        self.end = self.timestamp()

    @property
    def data(self):
        return {
            'header': ['Time'],
            'align': '>',
            'data': [f'{self.end - self.start}'],
        }


class CPUMemTracker(Tracker):
    """
    WARNING: This class tracks the memory consumption of the process, not the
             memory of torch.

    >>> from torch.nn import Sequential, ReLU, Linear
    >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
    >>> with track(net, CPUMemTracker) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> for t in trackers:  # doctest: +SKIP
    ...      print(t)
      0 Sequential:          1_724_416 B
      1   Linear:            1_724_416 B
      2   ReLU:              0 B
      3   Sequential:        0 B
      4     Linear:          0 B
      5     ReLU:            0 B
    """
    def get_mem(self):
        # return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss

    def pre(self, module, input):
        self.pre_mem = self.get_mem()

    def post(self, module, input, output):
        self.post_mem = self.get_mem()

    @property
    def data(self):
        return {
            'header': ['CPU Mem'],
            'align': '>',
            'data': [f'{self.post_mem - self.pre_mem:_} B'],
        }


class GPUMemTracker(Tracker):
    """

    Doctest runs on CPU, hence the GPU memory tracking is boring

    >>> from torch.nn import Sequential, ReLU, Linear
    >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
    >>> with track(net, GPUMemTracker) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> print(trackers)
                             CPU Mem
      0 Sequential:              0 B
      1   Linear:                0 B
      2   ReLU:                  0 B
      3   Sequential:            0 B
      4     Linear:              0 B
      5     ReLU:                0 B

cc
    """
    device = 0  # Use export CUDA_VISIBLE_DEVICES=1 to switch device

    def get_mem(self):
        return torch.cuda.memory_allocated(device=self.device)

    def pre(self, module, input):
        self.pre_mem = self.get_mem()

    def post(self, module, input, output):
        self.post_mem = self.get_mem()

    @property
    def data(self):
        return {
            'header': ['GPU Mem'],
            'align': '>',
            'data': [f'{self.post_mem - self.pre_mem:_} B'],
        }


class GPUTotPreMemTracker(GPUMemTracker):
    @property
    def data(self):
        return {
            'header': ['GPU Pre Mem'],
            'align': '>',
            'data': [f'{self.pre_mem:_} B'],
        }


class GPUTotPostMemTracker(GPUMemTracker):
    @property
    def data(self):
        return {
            'header': ['GPU Post Mem'],
            'align': '>',
            'data': [f'{self.post_mem:_} B'],
        }


class IOPNumTracker(Tracker):
    """
    Input Output Parameter Number of element Tracker.

    Tracks the number of values in the input (I) and output (O) and also the
    number of parameters (P) each layer has. Further it tracks, if the values
    need a gradient.
    The method `total_repr` can be used on any instance to print the total
    number.
    Note: For `total_repr` considers, that some tensors apear multiple times
          (e.g. as input and output), so the total numbers are smaller than
          the sum of the individual numbers.

    >>> from torch.nn import Sequential, ReLU, Linear
    >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
    >>> with track(net, IOPNumTracker) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> print(trackers)
                             #Parameters                       #IO Tensors
      0 Sequential:          P:      0 (requires_grad:      0) IO:     35 (requires_grad:     14)
      1   Linear:            P:   4000 (requires_grad:   4000) IO:   7021 (requires_grad:   7000)
      2   ReLU:              P:      0 (requires_grad:      0) IO:  14000 (requires_grad:  14000)
      3   Sequential:        P:      0 (requires_grad:      0) IO:   7014 (requires_grad:   7014)
      4     Linear:          P:   2002 (requires_grad:   2002) IO:   7014 (requires_grad:   7014)
      5     ReLU:            P:      0 (requires_grad:      0) IO:     28 (requires_grad:     28)

    >>> print(trackers[0].total_repr())
    P:   6002 (requires_grad:   6002) IO:  14049 (requires_grad:  14028)
    """
    local_dict = None

    @classmethod
    def flat_tensors(cls, obj):
        if isinstance(obj, (tuple, list)):
            for o in obj:
                yield from cls.flat_tensors(o)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from cls.flat_tensors(v)
        else:
            if isinstance(obj, torch.Tensor):
                yield obj
            elif dataclasses.is_dataclass(obj):
                yield from cls.flat_tensors([
                    getattr(obj, field.name)
                    for field in dataclasses.fields(obj)
                ])

    def maybe_init(self):
        import weakref
        if len(self.shared_dict) == 0:
            self.shared_dict['parameters_learnable'] = 0
            self.shared_dict['parameters_fixed'] = 0
            self.shared_dict['tensors_learnable'] = 0
            self.shared_dict['tensors_fixed'] = 0
            self.shared_dict['visited'] = _IDBasedWeakSet()

        if not self.local_dict:
            self.local_dict = {}
            self.local_dict['parameters_learnable'] = 0
            self.local_dict['parameters_fixed'] = 0
            self.local_dict['tensors_learnable'] = 0
            self.local_dict['tensors_fixed'] = 0
            self.local_dict['visited'] = _IDBasedWeakSet()

    @classmethod
    def get_size(cls, tensor):
        return tensor.numel()

    def maybe_add(self, tensor, learnable_key, fixed_key):

        if tensor in self.shared_dict['visited']:
            pass
        else:
            self.shared_dict['visited'].add(tensor)
            if tensor.requires_grad:
                self.shared_dict[learnable_key] += self.get_size(tensor)
            else:
                self.shared_dict[fixed_key] += self.get_size(tensor)

        if tensor not in self.local_dict['visited']:
            self.local_dict['visited'].add(tensor)
            if tensor.requires_grad:
                self.local_dict[learnable_key] += self.get_size(tensor)
            else:
                self.local_dict[fixed_key] += self.get_size(tensor)

    def pre(self, module, input):
        self.maybe_init()

        for p in module.parameters(recurse=self.leaf):
            self.maybe_add(p, 'parameters_learnable', 'parameters_fixed')

        for t in self.flat_tensors(input):
            self.maybe_add(t, 'tensors_learnable', 'tensors_fixed')

    def post(self, module, input, output):
        for t in self.flat_tensors(output):
            self.maybe_add(t, 'tensors_learnable', 'tensors_fixed')

    def _to_str(self, value):
        return f'{value:6_}'

    @property
    def data(self):
        l = self.local_dict
        pl = l['parameters_learnable']
        pf = l['parameters_fixed']
        tl = l['tensors_learnable']
        tf = l['tensors_fixed']
        return {
            'header': ['#Parameters', '#IO Tensors'],
            'align': '<<',
            'data': [
                f'P: {self._to_str(pl+pf)} (requires_grad: {self._to_str(pl)})',
                f'IO: {self._to_str(tl + tf)} (requires_grad: {self._to_str(tl)})',
            ],
        }

    def total_repr(self):
        l = self.shared_dict
        pl = l['parameters_learnable']
        pf = l['parameters_fixed']
        tl = l['tensors_learnable']
        tf = l['tensors_fixed']
        return f'P: {self._to_str(pl+pf)} (requires_grad: {self._to_str(pl)}) ' \
               f'IO: {self._to_str(tl+tf)} (requires_grad: {self._to_str(tl)})'


class IOPMemTracker(IOPNumTracker):
    """

    >>> from torch.nn import Sequential, ReLU, Linear
    >>> net = Sequential(Linear(3, 1000), ReLU(), Sequential(Linear(1000, 2), ReLU()))
    >>> with track(net, IOPMemTracker) as trackers:
    ...     _ = net(torch.randn(7, 3))
    >>> print(trackers)
                             Mem Parameters                        Mem IO Tensors
      0 Sequential:          P:      0 B (requires_grad:      0 B) IO:    140 B (requires_grad:     56 B)
      1   Linear:            P:  16000 B (requires_grad:  16000 B) IO:  28084 B (requires_grad:  28000 B)
      2   ReLU:              P:      0 B (requires_grad:      0 B) IO:  56000 B (requires_grad:  56000 B)
      3   Sequential:        P:      0 B (requires_grad:      0 B) IO:  28056 B (requires_grad:  28056 B)
      4     Linear:          P:   8008 B (requires_grad:   8008 B) IO:  28056 B (requires_grad:  28056 B)
      5     ReLU:            P:      0 B (requires_grad:      0 B) IO:    112 B (requires_grad:    112 B)

    >>> print(trackers[0].total_repr())
    P:  24008 B (requires_grad:  24008 B) IO:  56196 B (requires_grad:  56112 B)
    """
    @classmethod
    def get_size(cls, tensor):
        return tensor.nelement() * tensor.element_size()

    def _to_str(self, value):
        return f'{value:6_} B'

    @property
    def data(self):
        data = super().data
        data['header'] = ['Mem Parameters', 'Mem IO Tensors']
        return data


class OBackwardMemTracker(Tracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = []
        self.grad_sizes = []

    def post(self, module, input, output) -> None:
        for t in IOPNumTracker.flat_tensors(output):
            if t.requires_grad:
                self.hooks.append(t.register_hook(self._callback))

    def _callback(self, grad):
        if grad is None:
            self.grad_sizes.append('None')
        else:
            size = IOPMemTracker.get_size(grad)
            self.grad_sizes.append(f'{size:_}')
        return grad

    @property
    def data(self):
        if len(self.grad_sizes) == 0:
            grad_sizes = 'Missing'
            # grad_sizes = f'{self.grad_sizes[0]:_}'
        else:
            grad_sizes = ' + '.join(self.grad_sizes)
            grad_sizes += ' B'

        return {
            'header': ['Out Grad Mem'],
            'align': '>',
            'data': [f'{grad_sizes}'],
        }


class _IDBasedWeakSet:
    """
    >>> a = torch.tensor([1, 2])
    >>> b = torch.tensor([1, 2, 3])
    >>> s = _IDBasedWeakSet()
    >>> a in s, b in s
    (False, False)
    >>> s.add(a)
    >>> a in s, b in s
    (True, False)
    >>> s.add(b)
    >>> a in s, b in s
    (True, True)
    >>> s
    _IDBasedWeakSet({tensor([1, 2]), tensor([1, 2, 3])})
    >>> del a
    >>> s
    _IDBasedWeakSet({tensor([1, 2, 3])})
    """
    def __init__(self, items=None):
        self.data = {}
        if items:
            for i in items:
                self.add(i)

    def add(self, item):
        if item is None:
            raise ValueError(item)
        self.data[id(item)] = weakref.ref(item)

    def __contains__(self, item):
        if id(item) in self.data:
            ref = self.data[id(item)]()
            if ref is None:
                return False  # object was deleted
            else:
                return ref is item
        else:
            return False

    def __repr__(self):
        if self.data:
            s = [v() for v in self.data.values()]
            s = [repr(v) for v in s if v is not None]
            s = ', '.join(s)
            return f'{self.__class__.__name__}({{{s}}})'
        else:
            return f'{self.__class__.__name__}()'
