import torch
from padertorch.base import Module
from torch.nn.utils.rnn import PackedSequence


class LSTM(Module):
    def __init__(
            self,
            input_size: int = 513,
            hidden_size: int = 512,
            num_layers: int = 1,
            bidirectional: bool = False,
            dropout: bool = 0,
            batch_first: bool = True,
            output_states: bool = False
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional,
                                  dropout=dropout,
                                  batch_first=batch_first)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.output_states = output_states
        self.states = None

    def reset_states(self, x):
        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            max_batch_size = x.size(0) if self.batch_first \
                else x.size(1)

        num_directions = 2 if self.bidirectional else 1
        hx = x.new_zeros(self.num_layers * num_directions,
                         max_batch_size, self.hidden_size,
                         requires_grad=False)
        self.states = (hx, hx)

    def set_states(self, states):
        self.states = states

    def forward(self, x):
        if self.states is None:
            self.reset_states(x)
        h, states = self.lstm(x, self.states)
        if self.output_states:
            return h, states
        else:
            return h

