import torch
from padertorch.base import Module


class StatefulLSTM(Module):
    _states = None

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bidirectional: bool = False,
            dropout: float = 0.,
            batch_first: bool = True,
            save_states: bool = True
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
        self.save_states = save_states

    @property
    def states(self):
        return self._states

    @states.deleter
    def states(self):
        self._states = None

    @states.setter
    def states(self, states):
        self._states = states

    def forward(self, x):
        h, self.states = self.lstm(x, self.states)
        if not self.save_states:
            del self.states
        return h

