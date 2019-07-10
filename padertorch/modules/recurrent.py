import torch
from padertorch.base import Module


class StatefulLSTM(Module):
    _states = None

    def __init__(
            self,
            input_size: int = 513,
            hidden_size: int = 512,
            num_layers: int = 1,
            bidirectional: bool = False,
            dropout: float = 0.,
            batch_first: bool = True
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
        return h

