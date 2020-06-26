from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from padertorch.contrib.je.modules.global_pooling import compute_mask


class RNN(nn.Module):
    rnn_cls = None

    def __init__(
            self, input_size, hidden_size, num_layers=1,
            bias=True, dropout=0., bidirectional=False,
    ):
        super().__init__()
        self._rnn = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, seq_len=None):
        if seq_len is not None:
            x = pack_padded_sequence(
                x, seq_len, batch_first=self._rnn.batch_first
            )
        x, _ = self._rnn(x)
        if seq_len is not None:
            x = pad_packed_sequence(x, batch_first=self._rnn.batch_first)[0]
        return x


class GRU(RNN):
    rnn_cls = nn.GRU


class LSTM(RNN):
    rnn_cls = nn.LSTM


def reverse_sequence(x, seq_len=None):
    """
    >>> x, seq_len = (torch.cumsum(torch.ones((3,5,8)), dim=1), [4,5,2])
    >>> reverse_sequence(x, seq_len)
    >>> reverse_sequence(reverse_sequence(x, seq_len), seq_len)
    Args:
        x:
        seq_len:

    Returns:

    """
    if seq_len is None:
        return x.flip(1)
    else:
        T = x.shape[1]
        x = torch.cat((x, x), dim=1)
        x = torch.stack(
            tuple([
                x[i, seq_len[i]:seq_len[i] + T].flip(0) for i in range(len(x))
            ]),
            dim=0
        )
        mask = compute_mask(x, seq_len)
    return x * mask
