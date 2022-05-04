from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from padertorch.ops.sequence.mask import compute_mask
from einops import rearrange
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d


class RNN(nn.Module):
    rnn_cls = None

    def __init__(
            self, input_size, hidden_size, num_layers=1,
            bias=True, dropout=0., bidirectional=False, reverse=False,
            output_net=None,
    ):
        super().__init__()
        self.rnn = self.rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.reverse = reverse
        self.output_net = output_net

    def forward(self, x, seq_len):
        x = rearrange(x, 'b f t -> b t f')
        if self.reverse:
            x = reverse_sequence(x, seq_len=seq_len)
        if seq_len is not None:
            x = pack_padded_sequence(
                x, seq_len, batch_first=self.rnn.batch_first
            )
        x, _ = self.rnn(x)
        if seq_len is not None:
            x = pad_packed_sequence(x, batch_first=self.rnn.batch_first)[0]
        if self.reverse:
            x = reverse_sequence(x, seq_len=seq_len)
        x = rearrange(x, 'b t f -> b f t')
        if self.output_net is not None:
            x, seq_len = self.output_net(x, seq_len)
        return x, seq_len

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config['output_net'] is not None:
            config['output_net']['factory'] = CNN1d
            if config['output_net']['factory'] in [CNN1d, CNNTranspose1d]:
                config['output_net']['in_channels'] = config['hidden_size'] * (1 + config['bidirectional'])
            else:
                raise ValueError(f'output_net factory {config["output_net"]["factory"]} not allowed.')


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
