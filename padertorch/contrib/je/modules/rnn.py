from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from padertorch.ops.sequence.mask import compute_mask
from einops import rearrange
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.transformer import TransformerLayerStack


class RNN(nn.Module):
    def __init__(self, rnn, output_net=None, reverse=False):
        super().__init__()
        # self.rnn = self.rnn_cls(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bias=bias,
        #     batch_first=True,
        #     dropout=dropout,
        #     bidirectional=bidirectional,
        # )
        self.rnn = rnn
        self.output_net = output_net
        self.reverse = reverse

    def forward(self, x, seq_len):
        if self.rnn is not None:
            if isinstance(self.rnn, nn.RNNBase):
                self.rnn.flatten_parameters()
            x = rearrange(x, 'b f t -> b t f')
            if self.reverse:
                x = reverse_sequence(x, seq_len=seq_len)
            if isinstance(self.rnn, TransformerLayerStack):
                x, _ = self.rnn(x, seq_len)
            else:
                assert self.rnn.batch_first is True, self.rnn.batch_first
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

    def freeze(self, num_layers=None):
        # ToDo: does this method work with all RNN types inkl. Transformer?
        if num_layers == 0:
            return
        num_rnn_layers = self.rnn.num_layers if num_layers is None else min(num_layers, self.rnn.num_layers)
        for name, param in self.rnn.named_parameters():
            name_split = name.split('.')
            if name.endswith(f'_l{num_rnn_layers}') or (
                (len(name_split) > 1) and (name_split[1] == str(num_rnn_layers))
            ):
                break
            print(f'Freeze {name}')
            param.requires_grad = False
        num_out_layers = None if num_layers is None else max(num_layers - self.rnn.num_layers, 0)
        if self.output_net is not None:
            print(f'Freeze {num_out_layers} output_net layers')
            self.output_net.freeze(num_out_layers)

    @classmethod
    def finalize_dogmatic_config(cls, config):
        if config['output_net'] is not None:
            config['output_net']['factory'] = CNN1d
            if config['output_net']['factory'] in [CNN1d, CNNTranspose1d]:
                if config['rnn'] is not None:
                    config['output_net']['in_channels'] = config['rnn']['hidden_size'] * (1 + config['rnn']['bidirectional'])
            else:
                raise ValueError(f'output_net factory {config["output_net"]["factory"]} not allowed.')


class GRU(RNN):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['rnn'] = {
            'factory': nn.GRU,
            'num_layers': 1,
            'bias': True,
            'dropout': 0.,
            'bidirectional': False,
            'batch_first': True,
        }
        assert config['rnn'] is None or config['rnn']['factory'] == nn.GRU, config['rnn']
        super().finalize_dogmatic_config(config)


class LSTM(RNN):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['rnn'] = {
            'factory': nn.LSTM,
            'num_layers': 1,
            'bias': True,
            'dropout': 0.,
            'bidirectional': False,
            'batch_first': True,
        }
        assert config['rnn'] is None or config['rnn']['factory'] == nn.LSTM, config['rnn']
        super().finalize_dogmatic_config(config)


class TransformerEncoder(RNN):
    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['rnn'] = {
            'factory': TransformerLayerStack,
            'num_layers': 6,
            'hidden_size': 512,
            'num_heads': 8,
            'd_ff': 2048,
            'bidirectional': False,
            'cross_attention': False,
            'activation_ff': 'relu',
            'dropout': 0.,
            'positional_encoding': True,
        }
        assert config['rnn'] is None or config['rnn']['factory'] == TransformerLayerStack, config['rnn']
        assert config['rnn']['cross_attention'] is False, config['rnn']['cross_attention']
        super().finalize_dogmatic_config(config)


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
