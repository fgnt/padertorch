import warnings

import numpy as np
import paderbox as pb
import padertorch as pt
from einops import einops
from padertorch.contrib.cb.summary import ReviewSummary
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.contrib.je.modules.conv import CNN2d, Conv2d

from .loss import AngularPenaltySMLoss
from .eer_metrics import get_eer, get_dcf


class ResNet34(pt.Module):
    """
    Version of a ResNet34 adapted for speaker embedding extraction [1]. As base, the CNN wrapper from padertorch.contrib.je
    is used.
    The ResNet is then followed by an average pooling both over time and frequency axes to obtain speaker embeddings
    (d-vectors).
    [1] "ResNeXt and Res2Net Structures for Speaker Verification", Tianyan Zhou, Yong Zhao, Jian Wu,
    https://doi.org/10.48550/arXiv.2007.02480
    """
    def __init__(
            self,
            in_channels=1,
            channels=(64, 128, 256, 256),
            dvec_dim=256,
            activation_fn='relu',
            norm='batch',
            pre_activation=True,
    ):
        super().__init__()
        # ResNet34
        out_channels = 3*2*[channels[0]] + 4*2*[channels[1]] + 6*2*[channels[2]] + 3*2*[channels[3]]
        assert len(out_channels) == 32, len(out_channels)
        kernel_size = 32*[3]
        stride = 3*2*[(1, 1)] + [(2, 2)] + (4*2 - 1)*[(1, 1)] + 6*2*[(1, 1)] + [(2, 1)] + (3*2 - 1)*[(1, 1)]
        pool_size = 32 * [1]
        pool_stride = 32 * [1]
        pool_type = 32 * [None]
        residual_connections = 32 * [None]
        for i in range(0, 32, 2):
            residual_connections[i] = i+2
        norm = norm
        self.embedding_dim = dvec_dim
        self.input_convolution = Conv2d(in_channels, channels[0], kernel_size=3, stride=2, bias=False, norm=norm)
        self.resnet = CNN2d(
            input_layer=False,
            output_layer=False,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_type=pool_type,
            residual_connections=residual_connections,
            activation_fn=activation_fn,
            pre_activation=pre_activation,
            norm=norm,
            skip_connection_norm=True,
            skip_connection_pre_activation=True
        )
        self.output_convolution = Conv2d(channels[-1], dvec_dim, kernel_size=3, stride=(2, 1), bias=False,
                                         activation_fn='relu', norm=norm, pre_activation=True)
        self.bn = nn.BatchNorm1d(self.embedding_dim, affine=False)

    def forward(self, x, seq_len):
        """
        Args:
            x: input data in form of log mel features, Shape (B T F)
            seq_len: sequence lengths (number of frames) before padding of each input example, Shape (B)
        Returns:
            dvectors: E-dimensional d-vectors of the ResNet, Shape (B E)
            embeddings: Frame-level model output without time average poling, Shape (B E T)
        """
        # Add a singleton dimension for the convolutions
        # Shape (b t f) -> (b 1 t f)
        x = einops.rearrange(x, 'b t f -> b 1 f t')

        x, seq_len = self.input_convolution(x, seq_len)
        x, seq_len = self.resnet(x, seq_len)
        x, seq_len = self.output_convolution(x, seq_len)

        # Calculate Mean over reduced frequency dim (same len for each example)
        embeddings = Mean(axis=-2)(x)
        # Calculate Mean over reduced time dim to perform Time Average Pooling (different len for each example)
        dvectors = Mean(axis=-1)(embeddings, seq_len)

        dvectors = dvectors.view(-1, self.embedding_dim)
        dvectors = self.bn(dvectors)

        return dvectors, embeddings


class ResNet18(pt.Module):
    """
    ResNet18 configuration of the embedding extractor
    """
    def __init__(
            self,
            in_channels=1,
            channels=(64, 128, 128, 256),
            dvec_dim=128,
            activation_fn='relu',
            norm='batch',
            pre_activation=True,
    ):
        super().__init__()
        # ResNet18
        out_channels = 2*2*[channels[0]] + 2*2*[channels[1]] + 2*2*[channels[2]] + 2*2*[channels[3]]
        assert len(out_channels) == 16  # 18 - input conv - output conv
        kernel_size = 16*[3]
        stride = 2*2*[(1, 1)] + [(2,2)] + (2*2 - 1)*[(1, 1)] + 2*2*[(1, 1)] + [(2, 1)] + (2*2 -1)*[(1, 1)]
        pool_size = 16 * [1]
        pool_stride = 16 * [1]
        pool_type = 16 * [None]
        residual_connections = 16 * [None]
        for i in range(0, 16, 2):
            residual_connections[i] = i+2

        self.embedding_dim = dvec_dim
        self.input_convolution = Conv2d(in_channels, channels[0], kernel_size=3, stride=2, bias=False, norm=norm)
        self.resnet = CNN2d(
            input_layer=False,
            output_layer=False,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pool_size=pool_size,
            pool_stride=pool_stride,
            pool_type=pool_type,
            residual_connections=residual_connections,
            activation_fn=activation_fn,
            pre_activation=pre_activation,
            norm=norm,
            skip_connection_norm=True,
            skip_connection_pre_activation=True
        )
        self.output_convolution = Conv2d(channels[-1], dvec_dim, kernel_size=3, stride=(2, 1), bias=False,
                                         activation_fn='relu', norm=norm, pre_activation=True)

        self.bn = nn.BatchNorm1d(self.embedding_dim, affine=False)

    def forward(self, x, seq_len):
        """

        Args:
            x: input data in form of log mel features, Shape (B T F)
            seq_len: sequence lengths (number of frames) before padding of each input example, Shape (B)
        Returns:
            dvectors: E-dimensional d-vectors of the ResNet, Shape (B E)
            embeddings: Frame-level model output without time average poling, Shape (B E T)
        """
        # Add a singleton dimension as channel dimension of the ResNet
        # Shape (b t f) -> (b 1 t f)
        x = einops.rearrange(x, 'b t f -> b 1 f t')

        x, seq_len = self.input_convolution(x, seq_len)
        x, seq_len = self.resnet(x, seq_len)
        x, seq_len = self.output_convolution(x, seq_len)

        # Calculate Mean over reduced frequency dim (same len for each example)
        embeddings = Mean(axis=-1)(x)
        # Calculate Mean over reduced frequency dim (different len for each example)
        x = Mean(axis=-1)(embeddings, seq_len)
        x = x.view(-1, self.embedding_dim)
        x = self.bn(x)

        return x, embeddings


class DVectorModel(pt.Model):
    """
        Fundamental model for the speaker embedding extractor. Consists of a d-Vector extractor (speaker_net), and the
        used loss function (loss). By default, a ResNet34 is used for embedding extraction and an AAM-Softmax as
        classification loss.
    """

    def __init__(
            self,
            speaker_net: ResNet34,
            loss: AngularPenaltySMLoss,
            sampling_rate=16000,
    ):
        super().__init__()
        self.speaker_net = speaker_net
        self.sampling_rate = sampling_rate
        self.loss = loss

    @classmethod
    def finalize_dogmatic_config(cls, config):
        """
            Provides a default configuration. See `padertorch.configurable` for details.
        """
        config['speaker_net'] = {'factory': ResNet34}
        config['loss'] = {
            'factory': AngularPenaltySMLoss,
            'in_features': config['speaker_net']['dvec_dim'],
            # 'out_features': # has to be set to number classes in the dataset
        }

    def prepare_example(self, example):
        """
            Data preparation function for an example as expected by the d-Vector model
        """
        observation = example['audio_data']['observation']

        # Mean and variance normalization of observation
        observation = (observation - np.mean(observation)) / (np.std(observation) + 1e-7)
        # Extract 80-dimensional log-fbank features
        fbank_features = pb.transform.logfbank(
            observation, sample_rate=self.sampling_rate,
            number_of_filters=80,
        )

        return {
            'observation': observation.astype(np.float32),
            'features': fbank_features.astype(np.float32),
            'num_frames': fbank_features.shape[0],
            'speaker_id': example['speaker_id'],
            'example_id': example['example_id'],
        }

    def forward(self, example):
        sequence_lengths = example['num_frames']
        sequence = pt.pad_sequence(example['features'], batch_first=True)
        return self.speaker_net(sequence, sequence_lengths)

    def review(self, example, outputs):
        summary = ReviewSummary(sampling_rate=self.sampling_rate)
        target_labels = torch.tensor(example['speaker_id'])
        dvectors, embeddings = outputs
        # Compute AAM-Softmax loss
        loss = self.loss(dvectors, labels=target_labels)
        summary.add_to_loss(loss)
        summary.add_histogram('loss_', loss)
        summary.add_audio('observation', example['observation'][0])
        summary.add_spectrogram_image('features', torch.exp(example['features'][0]))

        if not self.training:
            # Normalize every embedding to length 1
            dvectors = dvectors / (torch.norm(dvectors, dim=1, keepdim=True) + 1e-8)
            # Save extracted d-Vectors of validation set to estimate an EER during modify_summary
            summary.add_buffer('example_ids', example['example_id'])
            summary.add_buffer('speaker_ids', example['speaker_id'])
            summary.add_buffer('dvectors', dvectors)
        return summary

    def modify_summary(self, summary):
        """
        Performs am approximate EER calculation on the validation set if possible. Takes the d-vectors for all
        embeddings in the validation set, shuffles them and then calculates the EER for these trial pairs
        Note: Only works for small numbers of speakers (e.g. 10) and small validation set sizes. For
        large validiation sets, the runtime of this setp will dominate the execution time!
        """
        if 'dvectors' in summary['buffers']:
            dvectors = sum(summary['buffers']['dvectors'], [])

            dvectors = np.concatenate(dvectors, axis=0)
            # Subtract global mean to obtain more meaningful cosine distance
            validation_mean = np.mean(dvectors, axis=0, keepdims=True)
            dvectors = dvectors - validation_mean
            examples = sum(summary['buffers']['example_ids'], [])
            speakers = sum(summary['buffers']['speaker_ids'], [])
            print('Obtaining approximate metrics for the validation step')
            if len(examples) > 5000:
                warnings.warn(f'Got {len(examples)} examples for inferring EER for validation. Large validation sets '
                              f'dramatically increase the computation time!')
            if len(set(speakers)) > 20:
                warnings.warn(f'Found {len(set(speakers))} different speakers in validation set. The accuracy of the '
                              f'estimated EER drops with increasingly large numbers of speaker due to the '
                              f'trial pairs!')
            indexer = list(range(len(examples)))
            np.random.default_rng(42).shuffle(indexer)
            scores = list()
            labels = list()
            for idx1, idx2 in enumerate(indexer):
                labels.append(speakers[idx1] == speakers[idx2])
                scores.append(1 - cosine(dvectors[idx1], dvectors[idx2]))

            eer = get_eer(scores, labels)
            dcf = get_dcf(scores, labels)
            print(f'(Pseudo) Equal error rate for validation is: {eer}')
            summary['scalars']['EER'] = eer
            summary['scalars']['minDCF'] = dcf
            summary['histograms']['scores'] = np.array(scores)
            summary['histograms']['score_distance'] = np.abs(np.array(labels) - np.array(scores))
            summary['buffers'].clear()
        return super().modify_summary(summary)

