import typing as tp

from einops import rearrange
import numpy as np
import padertorch as pt
from padertorch.contrib.je.modules.conv import CNN1d, CNNTranspose1d
from padertorch.contrib.je.modules.reduce import Mean
from padertorch.modules.normalization import normalize
from paderbox.utils.mapping import Dispatcher
import torch
from torchvision.utils import make_grid


def repeat_interleaved(x, n):
    """
    >>> repeat_interleaved(torch.Tensor([1,2,3]), 2)
    >>> repeat_interleaved(torch.Tensor([[1,2,3],[4,5,6]]), 2)
    Args:
        x:
        n:

    Returns:

    """
    n_repeats = (1, n) + (x.dim()-1)*(1,)
    return rearrange(
        x.unsqueeze(1).repeat(n_repeats), 'b n ... -> (b n) ...'
    )


def interleave(x, y, dim):
    """

    Args:
        x:
        y:
        dim:

    Returns:

    >>> interleave(torch.Tensor([[1,2,3]]),torch.Tensor([[4,5,6]]), dim=1)
    """
    assert len(x.shape) > dim >= 0, dim
    shape = [*x.shape]
    shape[dim] *= 2
    return torch.stack((x, y), dim=dim+1).view(shape)


def cosine_similarity(tensor, other, eps=1e-12):
    """

    Args:
        tensor: Shape (..., b, d)
        other: Shape (..., d, b)
        eps:

    Returns:

    """
    prod = torch.matmul(tensor, other)
    norm = (
        torch.linalg.norm(tensor, ord=None, dim=-1, keepdim=True)
        * torch.linalg.norm(other, ord=None, dim=-2, keepdim=True)
    )
    return prod / torch.maximum(norm, torch.tensor(eps).to(norm.device))


SIMILARITIES = Dispatcher(
    dot_product=torch.matmul,
    cosine=cosine_similarity,
)


class CPC1d(pt.Model):
    """Contrastive Predictive Coding on 1d data [1].

    [1]: Oord, Aäron van den et al. “Representation Learning with Contrastive
        Predictive Coding.” 2018.

    Args:
        encoder: Encoder network g_enc.
        fwd_projection_head: Auto-regressive network g_ar. If None, compute
            similarities on the encoder output.
        bwd_projection_head: Auto-regressive network that runs backward in time
            for backward predictions. If None, compute
            similarities on the encoder output.
        feature_extractor: Feature extractor on the input data before passing
            to the encoder.
        fwd_prediction_steps: List of prediction steps into the future. Must be
            positive.
        bwd_prediction_steps: List of prediction steps into the past. Must be
            negative.
        multi_view (bool): Whether to use multi-view training.
        second_view_extractor: If not None, use this feature extractor for the
            second view.
        temperature (float): Temperature for the softmax.
        l2_normalization (bool): Whether to apply l2 normalization to the
            features before computing the similarity.
        negatives_mode (str): How to sample negatives. One of 'batch', 'time',
            'all'.
        negative_to_target_dist (int): Distance of negatives to target in time
            steps.
        input_key (str): Key of the input in the batch.
        input_seq_len_key (str): Key of the sequence length in the batch.
        prefix (str): Prefix for the keys in the summary.
        similarity (str): Similarity function to use.
        batch_mask_key: If not None and negatives_mode is 'batch', set
            similarities between examples with the same key to -inf.
        time_mask_key: If not None, indicates that a time mask was passed
            to inputs. Then, only compute the loss where the time mask is 1.
        invert_batch_mask (bool): If True, draw negatives with the same key.
        negative_ratio (float): Percentage of negative samples to use.
            Uses the quantile of negatives with lowest similarity to the target.

    >>> config = CPC1d.get_config(dict(\
            encoder=dict(\
                factory=CNN1d,\
                in_channels=80,\
                out_channels=3*[32],\
                kernel_size=3,\
            ),\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
            fwd_prediction_steps=(10,20),\
            bwd_prediction_steps=(-20,-10),\
            input_key='stft',\
            input_seq_len_key='seq_len',\
            negatives_mode='time',\
            negative_to_target_dist=10,\
        ))
    >>> cpc = CPC1d.from_config(config)
    >>> inputs = {'stft': torch.rand((4, 1, 100, 257, 2)), 'seq_len': 4*[100]}
    >>> outputs = cpc(inputs)
    >>> config = CPC1d.get_config(dict(\
            encoder=dict(\
                factory=CNN1d,\
                in_channels=80,\
                out_channels=3*[32],\
                kernel_size=3,\
            ),\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
            fwd_prediction_steps=(10,20),\
            bwd_prediction_steps=(-20,-10),\
            multi_view=True,\
            negatives_mode='batch',\
            negative_to_target_dist=10,\
            input_key='stft',\
            input_seq_len_key='seq_len',\
        ))
    >>> cpc = CPC1d.from_config(config)
    >>> inputs = {'stft': torch.rand((4, 1, 100, 257, 2)), 'seq_len': 4*[100]}
    >>> outputs = cpc(inputs)
    >>> config = CPC1d.get_config(dict(\
            encoder=dict(\
                factory=HybridCNN,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
            ),\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
            fwd_prediction_steps=(10,20),\
            bwd_prediction_steps=(-20,-10),\
            multi_view=True,\
            negatives_mode='time',\
            input_key='stft',\
            input_seq_len_key='seq_len',\
        ))
    >>> cpc = CPC1d.from_config(config)
    >>> inputs = {'stft': torch.rand((4, 1, 100, 257, 2)), 'seq_len': 4*[100]}
    >>> outputs = cpc(inputs)
    >>> config = CPC1d.get_config(dict(\
            encoder=dict(\
                factory=HybridCNN,\
                cnn_2d=dict(\
                    in_channels=1, out_channels=3*[32], kernel_size=3, \
                ), \
                cnn_1d=dict(\
                    out_channels=3*[32], kernel_size=3\
                ),\
            ),\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
            fwd_prediction_steps=(10,20),\
            bwd_prediction_steps=(-20,-10),\
            multi_view=True,\
            negatives_mode='all',\
            input_key='stft',\
            input_seq_len_key='seq_len',\
        ))
    >>> cpc = CPC1d.from_config(config)
    >>> inputs = {'stft': torch.rand((4, 1, 100, 257, 2)), 'seq_len': 4*[100]}
    >>> outputs = cpc(inputs)
    >>> config = CPC1d.get_config(dict(\
            encoder=None,\
            fwd_projection_head=None,\
            bwd_projection_head=None,\
            feature_extractor=dict(\
                sample_rate=16000,\
                fft_length=512,\
                n_mels=80,\
            ),\
            fwd_prediction_steps=(10,20),\
            bwd_prediction_steps=(-20,-10),\
            multi_view=True,\
            negatives_mode='all',\
            negative_to_target_dist=10,\
            input_key='stft',\
            input_seq_len_key='seq_len',\
        ))
    >>> cpc = CPC1d.from_config(config)
    >>> inputs = {'stft': torch.rand((4, 1, 100, 257, 2)), 'seq_len': 4*[100]}
    >>> outputs = cpc(inputs)
    """
    def __init__(
        self, encoder, fwd_projection_head, bwd_projection_head, feature_extractor, *,
        fwd_prediction_steps, bwd_prediction_steps,
        multi_view=False, second_view_extractor=None,
        temperature=1., l2_normalization=False,
        negatives_mode='batch', negative_to_target_dist=0,
        input_key='x', input_seq_len_key='seq_len_x', prefix=None,
        similarity='dot_product',
        batch_mask_key: tp.Optional[str] = None,
        time_mask_key: tp.Optional[str] = None,
        invert_batch_mask: bool = False,
        negative_ratio: float = 1.,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        assert len(fwd_prediction_steps) > 0
        assert all([s >= 0 for s in fwd_prediction_steps])
        if bwd_prediction_steps:
            assert all([s <= 0 for s in bwd_prediction_steps])
            assert not ((fwd_projection_head is None) ^ (bwd_projection_head is None))

        self.fwd_projection_head = fwd_projection_head
        self.bwd_projection_head = bwd_projection_head
        self.fwd_prediction_steps = sorted(fwd_prediction_steps)
        self.bwd_prediction_steps = sorted(bwd_prediction_steps)
        self.multi_view = multi_view
        self.second_view_extractor = second_view_extractor
        self.temperature = temperature
        self.l2_normalization = l2_normalization
        self.negatives_mode = negatives_mode
        self.negative_to_target_dist = negative_to_target_dist
        self.input_key = input_key
        self.input_seq_len_key = input_seq_len_key
        self.prefix = "" if prefix is None else prefix + "_"
        self.similarity = SIMILARITIES[similarity]
        self.batch_mask_key = batch_mask_key
        self.time_mask_key = time_mask_key
        self.invert_batch_mask = invert_batch_mask
        self.negative_ratio = negative_ratio

    @property
    def prediction_steps(self):
        return self.bwd_prediction_steps + self.fwd_prediction_steps

    def encode(self, x, seq_len_x, target_shape=None):
        if self.multi_view:
            if self.second_view_extractor is not None:
                x_targets, seq_len_targets = self.second_view_extractor(x, seq_len=seq_len_x)
            elif self.feature_extractor is not None:
                x_targets, seq_len_targets = self.feature_extractor(x, seq_len=seq_len_x)
            else:
                raise ValueError(
                    'multi_view training requires two different views, i.e. '
                    'two separate feature_extractors or one feature_extractor '
                    'with data augmentation.'
                )
        else:
            x_targets = seq_len_targets = None
        if self.feature_extractor is not None:
            x, seq_len_x = self.feature_extractor(x, seq_len=seq_len_x)
        if x_targets is not None:
            x = torch.cat((x, x_targets), dim=0)
            assert (np.array(seq_len_x) == seq_len_targets).all()
            seq_len_x = np.concatenate((seq_len_x, seq_len_targets))
        if isinstance(self.encoder, (CNN1d, CNNTranspose1d)) and x.dim() != 3:
            assert x.dim() == 4, x.dim()
            x = rearrange(x, 'b d f t -> b (d f) t')
        if isinstance(self.encoder, CNNTranspose1d):
            if target_shape is not None:
                target_shape = list(target_shape)
                target_shape[1] = self.encoder.out_channels[-1]
            kwargs = dict(target_shape=target_shape)
        else:
            kwargs = {}
        if self.encoder is None:
            z, seq_len_z = x, seq_len_x
        else:
            z, seq_len_z = self.encoder(x, seq_len_x, **kwargs)
        if z.dim() == 4:
            z = rearrange(z, 'b d f t -> b (d f) t')
        return x, z, seq_len_z

    def predict(self, z, seq_len_z):
        if self.fwd_projection_head is None:
            n = len(self.fwd_prediction_steps)
            z_hat = z.repeat(1, n, 1)
        else:
            z_hat, _ = self.fwd_projection_head(z, seq_len_z)
        if self.bwd_prediction_steps:
            if self.bwd_projection_head is None:
                n = len(self.bwd_prediction_steps)
                z_hat_bwd = z.repeat(1, n, 1)
            else:
                z_hat_bwd, _ = self.bwd_projection_head(z, seq_len_z)
            z_hat = torch.cat((z_hat_bwd, z_hat), dim=1)
        return z_hat

    def contrast(self, z_hat, z, batch_labels=None):
        s = len(self.prediction_steps)
        b, d, t = z.shape
        if self.l2_normalization:
            z, *_ = normalize(z, None, None, 1, 0, 2, None, True, True, 1e-3)
            z_hat = rearrange(z_hat, 'b (s d) t -> b s d t', s=s, d=d)
            z_hat, *_ = normalize(
                z_hat, None, None, 2, 0, 3, None, True, True, 1e-3
            )
            z_hat = rearrange(z_hat, 'b s d t -> b (s d) t')

        assert z_hat.shape[1] == len(self.prediction_steps)*d, (
            z_hat.shape[1], len(self.prediction_steps)*d
        )
        logits = []
        if self.negatives_mode == "batch":
            z = rearrange(z, 'b d t -> t d b')  # .contiguous()
            z_hat = torch.split(rearrange(z_hat, 'b d t -> t b d'), d, dim=2)
            assert len(z_hat) == len(self.prediction_steps), (
                len(z_hat), len(self.prediction_steps)
            )
            if batch_labels is not None:
                # Mask negatives with same label
                if isinstance(batch_labels, (tuple, list)):
                    batch_labels = np.array(batch_labels)
                batch_mask = batch_labels[:, None] != batch_labels[None, :]
                if isinstance(batch_mask, np.ndarray):
                    batch_mask = (
                        torch.from_numpy(batch_mask).to(z.device)
                    )
                if self.invert_batch_mask:
                    # Draw negatives with same label
                    batch_mask = ~batch_mask
                else:
                    # Do not mask same example
                    eye = torch.eye(b).to(z.device).float()
                    batch_mask = batch_mask.float() + eye
                log_batch_mask = torch.log(batch_mask)[None, :, :]
            else:
                log_batch_mask = torch.zeros(1, b, b).to(z.device).float()
            for step, z_hat_s in zip(self.prediction_steps, z_hat):
                z_s = z
                if step < 0:
                    z_hat_s = z_hat_s[abs(step):]
                    z_s = z_s[:-abs(step)]
                elif step > 0:
                    z_hat_s = z_hat_s[:-step]
                    z_s = z_s[step:]
                similarities = (
                    self.similarity(z_hat_s, z_s) / self.temperature
                    + log_batch_mask
                )
                k = max(int(np.ceil(b * (1-self.negative_ratio))), 1)
                th = torch.topk(similarities, k, dim=2).values.min(2)\
                    .values[..., None]
                negatives_mask = (
                    (similarities <= th)
                    # Do no mask target
                    | torch.eye(b)[None].bool().to(similarities.device)
                ).float().log()
                logits.append(rearrange(
                    similarities + negatives_mask, 't b k -> b k t'
                ))
            targets = torch.arange(b)[:, None]
        elif self.negatives_mode == "time":
            z_hat = torch.split(rearrange(z_hat, 'b d t -> b t d'), d, dim=2)
            assert len(z_hat) == len(self.prediction_steps), (
                len(z_hat), len(self.prediction_steps)
            )
            if self.negative_to_target_dist > 1:
                ones = torch.ones((t, t))
                mask = (
                    1
                    - torch.triu(
                        ones, diagonal=-(self.negative_to_target_dist-1)
                    )
                    * torch.tril(ones, diagonal=self.negative_to_target_dist-1)
                    + torch.eye(t)
                )
                logmask = torch.log(mask)
            else:
                logmask = None
            for step, z_hat_s in zip(self.prediction_steps, z_hat):
                z_s = z
                if step < 0:
                    z_hat_s = z_hat_s[:, abs(step):]
                elif step > 0:
                    z_hat_s = z_hat_s[:, :-step]
                    z_s = torch.roll(z_s, -step, dims=-1)
                logits.append(rearrange(
                    self.similarity(z_hat_s, z_s) / self.temperature,
                    'b t k -> b k t'
                ))
                if logmask is not None:
                    logits[-1] = (
                        logits[-1]
                        + logmask[..., :logits[-1].shape[-1]].to(z.device)
                    )
            targets = torch.arange(t)[None]
        elif self.negatives_mode == "all":
            z = rearrange(z, 'b d t -> d b t')
            z_hat = torch.split(rearrange(z_hat, 'b d t -> b t d'), d, dim=2)
            assert len(z_hat) == len(self.prediction_steps), (
                len(z_hat), len(self.prediction_steps)
            )
            if self.negative_to_target_dist > 1:
                ones = torch.ones((t, t))
                mask = (
                    1
                    - torch.triu(
                        ones, diagonal=-(self.negative_to_target_dist-1)
                    )
                    * torch.tril(ones, diagonal=self.negative_to_target_dist-1)
                    + torch.eye(t)
                )
                logmask = torch.log(mask)
                logmask = torch.cat((logmask, torch.zeros((b-1)*t, t)), dim=0)
                logmask = torch.stack(
                    [torch.roll(logmask, i*t, dims=0) for i in range(b)], dim=0
                )
            else:
                logmask = None
            for step, z_hat_s in zip(self.prediction_steps, z_hat):
                z_s = z
                if step < 0:
                    z_hat_s = z_hat_s[:, abs(step):]
                elif step > 0:
                    z_hat_s = z_hat_s[:, :-step]
                    z_s = torch.roll(z_s, -step, dims=2)
                logits.append(rearrange(
                    self.similarity(z_hat_s, rearrange(z_s, 'd b t -> d (b t)'))
                    / self.temperature, 'b t k -> b k t'
                ))
                if logmask is not None:
                    logits[-1] = (
                        logits[-1]
                        + logmask[..., :logits[-1].shape[-1]].to(z.device)
                    )
            targets = torch.arange(b)[:, None] * t + torch.arange(t)
        else:
            raise ValueError(f'Unknown negatives_mode {self.negatives_mode}')

        return logits, targets.to(logits[0].device)

    def forward(self, inputs):
        x = inputs[self.input_key]
        seq_len = inputs[self.input_seq_len_key]
        x, z, seq_len_z = self.encode(
            x, seq_len_x=seq_len,
            target_shape=inputs.get(f'{self.prefix}target_shape', None),
        )
        if self.multi_view:
            z_, z = torch.split(z, z.shape[0]//2)
            seq_len_z = seq_len_z[:z_.shape[0]]
            z_hat = self.predict(z_, seq_len_z)
        else:
            z_hat = self.predict(z, seq_len_z)
        if self.batch_mask_key is not None:
            batch_labels = inputs[self.batch_mask_key]
        else:
            batch_labels = None
        logits, targets = self.contrast(z_hat, z, batch_labels)

        return logits, targets, seq_len_z, x

    def review(self, inputs, outputs):
        logits, targets, seq_len_z, x = outputs

        if self.time_mask_key is not None:
            time_labels = inputs[self.time_mask_key]
        else:
            time_labels = None

        if self.multi_view:
            x, x_targets = torch.split(x, x.shape[0]//2)
        else:
            x_targets = x

        accuracies = {}
        negatives = {}
        ce = 0.
        for step, logits_s in zip(self.prediction_steps, logits):
            seq_len_s = np.array(seq_len_z) - abs(step)
            targets_s = targets[:, :logits_s.shape[2]].expand((
                logits_s.shape[0], logits_s.shape[2]
            ))
            ce_s = torch.nn.CrossEntropyLoss(reduction='none')(
                logits_s, targets_s
            )
            hits = (torch.argmax(logits_s.detach(), dim=1) == targets_s).float()
            if time_labels is not None:
                time_mask = (
                    time_labels[:, :-abs(step)] & time_labels[:, abs(step):]
                ).float().to(ce_s.device)
                ce_s = ce_s * time_mask
                hits = hits * time_mask
                accuracy_correction = torch.from_numpy(
                    seq_len_s / (time_mask.sum(1).cpu().numpy() + 1)
                ).to(ce_s.device)
                active_batch_size = np.maximum(
                    torch.sum(time_mask.sum(1) > 0).cpu().numpy(), 1
                )
            else:
                accuracy_correction = 1.
                active_batch_size = logits_s.shape[0]
            ce = ce + Mean(axis=1)(ce_s, seq_len_s).mean()
            accuracies[f'step_{step}_accuracy'] = (
                Mean(axis=1)(hits, seq_len_s) * accuracy_correction
            ).cpu().numpy().sum() / active_batch_size
            negatives[f'step_{step}_negatives'] = (
                np.sum(logits_s.detach().cpu().numpy() > -np.inf, axis=1) - 1
            ).mean()
        ce = ce / len(self.prediction_steps)

        review = dict(
            losses=dict(
                ce=ce
            ),
            scalars=dict(
                overall_accuracy=np.mean(list(accuracies.values())),
                **accuracies,
                **negatives,
            ),
            histograms=dict(),
            images=dict(input_features=x[:3], target_features=x_targets[:3]),
        )
        return {
            key1: {f'{self.prefix}{key2}': value for key2, value in d.items()}
            for key1, d in review.items()
        }

    def modify_summary(self, summary):
        for key in [f'{self.prefix}input_features', f'{self.prefix}target_features']:
            if key not in summary['images']:
                continue
            image = summary['images'][key]
            if image.dim() == 3:
                image = image.unsqueeze(1)
            summary['images'][key] = make_grid(
                image.flip(2),  normalize=True, scale_each=False, nrow=1
            )
        return summary

    def get_posteriors(self, outputs):
        logits, targets, seq_len_z, *_ = outputs
        posteriors = []
        with torch.no_grad():
            for step, logits_s in zip(self.prediction_steps, logits):
                seq_len_s = np.array(seq_len_z) - abs(step)
                targets_s = targets[:, :logits_s.shape[2]].expand((
                    logits_s.shape[0], logits_s.shape[2]
                ))
                softmax = torch.softmax(logits_s, dim=1)
                posterior = softmax.gather(
                    1, targets_s.long().unsqueeze(1)
                ).squeeze(1)
                posteriors.append(Mean(axis=1)(posterior, seq_len_s))
        posteriors = torch.stack(posteriors, dim=1)
        posteriors = posteriors * np.sqrt(posteriors.shape[0])
        return posteriors.mean(1)


class CPCFeatureExtractor(pt.Module):
    def __init__(self, cpc):
        super().__init__()
        self.cpc = cpc

    def forward(self, x, seq_len=None):
        with torch.no_grad():
            x, seq_len = self.cpc.feature_extractor(x, seq_len)
            if (
                isinstance(self.cpc.encoder, (CNN1d, CNNTranspose1d))
                and x.dim() != 3
            ):
                assert x.dim() == 4, x.dim()
                x = rearrange(x, 'b d f t -> b (d f) t')
            z, seq_len_z = self.cpc.encoder(x, seq_len)
            if z.dim() == 4:
                z = rearrange(z, 'b d f t -> b (d f) t')
            return z, seq_len_z
