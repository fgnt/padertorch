import torch
import padertorch as pt


class AngularPenaltySMLoss(pt.Module):

    def __init__(
            self,
            in_features,
            out_features,
            loss_type='aam',
            eps=1e-7,
            s=None,
            m=None,
            reduce='mean'
    ):
        '''
        Modified Version of the angular softmax implementation found here:
        https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py

        Combination of the last fully connected layer and the classification loss, allowing in a modiciation of the
        logits to realize modified cross entropy loss criteria by penalizing the angle between the feature embeddings.

        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'aam']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        Angular Additional Margin (AAM) : https://arxiv.org/abs/1801.05599

        Args:
            in_features: Size of embedding dimension
            out_features: Number of target classes
            loss_type: Variant of angular softmax used for loss computation ['arcface', 'sphereface', 'aam']
            s: scale
            m: margin
            reduce: reduction type

        Returns:
            Angular Margin loss

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'aam']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'aam':
            self.s = 30.0 if not s else s
            self.m = 0.2 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

        self.reduce = reduce

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: bottleneck features of the embedding extractor, shape (B, E)
            labels: target class labels for classification, shape (B)

        Returns:

        """
        if len(embeddings.size()) == 1:
            embeddings = embeddings[None, :]
        assert len(embeddings) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for _, module in self.fc.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch.nn.functional.normalize(module.weight, p=2, dim=1)

        logits = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        logits = self.fc(logits)

        if self.loss_type == 'aam':
            numerator = self.s * (torch.diagonal(logits.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(logits.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))
        else:
            return NotImplementedError
        excl = torch.cat([torch.cat((logits[i, :y], logits[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)],
                         dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        if self.reduce == 'mean':
            return -torch.mean(L)
        else:
            return -1*L
