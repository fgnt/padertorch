import torch

from padertorch.base import Model
from padertorch import modules


class Classifier(Model):
    """
    >>> from paderbox.utils.nested import deflatten
    >>> classifier = Classifier.from_config(\
        Classifier.get_config(deflatten({\
            'net.kwargs.input_size': 40,\
            'net.kwargs.hidden_size': 2*[128],\
            'net.kwargs.output_size': 10\
        })))
    >>> inputs = (torch.zeros(8,40), torch.zeros(8).long(),)
    >>> outputs = classifier(inputs)
    >>> outputs.shape
    torch.Size([8, 10])
    >>> review = classifier.review(inputs, outputs)
    """
    def __init__(self, net, class_axis=-1, feature_key=0, target_key=1):
        super().__init__()
        self.net = net
        self.class_axis = class_axis
        self.feature_key = feature_key
        self.target_key = target_key

    @classmethod
    def get_signature(cls):
        signature = super().get_signature()
        signature['net'] = {'cls': modules.fully_connected_stack}
        return signature

    def forward(self, inputs):
        return self.net(inputs[self.feature_key])

    def review(self, inputs, outputs):
        targets = inputs[self.target_key]
        logits = outputs
        predictions = torch.argmax(outputs, dim=self.class_axis)
        if logits.dim() == 1:
            assert self.class_axis == 0
            logits = logits[None]
        elif self.class_axis not in [1, -(logits.dim() - 1)]:
            permutation = list(range(logits.dim()))
            permutation.insert(1, permutation.pop(self.class_axis))
            logits = logits.permute(permutation)
        ce = torch.nn.CrossEntropyLoss(reduction='none')(logits, targets)
        accuracy = (targets == predictions).float().mean()
        summary = dict(
            loss=ce.mean(),
            scalars=dict(accuracy=accuracy),
            histograms=dict(ce=ce, predictions=predictions)
        )
        return summary
