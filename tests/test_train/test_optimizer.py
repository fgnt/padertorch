import padertorch as pt
import torch


def test_frad_norm():
    lin = torch.nn.Linear(16, 8)
    opti = pt.optimizer.Adam()
    opti.set_parameters(lin.parameters())
    opti.zero_grad()
    l = lin.weight.sum()
    l.backward()
    grad_norm = opti.clip_grad()
    grad_norm_ref = torch.nn.utils.clip_grad_norm_(
        lin.parameters(), 10.
    )
    assert grad_norm == grad_norm_ref and grad_norm_ref > 0., \
        (grad_norm, grad_norm_ref)
