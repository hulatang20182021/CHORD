"""Verify actual head inputs and the five-loss sum, not just route labels."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chord/downstream/scripts"))
sys.path.insert(0, str(ROOT.parent / "LETTER-master/LETTER-TIGER"))
from modeling_matched_curriculum_strict_symmetric_shared_anchor import MatchedCurriculumLETTER


class Capture(nn.Module):
    def forward(self, value):
        self.value = value
        return value


@pytest.mark.parametrize("route", ["same_view", "cross_view"])
def test_head_inputs_loss_and_docs_gradient(route):
    model = MatchedCurriculumLETTER.__new__(MatchedCurriculumLETTER)
    nn.Module.__init__(model)
    model.pcsc_route_control = route
    model.pcsc_alignment = "positional"
    model.pcsc_h12_mode = "sum"
    model._pcsc_factor = 0.7
    model.pcsc_lambdas = dict.fromkeys(("base", "cf", "cfres", "comp", "res"), 1.0)
    names = ("consensus", "cf", "cfres", "sem_full", "res")
    for name in names:
        setattr(model, "pcsc_" + name + "_head", Capture())
    hidden = torch.randn(1, 4, 4, generator=torch.Generator().manual_seed(17), requires_grad=True)
    rows = torch.tensor([0])
    model._last_item_records = [(rows, 0, rows, torch.tensor([True]))]
    targets = [torch.tensor([[1., 2., 3., 4.]]) + k for k in range(5)]
    for attr, value in zip(("_pcsc_consensus", "_pcsc_zcf", "_pcsc_cfres", "_pcsc_zsem", "_pcsc_usem_raw"), targets):
        setattr(model, attr, value)
    loss, _ = model._pcsc_loss(hidden)
    h1, h2, h3 = hidden[:, 0], hidden[:, 1], hidden[:, 2]
    expected = [h1, h1+h3, h3, h1+h2, h2] if route == "same_view" else [h1, h1+h2, h2, h1+h3, h3]
    for name, value in zip(names, expected):
        torch.testing.assert_close(getattr(model, "pcsc_" + name + "_head").value, value)
    expected_loss = 0.7 * sum(model._cosine_loss(x, y) for x, y in zip(expected, targets))
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[:, :3].abs().sum() > 0
    assert hidden.grad[:, 3].abs().sum() == 0
