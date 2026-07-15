from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "chord" / "downstream" / "scripts"
LETTER_DIR = Path(
    os.environ.get(
        "LETTER_TIGER_DIR",
        Path(__file__).resolve().parents[2] / "LETTER-master" / "LETTER-TIGER",
    )
)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(LETTER_DIR))

from modeling_matched_curriculum_letter import MatchedCurriculumLETTER  # noqa: E402


class CaptureHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.last_input = None

    def forward(self, values):
        self.last_input = values.detach().clone()
        return self.projection(values)


def test_semantic_first_positional_legacy5_hidden_contract():
    model = MatchedCurriculumLETTER.__new__(MatchedCurriculumLETTER)
    nn.Module.__init__(model)
    model.pcsc_alignment = "positional"
    model.pcsc_h12_mode = "sum"
    model._pcsc_factor = 1.0
    model.pcsc_lambdas = {name: 1.0 for name in ("cf", "cfres", "base", "res", "comp")}

    hidden_dim = 4
    model.pcsc_cf_head = CaptureHead(hidden_dim, 128)
    model.pcsc_cfres_head = CaptureHead(hidden_dim, 128)
    model.pcsc_base_head = CaptureHead(hidden_dim, 768)
    model.pcsc_res_head = CaptureHead(hidden_dim, 768)
    model.pcsc_sem_head = CaptureHead(hidden_dim, 768)

    h1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    h2 = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    h3 = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    hidden = torch.stack([h1, h2, h3, torch.zeros_like(h1)], dim=1)
    rows = torch.tensor([0], dtype=torch.long)
    model._last_item_records = [
        (torch.tensor([0], dtype=torch.long), 0, rows, torch.tensor([True]))
    ]
    model._pcsc_zcf = torch.randn(1, 128)
    model._pcsc_cfres = torch.randn(1, 128)
    model._pcsc_zsem = torch.randn(1, 768)
    model._pcsc_zsem_base = torch.randn(1, 768)
    model._pcsc_usem_raw = torch.randn(1, 768)

    model._pcsc_loss(hidden)

    torch.testing.assert_close(model.pcsc_base_head.last_input, h1)
    torch.testing.assert_close(model.pcsc_sem_head.last_input, h1 + h2)
    torch.testing.assert_close(model.pcsc_res_head.last_input, h2)
    torch.testing.assert_close(model.pcsc_cf_head.last_input, h1 + h3)
    torch.testing.assert_close(model.pcsc_cfres_head.last_input, h3)
