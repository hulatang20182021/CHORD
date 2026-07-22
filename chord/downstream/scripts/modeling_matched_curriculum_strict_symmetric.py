from __future__ import annotations

import torch
import torch.nn as nn

from modeling_matched_curriculum_crossview import (
    MatchedCurriculumLETTER as LegacyCrossViewLETTER,
)


class MatchedCurriculumLETTER(LegacyCrossViewLETTER):
    """Four-objective symmetric CV-PCSC without semantic-base composition."""

    PCSC_CONTRACT_VERSION = "crossview-strict-symmetric-prefix-v1"
    POSITIONAL_PCSC_CONTRACT = (
        "h1_plus_h2_to_cf_full;h2_to_cf_residual;"
        "h1_plus_h3_to_semantic_full;h3_to_semantic_residual;"
        "no_semantic_base_or_additive_semantic_full_loss;fixed_total_auxiliary_budget"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.pcsc_base_head[0].in_features
        self.pcsc_sem_full_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        )
        for parameter in self.pcsc_base_head.parameters():
            parameter.requires_grad_(False)

    def _pcsc_loss(self, hidden):
        records = self._last_item_records
        if not records:
            zero = hidden.sum() * 0
            return zero, self._empty_metrics()

        batch_rows = torch.cat([record[0] for record in records])
        rows = torch.cat([record[2] for record in records])
        hard_mask = torch.cat([record[3] for record in records])
        lengths = torch.tensor(
            [record[0].numel() for record in records],
            device=hidden.device,
            dtype=torch.long,
        )
        starts = torch.repeat_interleave(
            torch.tensor(
                [record[1] for record in records],
                device=hidden.device,
                dtype=torch.long,
            ),
            lengths,
        )
        soft_skipped = int((~hard_mask).sum().item())
        batch_rows, rows, starts = (
            batch_rows[hard_mask],
            rows[hard_mask],
            starts[hard_mask],
        )
        if rows.numel() == 0:
            zero = hidden.sum() * 0
            return zero, self._empty_metrics(soft_skipped)

        if self.pcsc_alignment == "positional":
            offsets = torch.arange(3, device=hidden.device, dtype=torch.long)
        else:
            offsets = torch.tensor(
                [
                    self._component_position["shared"],
                    self._component_position["cfres"],
                    self._component_position["semres"],
                ],
                device=hidden.device,
                dtype=torch.long,
            )
        h1, h2, h3 = hidden[
            batch_rows[:, None], starts[:, None] + offsets[None, :]
        ].unbind(dim=1)
        if self.pcsc_h12_mode == "mean":
            h12 = (h1 + h2) / 2
        elif self.pcsc_h12_mode == "sum":
            h12 = h1 + h2
        else:
            h12 = h2
        h13 = h1 + h3

        zcf = self._pcsc_zcf[rows]
        cfres = self._pcsc_cfres[rows]
        zsem = self._pcsc_zsem[rows]
        semres = self._pcsc_usem_raw[rows]
        valid_res = semres.norm(dim=1) >= 1e-5
        losses = {
            "cf": self._cosine_loss(self.pcsc_cf_head(h12), zcf),
            "cfres": self._cosine_loss(self.pcsc_cfres_head(h2), cfres),
            "comp": self._cosine_loss(self.pcsc_sem_full_head(h13), zsem),
            "res": self._cosine_loss(
                self.pcsc_res_head(h3)[valid_res], semres[valid_res]
            ) if valid_res.any() else hidden.sum() * 0,
        }
        legacy_budget = float(sum(self.pcsc_lambdas.values()))
        active_budget = float(
            self.pcsc_lambdas["cf"]
            + self.pcsc_lambdas["cfres"]
            + self.pcsc_lambdas["comp"]
            + self.pcsc_lambdas["res"]
        )
        budget_scale = legacy_budget / active_budget
        total = budget_scale * sum(
            self._pcsc_factor * self.pcsc_lambdas[name] * loss
            for name, loss in losses.items()
        )
        return total, {
            "pcsc_item_count": int(rows.numel()),
            "pcsc_hard_item_ratio": rows.numel() / max(rows.numel() + soft_skipped, 1),
            "pcsc_soft_item_skipped": soft_skipped,
            "pcsc_res_valid_count": int(valid_res.sum()),
            "pcsc_l_cf": float(losses["cf"].detach()),
            "pcsc_l_cfres": float(losses["cfres"].detach()),
            "pcsc_l_base": 0.0,
            "pcsc_l_res": float(losses["res"].detach()),
            "pcsc_l_comp": float(losses["comp"].detach()),
            "pcsc_l_strict_sem_full": float(losses["comp"].detach()),
            "pcsc_strict_budget_scale": budget_scale,
        }

    @staticmethod
    def _empty_metrics(soft_skipped=0):
        return {
            "pcsc_item_count": 0,
            "pcsc_hard_item_ratio": 0.0,
            "pcsc_soft_item_skipped": int(soft_skipped),
            "pcsc_res_valid_count": 0,
            "pcsc_l_cf": 0.0,
            "pcsc_l_cfres": 0.0,
            "pcsc_l_base": 0.0,
            "pcsc_l_res": 0.0,
            "pcsc_l_comp": 0.0,
            "pcsc_l_strict_sem_full": 0.0,
            "pcsc_strict_budget_scale": 1.25,
        }
