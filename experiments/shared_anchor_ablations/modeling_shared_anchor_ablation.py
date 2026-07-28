"""Capacity-matched Beauty ablations for shared-anchored cross-view PCSC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modeling_matched_curriculum_crossview import MatchedCurriculumLETTER as _Base


class MatchedCurriculumLETTER(_Base):
    PCSC_CONTRACT_VERSION = "shared-anchor-ablation-v1"
    POSITIONAL_PCSC_CONTRACT = "role_aware_shared_anchor_ablation"

    def __init__(self, *args, pcsc_ablation_variant="a7_main", **kwargs):
        super().__init__(*args, **kwargs)
        allowed = {
            "a0_ceonly", "a1_same", "a2_prefix_same", "a3_prefix_cross",
            "a4_same_cross", "a5_same_add", "a6_same_cross_add", "a7_main",
        }
        if pcsc_ablation_variant not in allowed:
            raise ValueError(f"Unknown PCSC ablation: {pcsc_ablation_variant}")
        self.pcsc_ablation_variant = pcsc_ablation_variant
        d_model = self.config.d_model
        self.register_buffer("_pcsc_consensus", torch.empty(0, 128), persistent=False)
        self.pcsc_consensus_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.pcsc_sem_full_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(), nn.Linear(512, 768)
        )

    def configure_items(
        self, tokenizer, index_path, order_path, cf_path, sem_path,
        rqvae_checkpoint_path=None, cf_res_path=None, sem_base_path=None,
        sem_res_raw_path=None, shared_path=None,
    ):
        super().configure_items(
            tokenizer, index_path, order_path, cf_path, sem_path,
            rqvae_checkpoint_path=rqvae_checkpoint_path,
            cf_res_path=cf_res_path, sem_base_path=sem_base_path,
            sem_res_raw_path=sem_res_raw_path,
        )
        if self.pcsc_ablation_variant == "a0_ceonly":
            return
        if shared_path is None:
            raise ValueError("Shared-anchor ablations require --shared_emb")
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        order = [str(x) for x in json.loads(Path(order_path).read_text(encoding="utf-8"))]
        position = {item: row for row, item in enumerate(order)}
        shared = np.load(shared_path).astype(np.float32)
        if shared.shape != (len(order), 128):
            raise ValueError(f"Expected shared shape {(len(order), 128)}, got {shared.shape}")
        base, hashes, rows = len(tokenizer), [], []
        for item, sid in index.items():
            ids = [tokenizer(x, add_special_tokens=False)["input_ids"] for x in sid]
            if any(len(x) != 1 for x in ids):
                raise ValueError(f"Non-atomic SID for item {item}")
            ids = [x[0] for x in ids]
            value = ids[0]
            for token_id in ids[1:]:
                value = value * base + token_id
            hashes.append(value)
            rows.append(shared[position[str(item)]])
        sort = np.argsort(hashes)
        self._pcsc_consensus = torch.from_numpy(np.asarray(rows, dtype=np.float32)[sort])

    def _pcsc_loss(self, hidden):
        records = self._last_item_records
        if not records or self.pcsc_ablation_variant == "a0_ceonly":
            return hidden.sum() * 0, self._metrics()

        batch_rows = torch.cat([x[0] for x in records])
        target_rows = torch.cat([x[2] for x in records])
        hard = torch.cat([x[3] for x in records])
        lengths = torch.tensor([x[0].numel() for x in records], device=hidden.device)
        starts = torch.repeat_interleave(
            torch.tensor([x[1] for x in records], device=hidden.device), lengths
        )
        skipped = int((~hard).sum().item())
        batch_rows, target_rows, starts = batch_rows[hard], target_rows[hard], starts[hard]
        if target_rows.numel() == 0:
            return hidden.sum() * 0, self._metrics(skipped)

        pos = self._component_position
        hs = hidden[batch_rows, starts + pos["shared"]]
        hsem = hidden[batch_rows, starts + pos["semres"]]
        hcf = hidden[batch_rows, starts + pos["cfres"]]
        zshared = self._pcsc_consensus[target_rows]
        zcf, rcf = self._pcsc_zcf[target_rows], self._pcsc_cfres[target_rows]
        zsem, rsem = self._pcsc_zsem[target_rows], self._pcsc_usem_raw[target_rows]
        valid_sem = rsem.norm(dim=1) >= 1e-5

        anchor = self._cosine_loss(self.pcsc_consensus_head(hs), zshared)
        same = [
            self._cosine_loss(self.pcsc_sem_full_head(hs + hsem), zsem),
            self._cosine_loss(self.pcsc_res_head(hsem)[valid_sem], rsem[valid_sem])
            if valid_sem.any() else hidden.sum() * 0,
            self._cosine_loss(self.pcsc_cf_head(hs + hcf), zcf),
            self._cosine_loss(self.pcsc_cfres_head(hcf), rcf),
        ]
        cross = [
            self._cosine_loss(self.pcsc_cf_head(hs + hsem), zcf),
            self._cosine_loss(self.pcsc_cfres_head(hsem), rcf),
            self._cosine_loss(self.pcsc_sem_full_head(hs + hcf), zsem),
            self._cosine_loss(self.pcsc_res_head(hcf)[valid_sem], rsem[valid_sem])
            if valid_sem.any() else hidden.sum() * 0,
        ]
        additive = self._cosine_loss(
            self.pcsc_base_head(hs) + self.pcsc_res_head(hsem), zsem
        )
        variant = self.pcsc_ablation_variant
        if variant == "a1_same":
            objective = anchor + sum(same)
        elif variant == "a2_prefix_same":
            objective = anchor + 2.0 * (same[0] + same[2])
        elif variant == "a3_prefix_cross":
            objective = anchor + 2.0 * (cross[0] + cross[2])
        elif variant == "a4_same_cross":
            objective = anchor + 0.5 * (sum(same) + sum(cross))
        elif variant == "a5_same_add":
            objective = anchor + 0.5 * (sum(same) + 4.0 * additive)
        elif variant == "a6_same_cross_add":
            objective = anchor + (sum(same) + sum(cross) + 4.0 * additive) / 3.0
        else:
            objective = anchor + sum(cross)
        total = self._pcsc_factor * objective
        metrics = self._metrics(skipped)
        metrics.update({
            "pcsc_item_count": int(target_rows.numel()),
            "pcsc_hard_item_ratio": target_rows.numel() / max(target_rows.numel() + skipped, 1),
            "pcsc_l_consensus": float(anchor.detach()),
            "pcsc_l_same": float(torch.stack(same).sum().detach()),
            "pcsc_l_cross": float(torch.stack(cross).sum().detach()),
            "pcsc_l_additive": float(additive.detach()),
        })
        return total, metrics

    @staticmethod
    def _metrics(skipped=0):
        return {
            "pcsc_item_count": 0, "pcsc_hard_item_ratio": 0.0,
            "pcsc_soft_item_skipped": int(skipped), "pcsc_res_valid_count": 0,
            "pcsc_l_cf": 0.0, "pcsc_l_cfres": 0.0, "pcsc_l_base": 0.0,
            "pcsc_l_res": 0.0, "pcsc_l_comp": 0.0, "pcsc_l_consensus": 0.0,
            "pcsc_l_same": 0.0, "pcsc_l_cross": 0.0, "pcsc_l_additive": 0.0,
        }
