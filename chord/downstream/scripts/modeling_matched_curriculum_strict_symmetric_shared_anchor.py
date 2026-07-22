"""Strict symmetric cross-view PCSC with an explicit PLS shared anchor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from modeling_matched_curriculum_strict_symmetric import (
    MatchedCurriculumLETTER as _StrictSymmetricPCSC,
)


class MatchedCurriculumLETTER(_StrictSymmetricPCSC):
    PCSC_CONTRACT_VERSION = "crossview-strict-symmetric-shared-anchor-v1"
    POSITIONAL_PCSC_CONTRACT = (
        "h1_to_pls_shared;h1_plus_h2_to_cf_full;h2_to_cf_residual;"
        "h1_plus_h3_to_semantic_full;h3_to_semantic_residual;"
        "fixed_total_auxiliary_budget"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_pcsc_consensus", torch.empty(0, 128), persistent=False)
        self.pcsc_consensus_head = nn.Sequential(
            nn.Linear(self.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def configure_items(
        self,
        tokenizer,
        index_path,
        order_path,
        cf_path,
        sem_path,
        rqvae_checkpoint_path=None,
        cf_res_path=None,
        sem_base_path=None,
        sem_res_raw_path=None,
        shared_path=None,
    ):
        super().configure_items(
            tokenizer,
            index_path,
            order_path,
            cf_path,
            sem_path,
            rqvae_checkpoint_path=rqvae_checkpoint_path,
            cf_res_path=cf_res_path,
            sem_base_path=sem_base_path,
            sem_res_raw_path=sem_res_raw_path,
        )
        if shared_path is None:
            raise ValueError("Strict symmetric shared-anchor PCSC requires --shared_emb")

        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        order = [str(value) for value in json.loads(Path(order_path).read_text(encoding="utf-8"))]
        position = {item: row for row, item in enumerate(order)}
        shared = np.load(shared_path).astype(np.float32)
        if shared.shape != (len(order), 128):
            raise ValueError(f"Expected PLS consensus shape {(len(order), 128)}, got {shared.shape}")

        base = len(tokenizer)
        hashes, shared_rows = [], []
        for item, sid in index.items():
            token_ids = []
            for token in sid:
                ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    raise ValueError(f"SID token is not single-token: {token} -> {ids}")
                token_ids.append(ids[0])
            value = token_ids[0]
            for token_id in token_ids[1:]:
                value = value * base + token_id
            hashes.append(value)
            shared_rows.append(shared[position[str(item)]])
        sort = np.argsort(hashes)
        self._pcsc_consensus = torch.from_numpy(np.asarray(shared_rows, dtype=np.float32)[sort])

    def _pcsc_loss(self, hidden):
        strict_total, metrics = super()._pcsc_loss(hidden)
        records = self._last_item_records
        if not records:
            metrics["pcsc_l_consensus"] = 0.0
            return strict_total, metrics

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
        batch_rows, rows, starts = batch_rows[hard_mask], rows[hard_mask], starts[hard_mask]
        if rows.numel() == 0:
            metrics["pcsc_l_consensus"] = 0.0
            return strict_total, metrics

        h1 = hidden[batch_rows, starts]
        consensus_loss = self._cosine_loss(
            self.pcsc_consensus_head(h1), self._pcsc_consensus[rows]
        )

        # Undo the four-loss 5/4 scaling, then add the fifth anchor. This keeps
        # the total auxiliary-loss budget at five unit-weight objectives.
        strict_scale = float(metrics["pcsc_strict_budget_scale"])
        total = strict_total / strict_scale + (
            self._pcsc_factor * self.pcsc_lambdas["base"] * consensus_loss
        )
        metrics["pcsc_l_base"] = float(consensus_loss.detach())
        metrics["pcsc_l_consensus"] = float(consensus_loss.detach())
        metrics["pcsc_shared_anchor_budget_scale"] = 1.0
        return total, metrics
