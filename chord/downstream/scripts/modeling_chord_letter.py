from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_matched_curriculum_letter import MatchedCurriculumLETTER


class CHORDLETTER(MatchedCurriculumLETTER):
    def __init__(self, config, order, pcsc_aux=False, pcsc_h12_mode="mean", lambda_shared=1.0, lambda_level2=1.0, lambda_level3=1.0):
        super().__init__(
            config,
            mode="zcf",
            pcsc_aux=pcsc_aux,
            pcsc_h12_mode=pcsc_h12_mode,
            lambda_cf=lambda_shared,
            lambda_cfres=lambda_level2,
            lambda_base=0.0,
            lambda_res=lambda_level3,
            lambda_comp=0.0,
        )
        self.order = order
        self.pcsc_lambdas = {
            "shared": float(lambda_shared),
            "level2": float(lambda_level2),
            "level3": float(lambda_level3),
        }
        self.register_buffer("_pcsc_shared", torch.empty(0, 1), persistent=False)
        self.register_buffer("_pcsc_level2", torch.empty(0, 1), persistent=False)
        self.register_buffer("_pcsc_level3", torch.empty(0, 1), persistent=False)
        self.pcsc_shared_head = None
        self.pcsc_level2_head = None
        self.pcsc_level3_head = None

    def configure_pls_items(self, tokenizer, index_path, order_path, shared_path, level2_path, level3_path):
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        order = [str(x) for x in json.loads(Path(order_path).read_text(encoding="utf-8"))]
        shared = np.load(shared_path).astype(np.float32)
        level2 = np.load(level2_path).astype(np.float32)
        level3 = np.load(level3_path).astype(np.float32)
        if not (len(order) == len(shared) == len(level2) == len(level3)):
            raise ValueError("PLS target order mismatch")
        position = {item: row for row, item in enumerate(order)}
        if set(position) != set(map(str, index)):
            raise ValueError("item order and fixed SID index differ")
        base = len(tokenizer)
        hashes, shared_rows, level2_rows, level3_rows, sid_token_ids = [], [], [], [], []
        for item, sid in index.items():
            ids = []
            for token in sid:
                token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                if len(token_ids) != 1:
                    raise ValueError(f"SID token is not single-token: {token} -> {token_ids}")
                ids.append(token_ids[0])
            value = ids[0]
            for token_id in ids[1:]:
                value = value * base + token_id
            row = position[str(item)]
            hashes.append(value)
            shared_rows.append(shared[row])
            level2_rows.append(level2[row])
            level3_rows.append(level3[row])
            sid_token_ids.extend(ids)
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate SID hash detected")
        sort = np.argsort(hashes)
        self._hash_base = base
        self._sid_hashes = torch.from_numpy(np.asarray(hashes, dtype=np.int64)[sort])
        self._zcf = torch.from_numpy(np.asarray(shared_rows, dtype=np.float32)[sort])
        self._pcsc_shared = self._zcf.clone()
        self._pcsc_level2 = torch.from_numpy(np.asarray(level2_rows, dtype=np.float32)[sort])
        self._pcsc_level3 = torch.from_numpy(np.asarray(level3_rows, dtype=np.float32)[sort])
        self._pcsc_cfres = self._pcsc_level2
        self._pcsc_usem_raw = self._pcsc_level3
        self.pcsc_shared_head = nn.Sequential(nn.Linear(self.config.d_model, 256), nn.ReLU(), nn.Linear(256, self._pcsc_shared.shape[1]))
        self.pcsc_level2_head = nn.Sequential(nn.Linear(self.config.d_model, 512), nn.ReLU(), nn.Linear(512, self._pcsc_level2.shape[1]))
        self.pcsc_level3_head = nn.Sequential(nn.Linear(self.config.d_model, 512), nn.ReLU(), nn.Linear(512, self._pcsc_level3.shape[1]))
        with torch.no_grad():
            unique_ids = torch.tensor(sorted(set(sid_token_ids)), dtype=torch.long)
            self._hard_norm_target = float(self.shared(unique_ids).norm(dim=1).mean())

    def _pcsc_loss(self, hidden):
        hard_h1, hard_h2, hard_h3, target_rows = [], [], [], []
        soft_skipped = 0
        for batch_rows, start, rows, is_hard in self._last_item_records:
            soft_skipped += int((~is_hard).sum())
            if not is_hard.any():
                continue
            selected_batch, selected_rows = batch_rows[is_hard], rows[is_hard]
            hard_h1.append(hidden[selected_batch, start])
            hard_h2.append(hidden[selected_batch, start + 1])
            hard_h3.append(hidden[selected_batch, start + 2])
            target_rows.append(selected_rows)
        if not target_rows:
            zero = hidden.sum() * 0
            return zero, {
                "pcsc_item_count": 0,
                "pcsc_hard_item_ratio": 0.0,
                "pcsc_soft_item_skipped": soft_skipped,
                "pcsc_l_shared": 0.0,
                "pcsc_l_level2": 0.0,
                "pcsc_l_level3": 0.0,
            }
        h1 = torch.cat(hard_h1)
        h2 = torch.cat(hard_h2)
        h3 = torch.cat(hard_h3)
        rows = torch.cat(target_rows)
        shared_hat = self.pcsc_shared_head(h1)
        level2_hat = self.pcsc_level2_head(h2)
        level3_hat = self.pcsc_level3_head(h3)
        losses = {
            "shared": self._cosine_loss(shared_hat, self._pcsc_shared[rows]),
            "level2": self._cosine_loss(level2_hat, self._pcsc_level2[rows]),
            "level3": self._cosine_loss(level3_hat, self._pcsc_level3[rows]),
        }
        total = sum(self._pcsc_factor * self.pcsc_lambdas[k] * losses[k] for k in losses)
        return total, {
            "pcsc_item_count": len(rows),
            "pcsc_hard_item_ratio": len(rows) / max(len(rows) + soft_skipped, 1),
            "pcsc_soft_item_skipped": soft_skipped,
            **{f"pcsc_l_{k}": float(v.detach()) for k, v in losses.items()},
        }
