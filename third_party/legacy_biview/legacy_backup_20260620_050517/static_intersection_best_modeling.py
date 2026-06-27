from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_letter import LETTER


ROLE_LAYOUTS = {
    "original": {"shared": 1, "cf_private": 2, "sem_private": 3},
    "swapped_c1c2": {"cf_private": 1, "shared": 2, "sem_private": 3},
}


class StaticBestHardOnlyLETTER(LETTER):
    def __init__(
        self,
        config,
        pcsc_aux=False,
        pcsc_mode="original",
        lambda_cf=1.0,
        lambda_cfres=1.0,
        lambda_sem=1.0,
        lambda_semres=1.0,
    ):
        super().__init__(config)
        self.pcsc_aux = bool(pcsc_aux)
        self.pcsc_mode = pcsc_mode
        if pcsc_mode not in {"off", "original", "swapped_c1c2"}:
            raise ValueError(f"unknown pcsc_mode: {pcsc_mode}")
        self.role_layout = ROLE_LAYOUTS["original" if pcsc_mode in {"off", "original"} else "swapped_c1c2"]
        self.pcsc_lambdas = {
            "cf": float(lambda_cf),
            "cfres": float(lambda_cfres),
            "sem": float(lambda_sem),
            "semres": float(lambda_semres),
        }
        self._pcsc_factor = 0.0
        self._hash_base = 0
        self.last_curriculum_metrics = {}
        self._last_item_records = []

        self.register_buffer("_sid_hashes", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_pcsc_zcf", torch.empty(0, 128), persistent=False)
        self.register_buffer("_pcsc_cfres", torch.empty(0, 128), persistent=False)
        self.register_buffer("_pcsc_zsem", torch.empty(0, 768), persistent=False)
        self.register_buffer("_pcsc_semres", torch.empty(0, 768), persistent=False)

        self.pcsc_cf_head = nn.Sequential(nn.Linear(config.d_model, 256), nn.ReLU(), nn.Linear(256, 128))
        self.pcsc_cfres_head = nn.Sequential(nn.Linear(config.d_model, 256), nn.ReLU(), nn.Linear(256, 128))
        self.pcsc_sem_head = nn.Sequential(nn.Linear(config.d_model, 512), nn.ReLU(), nn.Linear(512, 768))
        self.pcsc_semres_head = nn.Sequential(nn.Linear(config.d_model, 512), nn.ReLU(), nn.Linear(512, 768))

    def configure_items(self, tokenizer, index_path, order_path, cf_path, sem_path, cf_res_path, sem_res_path):
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        order = [str(x) for x in json.loads(Path(order_path).read_text(encoding="utf-8"))]
        cf = np.load(cf_path).astype(np.float32)
        sem = np.load(sem_path).astype(np.float32)
        cf_res = np.load(cf_res_path).astype(np.float32)
        sem_res = np.load(sem_res_path).astype(np.float32)
        if not (len(order) == len(cf) == len(sem) == len(cf_res) == len(sem_res)):
            raise ValueError("resource length mismatch")
        position = {item: row for row, item in enumerate(order)}
        if set(position) != set(map(str, index)):
            raise ValueError("item order and fixed SID index differ")

        base = len(tokenizer)
        hashes, cf_rows, sem_rows, cfres_rows, semres_rows = [], [], [], [], []
        for item, sid in index.items():
            if len(sid) != 4:
                raise ValueError(f"expected 4 SID tokens for {item}")
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
            cf_rows.append(cf[row])
            sem_rows.append(sem[row])
            cfres_rows.append(cf_res[row])
            semres_rows.append(sem_res[row])
        if len(set(hashes)) != len(hashes):
            raise ValueError("duplicate SID hash detected")
        sort = np.argsort(hashes)
        self._hash_base = base
        self._sid_hashes = torch.from_numpy(np.asarray(hashes, dtype=np.int64)[sort])
        self._pcsc_zcf = torch.from_numpy(np.asarray(cf_rows, dtype=np.float32)[sort])
        self._pcsc_zsem = torch.from_numpy(np.asarray(sem_rows, dtype=np.float32)[sort])
        self._pcsc_cfres = torch.from_numpy(np.asarray(cfres_rows, dtype=np.float32)[sort])
        self._pcsc_semres = torch.from_numpy(np.asarray(semres_rows, dtype=np.float32)[sort])

    def set_pcsc_schedule_factor(self, factor):
        self._pcsc_factor = float(max(0.0, min(1.0, factor)))

    def _lookup(self, groups):
        value = groups[:, 0]
        for column in range(1, 4):
            value = value * self._hash_base + groups[:, column]
        rows = torch.searchsorted(self._sid_hashes, value)
        valid = rows < len(self._sid_hashes)
        safe = rows.clamp(max=len(self._sid_hashes) - 1)
        valid = valid & (self._sid_hashes[safe] == value)
        if not valid.all():
            raise RuntimeError(f"unmapped SID groups: {(~valid).sum().item()}")
        return safe

    def _hard_embeddings(self, input_ids, attention_mask):
        embeddings = self.shared(input_ids)
        records = []
        for start in range(0, input_ids.shape[1] - 3, 4):
            valid = attention_mask[:, start:start + 4].all(dim=1)
            if valid.any():
                batch_rows = torch.where(valid)[0]
                rows = self._lookup(input_ids[batch_rows, start:start + 4])
                records.append((batch_rows, start, rows))
        if self.training and not records:
            raise RuntimeError("no complete four-token history item found")
        mapped = sum(len(x[0]) for x in records)
        self._last_item_records = records
        self.last_curriculum_metrics = {
            "p_sid": 1.0,
            "soft_item_ratio": 0.0,
            "hard_item_ratio": 1.0,
            "mapped_item_count": mapped,
            "hard_only": True,
            "soft_curriculum_enabled": False,
            "nan": bool(not torch.isfinite(embeddings).all().item()),
        }
        return embeddings

    @staticmethod
    def _cosine_loss(pred, target):
        return (1 - F.cosine_similarity(
            F.normalize(pred, dim=1, eps=1e-8),
            F.normalize(target.detach(), dim=1, eps=1e-8),
            dim=1,
        )).mean()

    @staticmethod
    def _fuse(a, b):
        return (a + b) / 2.0

    def _pcsc_loss(self, hidden):
        if self.pcsc_mode == "off" or not self.pcsc_aux:
            return hidden.sum() * 0, {
                "pcsc_item_count": 0,
                "pcsc_l_cf": 0.0,
                "pcsc_l_cfres": 0.0,
                "pcsc_l_sem": 0.0,
                "pcsc_l_semres": 0.0,
            }
        role = self.role_layout
        h_shared, h_cf, h_sem, rows = [], [], [], []
        for batch_rows, start, item_rows in self._last_item_records:
            h_shared.append(hidden[batch_rows, start + role["shared"] - 1])
            h_cf.append(hidden[batch_rows, start + role["cf_private"] - 1])
            h_sem.append(hidden[batch_rows, start + role["sem_private"] - 1])
            rows.append(item_rows)
        if not rows:
            return hidden.sum() * 0, {"pcsc_item_count": 0}
        h_shared = torch.cat(h_shared)
        h_cf = torch.cat(h_cf)
        h_sem = torch.cat(h_sem)
        rows = torch.cat(rows)
        cf_hat = self.pcsc_cf_head(self._fuse(h_shared, h_cf))
        cfres_hat = self.pcsc_cfres_head(h_cf)
        sem_hat = self.pcsc_sem_head(self._fuse(h_shared, h_sem))
        semres_hat = self.pcsc_semres_head(h_sem)
        losses = {
            "cf": self._cosine_loss(cf_hat, self._pcsc_zcf[rows]),
            "cfres": self._cosine_loss(cfres_hat, self._pcsc_cfres[rows]),
            "sem": self._cosine_loss(sem_hat, self._pcsc_zsem[rows]),
            "semres": self._cosine_loss(semres_hat, self._pcsc_semres[rows]),
        }
        total = sum(self._pcsc_factor * self.pcsc_lambdas[k] * losses[k] for k in losses)
        return total, {
            "pcsc_item_count": len(rows),
            "pcsc_role_shared": role["shared"],
            "pcsc_role_cf_private": role["cf_private"],
            "pcsc_role_sem_private": role["sem_private"],
            **{f"pcsc_l_{k}": float(v.detach()) for k, v in losses.items()},
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        original_ids = input_ids
        if input_ids is not None:
            inputs_embeds = self._hard_embeddings(input_ids, attention_mask)
            outputs = super().forward(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=self.pcsc_aux,
                return_dict=True,
                **kwargs,
            )
        else:
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        sid_ce_loss = outputs.loss
        pcsc_loss = sid_ce_loss.sum() * 0
        if self.pcsc_aux and self.training and original_ids is not None:
            pcsc_loss, pcsc_metrics = self._pcsc_loss(outputs.encoder_last_hidden_state)
            self.last_curriculum_metrics.update(pcsc_metrics)
        outputs.loss = sid_ce_loss + pcsc_loss
        self.last_curriculum_metrics.update({
            "pcsc_factor": self._pcsc_factor,
            **{f"lambda_{k}_eff": self._pcsc_factor * v for k, v in self.pcsc_lambdas.items()},
            "sid_ce_loss": float(sid_ce_loss.detach()),
            "pcsc_aux_loss": float(pcsc_loss.detach()),
            "total_loss": float(outputs.loss.detach()),
        })
        return outputs
