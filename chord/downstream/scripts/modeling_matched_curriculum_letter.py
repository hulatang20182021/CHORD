from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_letter import LETTER


class MatchedCurriculumLETTER(LETTER):
    def __init__(
        self,
        config,
        mode,
        pcsc_aux=False,
        pcsc_h12_mode="mean",
        pcsc_alignment="component",
        sid_component_order="shared,cfres,semres",
        lambda_cf=0.003,
        lambda_cfres=0.001,
        lambda_base=0.002,
        lambda_res=0.001,
        lambda_comp=0.003,
    ):
        super().__init__(config)
        self.mode = mode
        self.pcsc_aux = bool(pcsc_aux)
        self.pcsc_h12_mode = pcsc_h12_mode
        if pcsc_alignment not in {"component", "positional"}:
            raise ValueError("pcsc_alignment must be one of: component, positional")
        self.pcsc_alignment = pcsc_alignment
        self.sid_component_order = tuple(x.strip() for x in str(sid_component_order).split(",") if x.strip())
        expected = {"shared", "cfres", "semres"}
        if len(self.sid_component_order) != 3 or set(self.sid_component_order) != expected:
            raise ValueError(
                "sid_component_order must be a comma-separated permutation of "
                "shared,cfres,semres"
            )
        self._component_position = {name: idx for idx, name in enumerate(self.sid_component_order)}
        self.pcsc_lambdas = {
            "cf": float(lambda_cf),
            "cfres": float(lambda_cfres),
            "base": float(lambda_base),
            "res": float(lambda_res),
            "comp": float(lambda_comp),
        }
        self._pcsc_factor = 0.0
        self._p_sid = 1.0
        self._force_soft = False
        self._hash_base = 0
        self.last_curriculum_metrics = {}
        self._last_item_records = []

        if mode == "zcf":
            self.soft_projector = nn.Linear(128, config.d_model)
            self.soft_norm = nn.LayerNorm(config.d_model)
            self.level_embeddings = nn.Parameter(torch.zeros(4, config.d_model))
            nn.init.normal_(self.level_embeddings, std=0.02)
            self.layer_projectors = None
            self.layer_norms = None
        elif mode in {"layered_rq", "layered_rq_pcsc"}:
            self.soft_projector = None
            self.soft_norm = None
            self.level_embeddings = None
            self.layer_projectors = nn.ModuleList([
                nn.Linear(64, config.d_model) for _ in range(3)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.d_model) for _ in range(3)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.register_buffer("_sid_hashes", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_zcf", torch.empty(0, 128), persistent=False)
        self.register_buffer("_layered_q", torch.empty(0, 3, 64), persistent=False)
        self.register_buffer("_pcsc_zcf", torch.empty(0, 128), persistent=False)
        self.register_buffer("_pcsc_cfres", torch.empty(0, 128), persistent=False)
        self.register_buffer("_pcsc_zsem", torch.empty(0, 768), persistent=False)
        self.register_buffer("_pcsc_zsem_base", torch.empty(0, 768), persistent=False)
        self.register_buffer("_pcsc_usem_raw", torch.empty(0, 768), persistent=False)
        self._hard_norm_target = 1.0
        self._hard_norm_targets = [1.0, 1.0, 1.0]

        self.pcsc_cf_head = nn.Sequential(nn.Linear(config.d_model, 256), nn.ReLU(), nn.Linear(256, 128))
        self.pcsc_cfres_head = nn.Sequential(nn.Linear(config.d_model, 256), nn.ReLU(), nn.Linear(256, 128))
        self.pcsc_base_head = nn.Sequential(nn.Linear(config.d_model, 512), nn.ReLU(), nn.Linear(512, 768))
        self.pcsc_res_head = nn.Sequential(nn.Linear(config.d_model, 512), nn.ReLU(), nn.Linear(512, 768))

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
    ):
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        order = [str(x) for x in json.loads(Path(order_path).read_text(encoding="utf-8"))]
        cf = np.load(cf_path).astype(np.float32)
        sem = np.load(sem_path).astype(np.float32)
        cf_res = np.load(cf_res_path).astype(np.float32) if cf_res_path else np.zeros_like(cf)
        sem_base = np.load(sem_base_path).astype(np.float32) if sem_base_path else np.zeros_like(sem)
        sem_res_raw = np.load(sem_res_raw_path).astype(np.float32) if sem_res_raw_path else sem - sem_base
        if not (len(order) == len(cf) == len(cf_res) == len(sem) == len(sem_base) == len(sem_res_raw)):
            raise ValueError("Embedding order mismatch")
        position = {item: row for row, item in enumerate(order)}
        if set(position) != set(map(str, index)):
            raise ValueError("item order and fixed SID index differ")

        codebooks = None
        if self.mode in {"layered_rq", "layered_rq_pcsc"}:
            if rqvae_checkpoint_path is None:
                raise ValueError("Layered RQ mode requires rqvae checkpoint")
            checkpoint = torch.load(rqvae_checkpoint_path, map_location="cpu", weights_only=False)
            state = checkpoint["model_state_dict"]
            codebooks = [
                state[f"codebooks.{level}"].detach().cpu().numpy().astype(np.float32)
                for level in range(3)
            ]

        base = len(tokenizer)
        hashes, zcf_rows, cfres_rows, pcsc_sem_rows, pcsc_base_rows, pcsc_res_rows = [], [], [], [], [], []
        layered_rows = []
        sid_token_ids = []
        level_token_ids = [[], [], []]
        for item, sid in index.items():
            if len(sid) != 4:
                raise ValueError(f"Expected four SID tokens for {item}")
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
            zcf_rows.append(cf[row])
            cfres_rows.append(cf_res[row])
            pcsc_sem_rows.append(sem[row])
            pcsc_base_rows.append(sem_base[row])
            pcsc_res_rows.append(sem_res_raw[row])
            sid_token_ids.extend(ids)
            if codebooks is not None:
                codes = [int(sid[level].split("_", 1)[1].rstrip(">")) for level in range(3)]
                layered_rows.append([codebooks[level][codes[level]] for level in range(3)])
                for level in range(3):
                    level_token_ids[level].append(ids[level])

        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate SID hash detected")
        sort = np.argsort(hashes)
        self._hash_base = base
        self._sid_hashes = torch.from_numpy(np.asarray(hashes, dtype=np.int64)[sort])
        self._zcf = torch.from_numpy(np.asarray(zcf_rows, dtype=np.float32)[sort])
        self._pcsc_zcf = self._zcf.clone()
        self._pcsc_cfres = torch.from_numpy(np.asarray(cfres_rows, dtype=np.float32)[sort])
        self._pcsc_zsem = torch.from_numpy(np.asarray(pcsc_sem_rows, dtype=np.float32)[sort])
        self._pcsc_zsem_base = torch.from_numpy(np.asarray(pcsc_base_rows, dtype=np.float32)[sort])
        self._pcsc_usem_raw = torch.from_numpy(np.asarray(pcsc_res_rows, dtype=np.float32)[sort])
        if layered_rows:
            self._layered_q = torch.from_numpy(np.asarray(layered_rows, dtype=np.float32)[sort])

        with torch.no_grad():
            unique_ids = torch.tensor(sorted(set(sid_token_ids)), dtype=torch.long)
            self._hard_norm_target = float(self.shared(unique_ids).norm(dim=1).mean())
            if level_token_ids[0]:
                self._hard_norm_targets = []
                for token_ids in level_token_ids:
                    unique = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
                    self._hard_norm_targets.append(float(self.shared(unique).norm(dim=1).mean()))

    def set_curriculum(self, p_sid, force_soft=False):
        self._p_sid = float(p_sid)
        self._force_soft = bool(force_soft)

    def set_pcsc_schedule_factor(self, factor):
        self._pcsc_factor = float(max(0.0, min(1.0, factor)))

    def projector_parameters(self):
        if self.mode == "zcf":
            return list(self.soft_projector.parameters()) + list(self.soft_norm.parameters()) + [self.level_embeddings]
        return list(self.layer_projectors.parameters()) + list(self.layer_norms.parameters())

    def _lookup(self, groups):
        value = groups[:, 0]
        for column in range(1, 4):
            value = value * self._hash_base + groups[:, column]
        rows = torch.searchsorted(self._sid_hashes, value)
        valid = rows < len(self._sid_hashes)
        safe = rows.clamp(max=len(self._sid_hashes) - 1)
        valid = valid & (self._sid_hashes[safe] == value)
        if not valid.all():
            raise RuntimeError(f"Unmapped history SID groups: {(~valid).sum().item()}")
        return safe

    def _soft_zcf(self, rows):
        output = self.soft_norm(self.soft_projector(self._zcf[rows]))
        scale = self._hard_norm_target / (output.shape[-1] ** 0.5)
        return output * scale

    def _soft_layered(self, rows):
        source = self._layered_q[rows]
        outputs = []
        for level in range(3):
            projected = self.layer_norms[level](self.layer_projectors[level](source[:, level]))
            scale = self._hard_norm_targets[level] / (projected.shape[-1] ** 0.5)
            outputs.append(projected * scale)
        return torch.stack(outputs, dim=1)

    def _curriculum_embeddings(self, input_ids, attention_mask):
        embeddings = self.shared(input_ids)
        if not self.training:
            self.last_curriculum_metrics = {
                "p_sid": 1.0, "soft_item_ratio": 0.0,
                "hard_item_ratio": 1.0, "nan": False,
            }
            return embeddings
        records = []
        for start in range(0, input_ids.shape[1] - 3, 4):
            valid = attention_mask[:, start:start + 4].all(dim=1)
            if valid.any():
                batch_rows = torch.where(valid)[0]
                rows = self._lookup(input_ids[batch_rows, start:start + 4])
                records.append((batch_rows, start, rows))
        if not records:
            raise RuntimeError("No complete four-token history item found")

        total_items = sum(len(x[0]) for x in records)
        p_sid = 0.0 if self._force_soft else self._p_sid
        hard_mask = torch.rand(total_items, device=input_ids.device) < p_sid
        cursor = 0
        soft_norms, hard_norms, proj_norms = [], [], []
        pcsc_records = []
        for batch_rows, start, rows in records:
            count = len(batch_rows)
            local_hard = hard_mask[cursor:cursor + count]
            cursor += count
            pcsc_records.append((batch_rows, start, rows, local_hard))
            hard = embeddings[batch_rows, start:start + 4]
            hard_norms.append(hard.norm(dim=2).reshape(-1))
            if (~local_hard).any():
                selected_batch = batch_rows[~local_hard]
                selected_rows = rows[~local_hard]
                if self.mode == "zcf":
                    base = self._soft_zcf(selected_rows)
                    soft = base[:, None, :] + self.level_embeddings[None, :, :]
                    embeddings[selected_batch, start:start + 4] = soft
                    proj_norms.append(base.norm(dim=1))
                    soft_norms.append(soft.norm(dim=2).reshape(-1))
                else:
                    soft = self._soft_layered(selected_rows)
                    embeddings[selected_batch, start:start + 3] = soft
                    soft_norms.append(soft.norm(dim=2).reshape(-1))

        soft_values = torch.cat(soft_norms) if soft_norms else torch.empty(0, device=input_ids.device)
        hard_values = torch.cat(hard_norms)
        proj_values = torch.cat(proj_norms) if proj_norms else torch.empty(0, device=input_ids.device)

        def stat(values, q):
            return float(torch.quantile(values.detach(), q)) if values.numel() else 0.0

        soft_ratio = float((~hard_mask).float().mean())
        self.last_curriculum_metrics = {
            "p_sid": p_sid,
            "soft_item_ratio": soft_ratio,
            "hard_item_ratio": 1.0 - soft_ratio,
            "mapped_item_count": total_items,
            "soft_norm_mean": float(soft_values.detach().mean()) if soft_values.numel() else 0.0,
            "soft_norm_median": stat(soft_values, 0.5),
            "hard_norm_mean": float(hard_values.detach().mean()),
            "hard_norm_median": stat(hard_values, 0.5),
            "proj_norm_mean": float(proj_values.detach().mean()) if proj_values.numel() else 0.0,
            "c4_kept_hard": self.mode != "zcf",
            "nan": bool(not torch.isfinite(embeddings).all().item()),
        }
        self._last_item_records = pcsc_records
        return embeddings

    @staticmethod
    def _cosine_loss(pred, target):
        return (1 - F.cosine_similarity(
            F.normalize(pred, dim=1, eps=1e-8),
            F.normalize(target.detach(), dim=1, eps=1e-8),
            dim=1,
        )).mean()

    def _pcsc_loss(self, hidden):
        hard_h_shared_cf, hard_h_cfres, hard_h_semres, target_rows = [], [], [], []
        soft_skipped = 0
        for batch_rows, start, rows, is_hard in self._last_item_records:
            soft_skipped += int((~is_hard).sum())
            if not is_hard.any():
                continue
            selected_batch, selected_rows = batch_rows[is_hard], rows[is_hard]
            if self.pcsc_alignment == "positional":
                # Keep the legacy five-loss contract tied to SID positions.
                h_shared = hidden[selected_batch, start]
                h_cfres = hidden[selected_batch, start + 1]
                h_semres = hidden[selected_batch, start + 2]
            else:
                component_hidden = {
                    name: hidden[selected_batch, start + offset]
                    for name, offset in self._component_position.items()
                }
                h_shared = component_hidden["shared"]
                h_cfres = component_hidden["cfres"]
                h_semres = component_hidden["semres"]
            if self.pcsc_h12_mode == "mean":
                h12 = (h_shared + h_cfres) / 2
            elif self.pcsc_h12_mode == "sum":
                h12 = h_shared + h_cfres
            else:
                h12 = h_cfres
            hard_h_shared_cf.append(h12)
            hard_h_cfres.append(h_cfres)
            hard_h_semres.append(h_semres)
            target_rows.append(selected_rows)
        if not target_rows:
            zero = hidden.sum() * 0
            return zero, {
                "pcsc_item_count": 0, "pcsc_hard_item_ratio": 0.0,
                "pcsc_soft_item_skipped": soft_skipped,
                "pcsc_res_valid_count": 0, "pcsc_l_cf": 0.0,
                "pcsc_l_cfres": 0.0, "pcsc_l_base": 0.0,
                "pcsc_l_res": 0.0, "pcsc_l_comp": 0.0,
            }
        h12 = torch.cat(hard_h_shared_cf)
        h2 = torch.cat(hard_h_cfres)
        h3 = torch.cat(hard_h_semres)
        rows = torch.cat(target_rows)
        cf_hat = self.pcsc_cf_head(h12)
        cfres_hat = self.pcsc_cfres_head(h2)
        base_hat = self.pcsc_base_head(h12)
        res_hat = self.pcsc_res_head(h3)
        zcf = self._pcsc_zcf[rows]
        cfres = self._pcsc_cfres[rows]
        zsem = self._pcsc_zsem[rows]
        zbase = self._pcsc_zsem_base[rows]
        ures = self._pcsc_usem_raw[rows]
        valid_res = ures.norm(dim=1) >= 1e-5
        losses = {
            "cf": self._cosine_loss(cf_hat, zcf),
            "cfres": self._cosine_loss(cfres_hat, cfres),
            "base": self._cosine_loss(base_hat, zbase),
            "comp": self._cosine_loss(base_hat + res_hat, zsem),
            "res": self._cosine_loss(res_hat[valid_res], ures[valid_res]) if valid_res.any() else hidden.sum() * 0,
        }
        total = sum(self._pcsc_factor * self.pcsc_lambdas[k] * losses[k] for k in losses)
        return total, {
            "pcsc_item_count": len(rows),
            "pcsc_hard_item_ratio": len(rows) / max(len(rows) + soft_skipped, 1),
            "pcsc_soft_item_skipped": soft_skipped,
            "pcsc_res_valid_count": int(valid_res.sum()),
            **{f"pcsc_l_{k}": float(v.detach()) for k, v in losses.items()},
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        original_ids = input_ids
        if input_ids is not None:
            inputs_embeds = self._curriculum_embeddings(input_ids, attention_mask)
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
