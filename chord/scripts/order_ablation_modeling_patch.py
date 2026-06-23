from __future__ import annotations

import torch

from modeling_matched_curriculum_letter import MatchedCurriculumLETTER


class OrderAwareMatchedCurriculumLETTER(MatchedCurriculumLETTER):
    """CHORD order ablation wrapper with role-aware PCSC targets."""

    def __init__(self, *args, chord_order="cf_first", **kwargs):
        super().__init__(*args, **kwargs)
        if chord_order not in {"cf_first", "sem_first"}:
            raise ValueError(f"Unknown chord_order: {chord_order}")
        self.chord_order = chord_order

    def pcsc_mapping_summary(self):
        if self.chord_order == "cf_first":
            return {
                "h2": "cf_residual",
                "h3": "semantic_residual",
                "fusion_h1_h2": "cf_embedding_and_semantic_base",
                "fusion_h1_h3": "unused",
            }
        return {
            "h2": "semantic_residual",
            "h3": "cf_residual",
            "fusion_h1_h2": "st5_embedding_and_semantic_base",
            "fusion_h1_h3": "cf_embedding",
        }

    def _fuse(self, h1, h_private):
        return (h1 + h_private) / 2 if self.pcsc_h12_mode == "mean" else h_private

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
                "pcsc_item_count": 0, "pcsc_hard_item_ratio": 0.0,
                "pcsc_soft_item_skipped": soft_skipped,
                "pcsc_res_valid_count": 0, "pcsc_l_cf": 0.0,
                "pcsc_l_cfres": 0.0, "pcsc_l_base": 0.0,
                "pcsc_l_res": 0.0, "pcsc_l_comp": 0.0,
            }
        h1 = torch.cat(hard_h1)
        h2 = torch.cat(hard_h2)
        h3 = torch.cat(hard_h3)
        rows = torch.cat(target_rows)

        zcf = self._pcsc_zcf[rows]
        cfres = self._pcsc_cfres[rows]
        zsem = self._pcsc_zsem[rows]
        zbase = self._pcsc_zsem_base[rows]
        ures = self._pcsc_usem_raw[rows]
        valid_res = ures.norm(dim=1) >= 1e-5

        if self.chord_order == "cf_first":
            cf_context = self._fuse(h1, h2)
            sem_context = self._fuse(h1, h2)
            cfres_context = h2
            semres_context = h3
        else:
            sem_context = self._fuse(h1, h2)
            cf_context = self._fuse(h1, h3)
            semres_context = h2
            cfres_context = h3

        cf_hat = self.pcsc_cf_head(cf_context)
        cfres_hat = self.pcsc_cfres_head(cfres_context)
        base_hat = self.pcsc_base_head(sem_context)
        res_hat = self.pcsc_res_head(semres_context)
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
