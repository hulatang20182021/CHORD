#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import T5Config


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "chord/downstream/scripts"))
from modeling_matched_curriculum_letter import MatchedCurriculumLETTER  # noqa: E402


def make_model(alignment: str, order: str = "shared,semres,cfres"):
    config = T5Config(
        vocab_size=128,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = MatchedCurriculumLETTER(
        config,
        mode="layered_rq_pcsc",
        pcsc_aux=True,
        pcsc_h12_mode="sum",
        pcsc_alignment=alignment,
        sid_component_order=order,
        lambda_cf=1.0,
        lambda_cfres=1.0,
        lambda_base=1.0,
        lambda_res=1.0,
        lambda_comp=1.0,
    )
    return model


def main() -> None:
    torch.manual_seed(42)
    positional = make_model("positional")
    role = make_model("role")
    role.load_state_dict(positional.state_dict())
    for model in (positional, role):
        model._pcsc_factor = 0.7
        model._pcsc_zcf = torch.randn(7, 128)
        model._pcsc_cfres = torch.randn(7, 128)
        model._pcsc_zsem = torch.randn(7, 768)
        model._pcsc_zsem_base = torch.randn(7, 768)
        model._pcsc_usem_raw = torch.randn(7, 768)
        model._last_item_records = [
            (
                torch.tensor([0, 1, 2], dtype=torch.long),
                2,
                torch.tensor([1, 3, 5], dtype=torch.long),
                torch.tensor([True, True, True]),
            )
        ]
    role._pcsc_zcf.copy_(positional._pcsc_zcf)
    role._pcsc_cfres.copy_(positional._pcsc_cfres)
    role._pcsc_zsem.copy_(positional._pcsc_zsem)
    role._pcsc_zsem_base.copy_(positional._pcsc_zsem_base)
    role._pcsc_usem_raw.copy_(positional._pcsc_usem_raw)
    hidden = torch.randn(3, 9, 32)
    positional_loss, positional_metrics = positional._pcsc_loss(hidden)
    role_loss, role_metrics = role._pcsc_loss(hidden)
    metric_diffs = {
        key: abs(float(positional_metrics[key]) - float(role_metrics[key]))
        for key in positional_metrics
        if key.startswith("pcsc_l_") and key in role_metrics
    }
    permutation_results = {}
    canonical_order = ("shared", "semres", "cfres")
    for order in ("shared,cfres,semres", "semres,shared,cfres"):
        permuted = make_model("role", order)
        permuted.load_state_dict(positional.state_dict())
        permuted._pcsc_factor = positional._pcsc_factor
        permuted._pcsc_zcf = positional._pcsc_zcf.clone()
        permuted._pcsc_cfres = positional._pcsc_cfres.clone()
        permuted._pcsc_zsem = positional._pcsc_zsem.clone()
        permuted._pcsc_zsem_base = positional._pcsc_zsem_base.clone()
        permuted._pcsc_usem_raw = positional._pcsc_usem_raw.clone()
        permuted._last_item_records = positional._last_item_records
        permuted_hidden = hidden.clone()
        order_tuple = tuple(order.split(","))
        for target_pos, component in enumerate(order_tuple):
            source_pos = canonical_order.index(component)
            permuted_hidden[:, 2 + target_pos] = hidden[:, 2 + source_pos]
        permuted_loss, permuted_metrics = permuted._pcsc_loss(permuted_hidden)
        permuted_metric_diffs = {
            key: abs(float(positional_metrics[key]) - float(permuted_metrics[key]))
            for key in positional_metrics
            if key.startswith("pcsc_l_") and key in permuted_metrics
        }
        permutation_results[order] = {
            "loss": float(permuted_loss.detach()),
            "loss_diff": abs(float(positional_loss.detach()) - float(permuted_loss.detach())),
            "max_metric_diff": max(permuted_metric_diffs.values(), default=0.0),
        }

    result = {
        "passed": torch.equal(positional_loss, role_loss) and max(metric_diffs.values(), default=0.0) == 0.0,
        "positional_loss": float(positional_loss.detach()),
        "role_loss": float(role_loss.detach()),
        "max_metric_diff": max(metric_diffs.values(), default=0.0),
        "metric_diffs": metric_diffs,
        "permutation_results": permutation_results,
    }
    result["passed"] = result["passed"] and all(
        values["loss_diff"] == 0.0 and values["max_metric_diff"] == 0.0
        for values in permutation_results.values()
    )
    output = PROJECT / "results/pcsc_factorial_equivalence/role_vs_positional_semfirst.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
