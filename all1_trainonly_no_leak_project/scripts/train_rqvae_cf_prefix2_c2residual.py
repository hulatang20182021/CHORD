#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


def load_order(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        for key in ("item_id_order", "item_order", "order"):
            if isinstance(raw.get(key), list):
                return [str(x) for x in raw[key]]
    raise SystemExit(f"Unsupported item order format: {path}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CFPrefix2C2ResidualVQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        cf_dim: int = 128,
        residual_dim: int = 128,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02) for _ in range(num_quantizers)]
        )
        self.q_proj = nn.Linear(latent_dim, proj_dim)
        self.cf_proj = nn.Linear(cf_dim, proj_dim)
        self.q2_residual_proj = nn.Linear(latent_dim, proj_dim)
        self.residual_proj = nn.Linear(residual_dim, proj_dim)
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor, seed: int) -> None:
        encoded = self.encoder(data).detach().cpu().numpy()
        residual = encoded.copy()
        for codebook in self.codebooks:
            km = KMeans(n_clusters=self.codebook_size, random_state=seed, n_init=10, max_iter=100).fit(residual)
            centers = km.cluster_centers_.astype(np.float32)
            codebook.copy_(torch.from_numpy(centers).to(codebook.device))
            residual = residual - centers[km.labels_]

    def quantize(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        h2_quantized = torch.zeros_like(z)
        losses = []
        indices = []
        q_vectors = []
        for level, codebook in enumerate(self.codebooks):
            dist = torch.cdist(residual, codebook)
            idx = torch.argmin(dist, dim=1)
            q = codebook[idx]
            losses.append(F.mse_loss(q, residual.detach()) + 0.25 * F.mse_loss(q.detach(), residual))
            quantized = quantized + q
            if level < 2:
                h2_quantized = h2_quantized + q
            residual = residual - q
            indices.append(idx)
            q_vectors.append(q)
        full_st = z + (quantized - z).detach()
        h2_st = z + (h2_quantized - z).detach()
        return (
            full_st,
            h2_st,
            torch.stack(losses).mean(),
            torch.stack(indices, dim=1),
            q_vectors[0],
            q_vectors[1],
            q_vectors[2],
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        q_full, h2, vq_loss, indices, q1, q2, q3 = self.quantize(z)
        return self.decoder(q_full), vq_loss, indices, h2, q1, q2, q3

    def cf_logits(self, h2: torch.Tensor, cf: torch.Tensor, temperature: float) -> torch.Tensor:
        q_emb = F.normalize(self.q_proj(h2), dim=1)
        cf_emb = F.normalize(self.cf_proj(cf), dim=1)
        return q_emb @ cf_emb.T / temperature

    def residual_logits(self, q2: torch.Tensor, residual: torch.Tensor, temperature: float) -> torch.Tensor:
        q2_emb = F.normalize(self.q2_residual_proj(q2), dim=1)
        r_emb = F.normalize(self.residual_proj(residual), dim=1)
        return q2_emb @ r_emb.T / temperature


def symmetric_infonce(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def code_usage(model: CFPrefix2C2ResidualVQVAE, x: torch.Tensor, device: torch.device, batch_size: int) -> list[dict[str, int]]:
    model.eval()
    all_idx = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].to(device)
        _, _, indices, _, _, _, _ = model(batch)
        all_idx.append(indices.cpu())
    indices = torch.cat(all_idx, dim=0)
    return [
        {"level": i + 1, "used_codes": int(torch.unique(indices[:, i]).numel()), "total_codes": model.codebook_size}
        for i in range(indices.shape[1])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--st5_emb", "--input", dest="st5_emb", required=True)
    parser.add_argument("--cf_emb", "--cf_input", dest="cf_emb", required=True)
    parser.add_argument("--residual_emb", required=True)
    parser.add_argument("--item_order", "--text_item_order", dest="item_order", required=True)
    parser.add_argument("--cf_order", required=True)
    parser.add_argument("--output_dir", "--out_dir", dest="output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_cf", "--lambda_cf_prefix2", dest="lambda_cf", type=float, default=0.0005)
    parser.add_argument("--lambda_res", type=float, required=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--residual_temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_q2_residual_info_nce", action="store_true")
    parser.add_argument("--mask_zero_residual", action="store_true")
    args = parser.parse_args()

    if not args.use_q2_residual_info_nce:
        raise SystemExit("This script is only for --use_q2_residual_info_nce.")
    if args.mask_zero_residual:
        raise SystemExit("mask_zero_residual is intentionally unsupported in this run; use false.")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty checkpoint dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    text_order = load_order(Path(args.item_order))
    cf_order = load_order(Path(args.cf_order))
    if text_order != cf_order:
        raise SystemExit("ST5 and CF item orders are not identical.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    x_np = np.load(args.st5_emb).astype(np.float32)
    cf_np = np.load(args.cf_emb).astype(np.float32)
    res_np = np.load(args.residual_emb).astype(np.float32)
    if x_np.shape[0] != cf_np.shape[0] or cf_np.shape != res_np.shape or x_np.shape[0] != len(text_order):
        raise SystemExit(
            f"Input shape/order mismatch: st5={x_np.shape}, cf={cf_np.shape}, residual={res_np.shape}, order={len(text_order)}"
        )
    for name, arr in (("st5", x_np), ("cf", cf_np), ("residual", res_np)):
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name} input contains NaN or inf.")

    residual_norms = np.linalg.norm(res_np, axis=1)
    x = torch.from_numpy(x_np)
    cf = torch.from_numpy(cf_np)
    residual = torch.from_numpy(res_np)
    device = torch.device(args.device)
    model = CFPrefix2C2ResidualVQVAE(input_dim=x_np.shape[1], cf_dim=cf_np.shape[1], residual_dim=res_np.shape[1]).to(device)
    model.init_codebooks(x.to(device), seed=args.seed)
    loader = DataLoader(TensorDataset(x, cf, residual), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        totals = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "vq_loss": 0.0,
            "commit_loss": 0.0,
            "cf_prefix2_loss": 0.0,
            "q2_residual_loss": 0.0,
        }
        for batch_x, batch_cf, batch_res in loader:
            batch_x = batch_x.to(device)
            batch_cf = batch_cf.to(device)
            batch_res = batch_res.to(device)
            opt.zero_grad()
            recon, vq_loss, _, h2, _, q2, _ = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)
            cf_loss = symmetric_infonce(model.cf_logits(h2, batch_cf, args.temperature))
            res_loss = symmetric_infonce(model.residual_logits(q2, batch_res, args.residual_temperature))
            loss = recon_loss + vq_loss + args.lambda_cf * cf_loss + args.lambda_res * res_loss
            loss.backward()
            opt.step()
            bs = len(batch_x)
            totals["loss"] += loss.item() * bs
            totals["reconstruction_loss"] += recon_loss.item() * bs
            totals["vq_loss"] += vq_loss.item() * bs
            totals["commit_loss"] += vq_loss.item() * bs
            totals["cf_prefix2_loss"] += cf_loss.item() * bs
            totals["q2_residual_loss"] += res_loss.item() * bs
        row = {"epoch": epoch + 1, **{k: v / len(x) for k, v in totals.items()}}
        row["codebook_usage"] = code_usage(model, x, device, args.batch_size)
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": x_np.shape[1],
                        "latent_dim": 64,
                        "codebook_size": 256,
                        "num_quantizers": 3,
                        "cf_dim": cf_np.shape[1],
                        "residual_dim": res_np.shape[1],
                        "proj_dim": 128,
                    },
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "train_args": vars(args),
                    "method": "cf_prefix2_c2residual_st5_rqvae",
                    "cf_alignment_target": "h2=q1+q2",
                    "residual_alignment_target": "q2 only",
                },
                out / "best_model.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": x_np.shape[1],
                "latent_dim": 64,
                "codebook_size": 256,
                "num_quantizers": 3,
                "cf_dim": cf_np.shape[1],
                "residual_dim": res_np.shape[1],
                "proj_dim": 128,
            },
            "epoch": args.epochs,
            "best_loss": best_loss,
            "train_args": vars(args),
            "method": "cf_prefix2_c2residual_st5_rqvae",
            "cf_alignment_target": "h2=q1+q2",
            "residual_alignment_target": "q2 only",
        },
        out / "final_model.pt",
    )
    save_json(history, out / "training_history.json")

    final = history[-1]
    summary = {
        "method": "cf_prefix2_c2residual_st5_rqvae",
        "dataset": args.dataset,
        "cf_alignment_target": "h2=q1+q2",
        "residual_alignment_target": "q2 only",
        "final_loss": final["loss"],
        "final_reconstruction_loss": final["reconstruction_loss"],
        "final_vq_loss": final["vq_loss"],
        "final_commit_loss": final["commit_loss"],
        "final_cf_prefix2_loss": final["cf_prefix2_loss"],
        "final_q2_residual_loss": final["q2_residual_loss"],
        "best_loss": best_loss,
        "epochs": args.epochs,
        "seed": args.seed,
        "lambda_cf": args.lambda_cf,
        "lambda_res": args.lambda_res,
        "temperature": args.temperature,
        "residual_temperature": args.residual_temperature,
        "st5_embedding_path": args.st5_emb,
        "cf_embedding_path": args.cf_emb,
        "residual_embedding_path": args.residual_emb,
        "residual_norm_min": float(residual_norms.min()),
        "residual_norm_mean": float(residual_norms.mean()),
        "residual_norm_median": float(np.median(residual_norms)),
        "residual_norm_max": float(residual_norms.max()),
        "codebook_usage": final["codebook_usage"],
    }
    save_json(summary, out / "training_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
