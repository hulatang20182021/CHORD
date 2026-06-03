#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


class CFOnlyResidualVQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        codebook_size: int = 256,
        num_quantizers: int = 3,
        cf_dim: int = 128,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02) for _ in range(num_quantizers)])
        self.q_proj = nn.Linear(latent_dim, proj_dim)
        self.cf_proj = nn.Linear(cf_dim, proj_dim)
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

    @torch.no_grad()
    def init_codebooks(self, data: torch.Tensor) -> None:
        encoded = self.encoder(data).detach().cpu().numpy()
        residual = encoded.copy()
        for codebook in self.codebooks:
            km = KMeans(n_clusters=self.codebook_size, random_state=2024, n_init=10, max_iter=100).fit(residual)
            centers = km.cluster_centers_.astype(np.float32)
            codebook.copy_(torch.from_numpy(centers).to(codebook.device))
            residual = residual - centers[km.labels_]

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        losses, indices = [], []
        for codebook in self.codebooks:
            dist = torch.cdist(residual, codebook)
            idx = torch.argmin(dist, dim=1)
            q = codebook[idx]
            losses.append(F.mse_loss(q, residual.detach()) + 0.25 * F.mse_loss(q.detach(), residual))
            quantized = quantized + q
            residual = residual - q
            indices.append(idx)
        quantized_st = z + (quantized - z).detach()
        return quantized_st, torch.stack(losses).mean(), torch.stack(indices, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        q, vq_loss, indices = self.quantize(z)
        return self.decoder(q), vq_loss, indices, q

    def cf_logits(self, q: torch.Tensor, cf: torch.Tensor, temperature: float) -> torch.Tensor:
        q_emb = F.normalize(self.q_proj(q), dim=1)
        cf_emb = F.normalize(self.cf_proj(cf), dim=1)
        return q_emb @ cf_emb.T / temperature


def symmetric_infonce(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cf_input", required=True)
    parser.add_argument("--text_item_order", required=True)
    parser.add_argument("--cf_item_order", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_cf_global", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty checkpoint dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    text_order = json.loads(Path(args.text_item_order).read_text(encoding="utf-8"))
    cf_order = json.loads(Path(args.cf_item_order).read_text(encoding="utf-8"))
    if [str(x) for x in text_order] != [str(x) for x in cf_order]:
        raise SystemExit("Text and CF item orders are not identical.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x_np = np.load(args.input).astype(np.float32)
    cf_np = np.load(args.cf_input).astype(np.float32)
    if x_np.shape[0] != cf_np.shape[0]:
        raise SystemExit(f"Input item count mismatch: text={x_np.shape}, cf={cf_np.shape}")
    if not np.isfinite(x_np).all() or not np.isfinite(cf_np).all():
        raise SystemExit("Input contains NaN or inf.")

    x = torch.from_numpy(x_np)
    cf = torch.from_numpy(cf_np)
    device = torch.device(args.device)
    model = CFOnlyResidualVQVAE(input_dim=x_np.shape[1], cf_dim=cf_np.shape[1]).to(device)
    model.init_codebooks(x.to(device))
    loader = DataLoader(TensorDataset(x, cf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "vq_loss": 0.0, "cf_loss": 0.0}
        for batch_x, batch_cf in loader:
            batch_x = batch_x.to(device)
            batch_cf = batch_cf.to(device)
            opt.zero_grad()
            recon, vq_loss, _, q = model(batch_x)
            recon_loss = F.mse_loss(recon, batch_x)
            cf_loss = symmetric_infonce(model.cf_logits(q, batch_cf, args.temperature))
            loss = recon_loss + vq_loss + args.lambda_cf_global * cf_loss
            loss.backward()
            opt.step()
            bs = len(batch_x)
            totals["loss"] += loss.item() * bs
            totals["reconstruction_loss"] += recon_loss.item() * bs
            totals["vq_loss"] += vq_loss.item() * bs
            totals["cf_loss"] += cf_loss.item() * bs
        row = {"epoch": epoch + 1, **{k: v / len(x) for k, v in totals.items()}}
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
                        "proj_dim": 128,
                    },
                    "epoch": epoch + 1,
                    "best_loss": best_loss,
                    "train_args": vars(args),
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
                "proj_dim": 128,
            },
            "epoch": args.epochs,
            "best_loss": best_loss,
            "train_args": vars(args),
        },
        out / "final_model.pt",
    )
    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    model.eval()
    with torch.no_grad():
        _, _, indices, _ = model(x.to(device))
    usage = [{"level": i + 1, "used_codes": int(torch.unique(indices[:, i]).numel()), "total_codes": 256} for i in range(indices.shape[1])]
    summary = {
        "final_loss": history[-1]["loss"],
        "final_reconstruction_loss": history[-1]["reconstruction_loss"],
        "final_vq_loss": history[-1]["vq_loss"],
        "final_cf_loss": history[-1]["cf_loss"],
        "best_loss": best_loss,
        "epochs": args.epochs,
        "lambda_cf_global": args.lambda_cf_global,
        "temperature": args.temperature,
        "codebook_usage": usage,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
