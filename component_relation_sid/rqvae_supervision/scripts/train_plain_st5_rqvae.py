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


class PlainResidualVQVAE(nn.Module):
    def __init__(self, input_dim: int = 768, latent_dim: int = 64, codebook_size: int = 256, num_quantizers: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02) for _ in range(num_quantizers)])
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        q, vq_loss, indices = self.quantize(z)
        return self.decoder(q), vq_loss, indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty checkpoint dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x_np = np.load(args.input).astype(np.float32)
    tensor = torch.from_numpy(x_np)
    device = torch.device(args.device)
    model = PlainResidualVQVAE(input_dim=x_np.shape[1]).to(device)
    model.init_codebooks(tensor.to(device))
    loader = DataLoader(TensorDataset(tensor), batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = recon_total = vq_total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, vq_loss, _ = model(batch)
            recon_loss = F.mse_loss(recon, batch)
            loss = recon_loss + vq_loss
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            recon_total += recon_loss.item() * len(batch)
            vq_total += vq_loss.item() * len(batch)
        row = {"epoch": epoch + 1, "loss": total / len(tensor), "reconstruction_loss": recon_total / len(tensor), "vq_loss": vq_total / len(tensor)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save({"model_state_dict": model.state_dict(), "config": {"input_dim": x_np.shape[1], "latent_dim": 64, "codebook_size": 256, "num_quantizers": 3}, "epoch": epoch + 1, "best_loss": best_loss}, out / "best_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": {"input_dim": x_np.shape[1], "latent_dim": 64, "codebook_size": 256, "num_quantizers": 3}, "epoch": args.epochs, "best_loss": best_loss}, out / "final_model.pt")
    (out / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    usage = []
    model.eval()
    with torch.no_grad():
        _, _, indices = model(tensor.to(device))
    for level in range(indices.shape[1]):
        usage.append({"level": level + 1, "used_codes": int(torch.unique(indices[:, level]).numel()), "total_codes": 256})
    summary = {"final_loss": history[-1]["loss"], "final_reconstruction_loss": history[-1]["reconstruction_loss"], "best_loss": best_loss, "epochs": args.epochs, "codebook_usage": usage}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
