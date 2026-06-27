#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

from project_paths import assert_new_base_only, save_json


def cosine_distance(x, y):
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean()


class BiViewSharedPrivateTokenizer(nn.Module):
    def __init__(self, input_dim_sem=768, input_dim_cf=128, latent_dim=64, codebook_size=256):
        super().__init__()
        self.sem_encoder = nn.Sequential(nn.Linear(input_dim_sem, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.cf_encoder = nn.Sequential(nn.Linear(input_dim_cf, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
        ])
        self.cf_decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim_cf))
        self.sem_decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim_sem))
        self.shared_cf_head = nn.Linear(latent_dim, input_dim_cf)
        self.shared_sem_head = nn.Linear(latent_dim, input_dim_sem)
        self.cfres_head = nn.Linear(latent_dim, input_dim_cf)
        self.semres_head = nn.Linear(latent_dim, input_dim_sem)

    def _vq(self, z, level):
        cb = self.codebooks[level]
        dist = torch.cdist(z, cb)
        ids = dist.argmin(dim=1)
        q = F.embedding(ids, cb)
        vq = F.mse_loss(q.detach(), z) + 0.25 * F.mse_loss(q, z.detach())
        q_st = z + (q - z).detach()
        return q_st, ids, vq

    def encode(self, sem, cf):
        z_sem = self.sem_encoder(sem)
        z_cf = self.cf_encoder(cf)
        z_shared = 0.5 * (z_sem + z_cf)
        q1, c1, vq1 = self._vq(z_shared, 0)
        q2, c2, vq2 = self._vq(z_cf - q1.detach(), 1)
        q3, c3, vq3 = self._vq(z_sem - q1.detach(), 2)
        return z_sem, z_cf, q1, q2, q3, c1, c2, c3, vq1 + vq2 + vq3

    def forward(self, sem, cf):
        z_sem, z_cf, q1, q2, q3, c1, c2, c3, vq = self.encode(sem, cf)
        cf_hat = self.cf_decoder(q1 + q2)
        sem_hat = self.sem_decoder(q1 + q3)
        return {
            "z_sem": z_sem, "z_cf": z_cf, "q1": q1, "q2": q2, "q3": q3,
            "c1": c1, "c2": c2, "c3": c3, "vq": vq,
            "cf_hat": cf_hat, "sem_hat": sem_hat,
            "shared_cf": self.shared_cf_head(q1),
            "shared_sem": self.shared_sem_head(q1),
            "cfres": self.cfres_head(q2),
            "semres": self.semres_head(q3),
        }


def init_codebooks(model, st5, cf, codebook_size, seed, device):
    model.eval()
    with torch.no_grad():
        sem = torch.from_numpy(st5).to(device)
        cft = torch.from_numpy(cf).to(device)
        z_sem = model.sem_encoder(sem).cpu().numpy()
        z_cf = model.cf_encoder(cft).cpu().numpy()
    z_shared = 0.5 * (z_sem + z_cf)
    targets = [z_shared, z_cf - z_shared, z_sem - z_shared]
    for i, target in enumerate(targets):
        km = MiniBatchKMeans(n_clusters=codebook_size, random_state=seed, batch_size=2048, n_init=10)
        km.fit(target)
        with torch.no_grad():
            model.codebooks[i].copy_(torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device))


def save_checkpoint(path, model, config, epoch, best_loss):
    state = model.state_dict()
    ckpt_state = {k: v.detach().cpu() for k, v in state.items()}
    # MatchedCurriculumLETTER expects these exact keys.
    ckpt_state["codebooks.0"] = model.codebooks[0].detach().cpu()
    ckpt_state["codebooks.1"] = model.codebooks[1].detach().cpu()
    ckpt_state["codebooks.2"] = model.codebooks[2].detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": ckpt_state,
        "config": config,
        "epoch": epoch,
        "best_loss": best_loss,
    }, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--st5_emb", required=True)
    p.add_argument("--cf_emb", required=True)
    p.add_argument("--cf_base_emb", required=True)
    p.add_argument("--cf_residual_emb", required=True)
    p.add_argument("--sem_base_emb", required=True)
    p.add_argument("--sem_residual_emb", required=True)
    p.add_argument("--item_order", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--lambda_cf_rec", type=float, default=1.0)
    p.add_argument("--lambda_sem_rec", type=float, default=1.0)
    p.add_argument("--lambda_shared", type=float, default=0.01)
    p.add_argument("--lambda_cf_private", type=float, default=0.05)
    p.add_argument("--lambda_sem_private", type=float, default=0.05)
    p.add_argument("--lambda_orth", type=float, default=0.001)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    output = Path(args.output_dir)
    assert_new_base_only([output])
    if (output / "best_model.pt").exists():
        print(f"SKIP existing tokenizer: {output / 'best_model.pt'}")
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty incomplete output_dir: {output}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    st5 = np.load(args.st5_emb).astype(np.float32)
    cf = np.load(args.cf_emb).astype(np.float32)
    cf_base = np.load(args.cf_base_emb).astype(np.float32)
    cf_res = np.load(args.cf_residual_emb).astype(np.float32)
    sem_base = np.load(args.sem_base_emb).astype(np.float32)
    sem_res = np.load(args.sem_residual_emb).astype(np.float32)
    arrays = [st5, cf, cf_base, cf_res, sem_base, sem_res]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Non-finite input")

    model = BiViewSharedPrivateTokenizer(st5.shape[1], cf.shape[1], args.latent_dim, args.codebook_size).to(device)
    init_codebooks(model, st5, cf, args.codebook_size, args.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = TensorDataset(*(torch.from_numpy(x) for x in arrays))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))

    config = vars(args).copy()
    config.update({
        "method": "biview_shared_private_tokenizer",
        "input_dim_sem": int(st5.shape[1]),
        "input_dim_cf": int(cf.shape[1]),
        "num_quantizers": 3,
    })
    output.mkdir(parents=True, exist_ok=True)
    save_json(config, output / "config.json")
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        sums, seen = {}, 0
        model.train()
        all_c = [[], [], []]
        for batch in loader:
            sem, cft, cf_b, cf_r, sem_b, sem_r = [x.to(device) for x in batch]
            out = model(sem, cft)
            cf_rec = F.mse_loss(out["cf_hat"], cft) + 0.1 * cosine_distance(out["cf_hat"], cft)
            sem_rec = F.mse_loss(out["sem_hat"], sem)
            shared = cosine_distance(out["shared_cf"], cf_b) + F.mse_loss(out["shared_sem"], sem_b)
            cf_private = cosine_distance(out["cfres"], cf_r)
            sem_private = F.mse_loss(out["semres"], sem_r)
            orth = (
                F.cosine_similarity(out["q1"], out["q2"], dim=1).pow(2).mean()
                + F.cosine_similarity(out["q1"], out["q3"], dim=1).pow(2).mean()
                + F.cosine_similarity(out["q2"], out["q3"], dim=1).pow(2).mean()
            )
            loss = (
                args.lambda_cf_rec * cf_rec
                + args.lambda_sem_rec * sem_rec
                + out["vq"]
                + args.lambda_shared * shared
                + args.lambda_cf_private * cf_private
                + args.lambda_sem_private * sem_private
                + args.lambda_orth * orth
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = len(sem)
            seen += bs
            values = {
                "loss": loss, "cf_rec": cf_rec, "sem_rec": sem_rec,
                "shared": shared, "cf_private": cf_private,
                "sem_private": sem_private, "orth": orth, "vq": out["vq"],
            }
            for k, v in values.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * bs
            for i, key in enumerate(["c1", "c2", "c3"]):
                all_c[i].append(out[key].detach().cpu().numpy())
        row = {k: v / seen for k, v in sums.items()}
        codes = [np.concatenate(x) for x in all_c]
        row.update({
            "epoch": epoch,
            "c1_unique": int(len(np.unique(codes[0]))),
            "c2_unique": int(len(np.unique(codes[1]))),
            "c3_unique": int(len(np.unique(codes[2]))),
        })
        row["structure_valid"] = row["c1_unique"] >= 60 and row["c2_unique"] >= 180 and row["c3_unique"] >= 180
        history.append(row)
        print(json.dumps(row), flush=True)
        with (output / "training_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row["structure_valid"] and row["loss"] < best_loss:
            best_loss = row["loss"]
            save_checkpoint(output / "best_model.pt", model, config, epoch, best_loss)
        save_checkpoint(output / "best_unconstrained_model.pt", model, config, epoch, row["loss"])
    save_checkpoint(output / "final_model.pt", model, config, args.epochs, history[-1]["loss"])
    save_json({"history": history, "best_loss": best_loss, "has_structure_valid_epoch": best_loss < float("inf")}, output / "training_summary.json")
    if not (output / "best_model.pt").exists():
        print("WARNING: no structure-valid epoch; best_model.pt not written")


if __name__ == "__main__":
    main()
