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


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


def cosine_distance(x, y):
    return (1.0 - F.cosine_similarity(x, y, dim=-1)).mean()


def subspace_diff_loss(shared, private):
    shared = shared - shared.mean(dim=0, keepdim=True)
    private = private - private.mean(dim=0, keepdim=True)
    shared = F.normalize(shared, dim=0, eps=1e-8)
    private = F.normalize(private, dim=0, eps=1e-8)
    return torch.mean((shared.T @ private) ** 2)


def variance_loss(z, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


class BiViewDSNLossTokenizer(nn.Module):
    def __init__(self, input_dim_sem=768, input_dim_cf=128, latent_dim=64, codebook_size=256):
        super().__init__()
        self.sem_shared_encoder = mlp(input_dim_sem, 256, latent_dim)
        self.cf_shared_encoder = mlp(input_dim_cf, 128, latent_dim)
        self.sem_private_encoder = mlp(input_dim_sem, 256, latent_dim)
        self.cf_private_encoder = mlp(input_dim_cf, 128, latent_dim)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
            nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.02),
        ])
        self.cf_decoder = mlp(latent_dim, 128, input_dim_cf)
        self.sem_decoder = mlp(latent_dim, 256, input_dim_sem)
        self.shared_cf_head = nn.Linear(latent_dim, input_dim_cf)
        self.shared_sem_head = nn.Linear(latent_dim, input_dim_sem)
        self.cfres_head = nn.Linear(latent_dim, input_dim_cf)
        self.semres_head = nn.Linear(latent_dim, input_dim_sem)

    def _vq(self, z, level):
        cb = self.codebooks[level]
        ids = torch.cdist(z, cb).argmin(dim=1)
        q = F.embedding(ids, cb)
        vq = F.mse_loss(q.detach(), z) + 0.25 * F.mse_loss(q, z.detach())
        q_st = z + (q - z).detach()
        return q_st, ids, vq

    def encode(self, sem, cf):
        z_sem_shared = self.sem_shared_encoder(sem)
        z_cf_shared = self.cf_shared_encoder(cf)
        z_sem_private = self.sem_private_encoder(sem)
        z_cf_private = self.cf_private_encoder(cf)
        z_shared = 0.5 * (z_sem_shared + z_cf_shared)
        q1, c1, vq1 = self._vq(z_shared, 0)
        q2, c2, vq2 = self._vq(z_cf_private, 1)
        q3, c3, vq3 = self._vq(z_sem_private, 2)
        return {
            "z_sem_shared": z_sem_shared,
            "z_cf_shared": z_cf_shared,
            "z_sem_private": z_sem_private,
            "z_cf_private": z_cf_private,
            "z_shared": z_shared,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "vq": vq1 + vq2 + vq3,
        }

    def forward(self, sem, cf):
        out = self.encode(sem, cf)
        q1, q2, q3 = out["q1"], out["q2"], out["q3"]
        cf_hat = self.cf_decoder(q1 + q2)
        sem_hat = self.sem_decoder(q1 + q3)
        q1_delta = q1.detach()
        cf_shared_for_delta = self.cf_decoder(q1_delta).detach()
        sem_shared_for_delta = self.sem_decoder(q1_delta).detach()
        cf_full_for_delta = self.cf_decoder(q1_delta + q2)
        sem_full_for_delta = self.sem_decoder(q1_delta + q3)
        out.update({
            "cf_hat": cf_hat,
            "sem_hat": sem_hat,
            "cf_shared_hat": self.cf_decoder(q1),
            "sem_shared_hat": self.sem_decoder(q1),
            "cf_delta_hat": cf_full_for_delta - cf_shared_for_delta,
            "sem_delta_hat": sem_full_for_delta - sem_shared_for_delta,
            "shared_cf": self.shared_cf_head(q1),
            "shared_sem": self.shared_sem_head(q1),
            "cfres": self.cfres_head(q2),
            "semres": self.semres_head(q3),
        })
        return out


@torch.no_grad()
def init_codebooks(model, st5, cf, codebook_size, seed, device):
    model.eval()
    z_sem_shared, z_cf_shared, z_sem_private, z_cf_private = [], [], [], []
    for start in range(0, len(st5), 2048):
        sem = torch.from_numpy(st5[start:start + 2048]).to(device)
        cft = torch.from_numpy(cf[start:start + 2048]).to(device)
        z_sem_shared.append(model.sem_shared_encoder(sem).cpu().numpy())
        z_cf_shared.append(model.cf_shared_encoder(cft).cpu().numpy())
        z_sem_private.append(model.sem_private_encoder(sem).cpu().numpy())
        z_cf_private.append(model.cf_private_encoder(cft).cpu().numpy())
    z_sem_shared = np.concatenate(z_sem_shared)
    z_cf_shared = np.concatenate(z_cf_shared)
    targets = [
        0.5 * (z_sem_shared + z_cf_shared),
        np.concatenate(z_cf_private),
        np.concatenate(z_sem_private),
    ]
    for level, target in enumerate(targets):
        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            random_state=seed,
            batch_size=2048,
            n_init=10,
        )
        km.fit(target)
        model.codebooks[level].copy_(torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device))


@torch.no_grad()
def full_usage(model, st5, cf, device, batch_size=2048):
    model.eval()
    rows = [[], [], []]
    for start in range(0, len(st5), batch_size):
        sem = torch.from_numpy(st5[start:start + batch_size]).to(device)
        cft = torch.from_numpy(cf[start:start + batch_size]).to(device)
        out = model(sem, cft)
        for idx, key in enumerate(("c1", "c2", "c3")):
            rows[idx].append(out[key].cpu().numpy())
    return [int(len(np.unique(np.concatenate(x)))) for x in rows]


def save_checkpoint(path, model, config, epoch, loss):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    state["codebooks.0"] = model.codebooks[0].detach().cpu()
    state["codebooks.1"] = model.codebooks[1].detach().cpu()
    state["codebooks.2"] = model.codebooks[2].detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": state,
        "config": config,
        "epoch": epoch,
        "best_loss": loss,
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
    p.add_argument("--lambda_similarity", type=float, default=0.05)
    p.add_argument("--lambda_difference", type=float, default=0.01)
    p.add_argument("--lambda_shared", type=float, default=0.01)
    p.add_argument("--lambda_cf_private", type=float, default=0.05)
    p.add_argument("--lambda_sem_private", type=float, default=0.20)
    p.add_argument("--lambda_cf_delta", type=float, default=0.05)
    p.add_argument("--lambda_sem_delta", type=float, default=0.20)
    p.add_argument("--lambda_orth", type=float, default=0.001)
    p.add_argument("--lambda_var", type=float, default=0.001)
    p.add_argument("--difference_warmup_epochs", type=int, default=10)
    p.add_argument("--min_diff_batch_size", type=int, default=128)
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
    if len({len(x) for x in arrays}) != 1:
        raise ValueError("Input length mismatch")

    model = BiViewDSNLossTokenizer(st5.shape[1], cf.shape[1], args.latent_dim, args.codebook_size).to(device)
    init_codebooks(model, st5, cf, args.codebook_size, args.seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = TensorDataset(*(torch.from_numpy(x) for x in arrays))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))

    config = {
        "method": "biview_dsnloss_tokenizer",
        "variant": "biview_sp_dsnloss_v1",
        "input_dim_sem": int(st5.shape[1]),
        "input_dim_cf": int(cf.shape[1]),
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "num_quantizers": 3,
        "dataset": args.dataset,
        "seed": args.seed,
        "epochs": args.epochs,
        "lambda_cf_rec": args.lambda_cf_rec,
        "lambda_sem_rec": args.lambda_sem_rec,
        "lambda_similarity": args.lambda_similarity,
        "lambda_difference": args.lambda_difference,
        "lambda_shared": args.lambda_shared,
        "lambda_cf_private": args.lambda_cf_private,
        "lambda_sem_private": args.lambda_sem_private,
        "lambda_cf_delta": args.lambda_cf_delta,
        "lambda_sem_delta": args.lambda_sem_delta,
        "lambda_orth": args.lambda_orth,
        "lambda_var": args.lambda_var,
        "difference_warmup_epochs": args.difference_warmup_epochs,
        "min_diff_batch_size": args.min_diff_batch_size,
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(config, output / "config.json")

    best_valid_loss = float("inf")
    best_any_loss = float("inf")
    best_valid_epoch = None
    best_any_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        sums, seen = {}, 0
        model.train()
        difference_factor = min(1.0, epoch / max(args.difference_warmup_epochs, 1))
        lambda_difference_eff = args.lambda_difference * difference_factor
        for batch in loader:
            sem, cft, cf_b, cf_r, sem_b, sem_r = [x.to(device) for x in batch]
            out = model(sem, cft)
            cf_rec = F.mse_loss(out["cf_hat"], cft) + 0.1 * cosine_distance(out["cf_hat"], cft)
            sem_rec = F.mse_loss(out["sem_hat"], sem)
            similarity = cosine_distance(out["z_sem_shared"], out["z_cf_shared"])
            if len(sem) >= args.min_diff_batch_size:
                difference = (
                    subspace_diff_loss(out["z_sem_shared"], out["z_sem_private"])
                    + subspace_diff_loss(out["z_cf_shared"], out["z_cf_private"])
                )
            else:
                difference = torch.zeros((), device=device)
            shared = cosine_distance(out["shared_cf"], cf_b) + F.mse_loss(out["shared_sem"], sem_b)
            cf_private = cosine_distance(out["cfres"], cf_r)
            sem_private = F.mse_loss(out["semres"], sem_r)
            cf_delta = F.mse_loss(out["cf_delta_hat"], cf_r)
            sem_delta = F.mse_loss(out["sem_delta_hat"], sem_r)
            orth = (
                F.cosine_similarity(out["q1"], out["q2"], dim=1).pow(2).mean()
                + F.cosine_similarity(out["q1"], out["q3"], dim=1).pow(2).mean()
                + F.cosine_similarity(out["q2"], out["q3"], dim=1).pow(2).mean()
            )
            var = variance_loss(out["z_sem_shared"]) + variance_loss(out["z_cf_shared"])
            loss = (
                args.lambda_cf_rec * cf_rec
                + args.lambda_sem_rec * sem_rec
                + out["vq"]
                + args.lambda_similarity * similarity
                + lambda_difference_eff * difference
                + args.lambda_shared * shared
                + args.lambda_cf_private * cf_private
                + args.lambda_sem_private * sem_private
                + args.lambda_cf_delta * cf_delta
                + args.lambda_sem_delta * sem_delta
                + args.lambda_orth * orth
                + args.lambda_var * var
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = len(sem)
            seen += bs
            values = {
                "loss": loss,
                "cf_rec": cf_rec,
                "sem_rec": sem_rec,
                "similarity": similarity,
                "difference": difference,
                "shared": shared,
                "cf_private": cf_private,
                "sem_private": sem_private,
                "cf_delta": cf_delta,
                "sem_delta": sem_delta,
                "orth": orth,
                "var_loss": var,
                "vq": out["vq"],
                "z_sem_shared_std": out["z_sem_shared"].std(dim=0).mean(),
                "z_cf_shared_std": out["z_cf_shared"].std(dim=0).mean(),
            }
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * bs

        c1_unique, c2_unique, c3_unique = full_usage(model, st5, cf, device)
        row = {key: value / seen for key, value in sums.items()}
        row.update({
            "epoch": epoch,
            "lambda_difference_eff": lambda_difference_eff,
            "lambda_similarity_eff": args.lambda_similarity,
            "c1_unique": c1_unique,
            "c2_unique": c2_unique,
            "c3_unique": c3_unique,
        })
        row["structure_valid"] = c1_unique >= 60 and c2_unique >= 180 and c3_unique >= 180
        history.append(row)
        print(json.dumps(row), flush=True)
        with (output / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if row["loss"] < best_any_loss:
            best_any_loss = row["loss"]
            best_any_epoch = epoch
            save_checkpoint(output / "best_unconstrained_model.pt", model, config, epoch, best_any_loss)
        if row["structure_valid"] and row["loss"] < best_valid_loss:
            best_valid_loss = row["loss"]
            best_valid_epoch = epoch
            save_checkpoint(output / "best_model.pt", model, config, epoch, best_valid_loss)

    save_checkpoint(output / "final_model.pt", model, config, args.epochs, history[-1]["loss"])
    save_json({
        "history": history,
        "best_structure_valid_loss": best_valid_loss if best_valid_loss < float("inf") else None,
        "best_structure_valid_epoch": best_valid_epoch,
        "best_unconstrained_loss": best_any_loss,
        "best_unconstrained_epoch": best_any_epoch,
        "final_epoch": args.epochs,
        "has_structure_valid_epoch": best_valid_loss < float("inf"),
    }, output / "training_summary.json")
    if not (output / "best_model.pt").exists():
        print("WARNING: no structure-valid epoch; best_model.pt not written")


if __name__ == "__main__":
    main()
