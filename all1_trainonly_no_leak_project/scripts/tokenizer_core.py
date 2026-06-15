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


def save_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Model(nn.Module):
    def __init__(self, input_dim=768, cf_dim=128, latent_dim=64, codebook_size=256, num_quantizers=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, input_dim))
        self.codebooks = nn.ParameterList([nn.Parameter(torch.randn(codebook_size, latent_dim) * .02) for _ in range(num_quantizers)])
        self.cf12_proj = nn.Linear(latent_dim, cf_dim)
        self.cf_target_proj = nn.Linear(cf_dim, cf_dim)
        self.cf_base = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, cf_dim))
        self.cfres2_proj = nn.Linear(latent_dim, cf_dim)
        self.cfres_target_proj = nn.Linear(cf_dim, cf_dim)
        self.sem_base = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, input_dim))
        self.semres3_proj = nn.Linear(latent_dim, input_dim)
        self.codebook_size, self.num_quantizers = codebook_size, num_quantizers

    @torch.no_grad()
    def init_codebooks(self, x, seed):
        residual = self.encoder(x).cpu().numpy()
        for cb in self.codebooks:
            km = KMeans(n_clusters=self.codebook_size, random_state=seed, n_init=10, max_iter=100).fit(residual)
            centers = km.cluster_centers_.astype(np.float32)
            cb.copy_(torch.from_numpy(centers).to(cb.device))
            residual -= centers[km.labels_]

    def quantize(self, z):
        residual, total, losses, ids, qs = z, torch.zeros_like(z), [], [], []
        for cb in self.codebooks:
            idx = torch.cdist(residual, cb).argmin(1)
            q = cb[idx]
            losses.append(F.mse_loss(q, residual.detach()) + .25 * F.mse_loss(q.detach(), residual))
            total = total + q
            residual = residual - q
            ids.append(idx)
            qs.append(q)
        return z + (total-z).detach(), torch.stack(losses).mean(), torch.stack(ids, 1), qs

    def forward(self, x):
        z = self.encoder(x)
        q, vq, ids, qs = self.quantize(z)
        return self.decoder(q), vq, ids, qs


def cosine_loss(a, b):
    return (1 - F.cosine_similarity(a, b, dim=1)).mean()

def symmetric_infonce(a, b, temperature=0.1):
    logits = F.normalize(a, dim=1) @ F.normalize(b, dim=1).T / temperature
    labels = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def usage(model, x, device, batch):
    model.eval(); rows = []
    for start in range(0, len(x), batch):
        rows.append(model(x[start:start+batch].to(device))[2].cpu())
    ids = torch.cat(rows)
    return [int(torch.unique(ids[:, i]).numel()) for i in range(3)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Beauty"); p.add_argument("--st5_emb", required=True)
    p.add_argument("--cf_emb", required=True); p.add_argument("--cf_residual_emb", required=True)
    p.add_argument("--sem_residual_emb", required=True); p.add_argument("--item_order", required=True)
    p.add_argument("--output_dir", required=True); p.add_argument("--lambda_cf12", type=float, required=True)
    p.add_argument("--lambda_cfres2", type=float, required=True); p.add_argument("--lambda_semres3", type=float, required=True)
    p.add_argument("--epochs", type=int, default=60); p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--init_checkpoint", default=None)
    args = p.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(path.name not in {"train.log", "cfres_init"} for path in out.iterdir()):
        raise SystemExit(f"Refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    st5 = np.load(args.st5_emb).astype(np.float32); cf = np.load(args.cf_emb).astype(np.float32)
    cf_residual = np.load(args.cf_residual_emb).astype(np.float32)
    sem_residual = np.load(args.sem_residual_emb).astype(np.float32)
    order = [str(x) for x in json.load(open(args.item_order, encoding="utf-8"))]
    if len(order) != len(st5) or len(st5) != len(cf) or len(cf_residual) != len(st5) or len(sem_residual) != len(st5) or not all(np.isfinite(v).all() for v in (st5,cf,cf_residual,sem_residual)):
        raise SystemExit("Input shape/order/finite check failed")
    x, y, yr, sr = map(torch.from_numpy, (st5, cf, cf_residual, sem_residual))
    device = torch.device(args.device)
    model = Model(st5.shape[1], cf.shape[1]).to(device)
    if args.init_checkpoint:
        base = torch.load(args.init_checkpoint, map_location=device, weights_only=False)["model_state_dict"]
        mapped = {}
        for key, value in base.items():
            target = key
            if key.startswith("q_proj."):
                target = key.replace("q_proj.", "cf12_proj.")
            elif key.startswith("cf_proj."):
                target = key.replace("cf_proj.", "cf_target_proj.")
            elif key.startswith("q2_residual_proj."):
                target = key.replace("q2_residual_proj.", "cfres2_proj.")
            elif key.startswith("residual_proj."):
                target = key.replace("residual_proj.", "cfres_target_proj.")
            if target in model.state_dict() and model.state_dict()[target].shape == value.shape:
                mapped[target] = value
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        print(json.dumps({"initialized_from": args.init_checkpoint, "loaded_tensors": len(mapped),
                          "new_parameters": missing, "unexpected": unexpected}), flush=True)
    else:
        model.init_codebooks(x.to(device), args.seed)
    loader = DataLoader(TensorDataset(x, y, yr, sr), args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    config = {"method":"trainonly_cf_prefix_semres_c3","source_cf":"trainonly_cf_svd","no_full_sequence_cf_used":True,"lambda_cf12":args.lambda_cf12,
              "lambda_cfres2":args.lambda_cfres2,"lambda_semres3":args.lambda_semres3,
              "epochs":args.epochs,"seed":args.seed,"dataset":args.dataset,"codebook_size":256,
              "num_quantizers":3,"input_dim":st5.shape[1],"cf_dim":cf.shape[1],"latent_dim":64}
    config["init_checkpoint"] = args.init_checkpoint
    save_json(config, out/"config.json")
    history=[]; best=float("inf"); best_any=float("inf"); best_epoch=None
    for epoch in range(1, args.epochs+1):
        model.train(); sums={k:0. for k in ["loss","recon","vq","cf12","cfres2","semres3","cf12_cos","cfres2_cos","semres3_cos"]}; seen=0
        for bx, by, bcfr, bsemr in loader:
            bx, by, bcfr, bsemr = bx.to(device), by.to(device), bcfr.to(device), bsemr.to(device); opt.zero_grad()
            recon, vq, _, qs = model(bx); q1,q2,q3=qs; h12=q1+q2
            cf12_pred=model.cf12_proj(h12)
            cfres_pred=model.cfres2_proj(q2); semres_pred=model.semres3_proj(q3)
            cf_target=model.cf_target_proj(by); cfres_target=model.cfres_target_proj(bcfr)
            recon_l=F.mse_loss(recon,bx); cf12=symmetric_infonce(cf12_pred,cf_target)
            cfres=symmetric_infonce(cfres_pred,cfres_target); semres=cosine_loss(semres_pred,bsemr)
            loss=recon_l+vq+args.lambda_cf12*cf12+args.lambda_cfres2*cfres+args.lambda_semres3*semres
            loss.backward(); opt.step(); n=len(bx); seen+=n
            values={"loss":loss,"recon":recon_l,"vq":vq,"cf12":cf12,"cfres2":cfres,"semres3":semres,
                    "cf12_cos":F.cosine_similarity(cf12_pred,cf_target,dim=1).mean(),
                    "cfres2_cos":F.cosine_similarity(cfres_pred,cfres_target,dim=1).mean(),"semres3_cos":1-semres}
            for k,v in values.items(): sums[k]+=float(v.detach())*n
        row={"epoch":epoch,**{k:v/seen for k,v in sums.items()}}
        u=usage(model,x,device,args.batch_size)
        row.update({"c1_unique":u[0],"c2_unique":u[1],"c3_unique":u[2]})
        row["structure_valid"] = bool(60 <= u[0] <= 140 and u[1] >= 180 and u[2] >= 180)
        history.append(row); print(json.dumps(row),flush=True)
        state={"model_state_dict":model.state_dict(),"config":config,"epoch":epoch,"row":row}
        if row["loss"] < best_any:
            best_any=row["loss"]; torch.save(state,out/"best_unconstrained_model.pt")
        if row["structure_valid"] and row["loss"] < best:
            best=row["loss"]; best_epoch=epoch; torch.save(state,out/"best_model.pt")
    torch.save(state,out/"final_model.pt")
    if best_epoch is None:
        fallback=torch.load(out/"best_unconstrained_model.pt",map_location="cpu",weights_only=False)
        torch.save(fallback,out/"best_model.pt");best=best_any;best_epoch=int(fallback["epoch"])
    save_json({"config":config,"checkpoint_selection":"lowest loss among structure-valid epochs; unconstrained fallback only if none",
               "best_loss":best,"best_epoch":best_epoch,"best_structure_valid":bool(any(r["structure_valid"] for r in history)),
               "history":history,"final":history[-1]},out/"training_summary.json")

if __name__ == "__main__": main()
