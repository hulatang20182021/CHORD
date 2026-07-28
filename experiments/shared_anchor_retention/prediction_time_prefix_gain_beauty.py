#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[2]


def clean_sid(s: str) -> str:
    return str(s).strip().replace(" ", "")


def ndcg_from_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 2)


def summarize_prefix_records(records: list[dict[str, Any]], ks=(1, 5, 10, 20)) -> dict[str, Any]:
    out: dict[str, Any] = {"sample_count": len(records)}
    depths = [("c1", 1), ("c1c2", 2), ("c1c2c3", 3), ("full", 4)]
    for name, depth in depths:
        for k in ks:
            hits = []
            ndcgs = []
            for rec in records:
                rank = rec.get(f"rank_{name}")
                hit = rank is not None and rank < k
                hits.append(float(hit))
                ndcgs.append(ndcg_from_rank(rank) if hit else 0.0)
            out[f"{name}_hit@{k}"] = float(np.mean(hits)) if hits else 0.0
            out[f"{name}_ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    # Incremental retention: among users whose shallower prefix appears in top-k, how often the next prefix also appears.
    for k in ks:
        for shallow, deep in [("c1", "c1c2"), ("c1c2", "c1c2c3"), ("c1c2c3", "full")]:
            denom = sum(1 for r in records if r.get(f"rank_{shallow}") is not None and r[f"rank_{shallow}"] < k)
            numer = sum(
                1
                for r in records
                if r.get(f"rank_{shallow}") is not None
                and r[f"rank_{shallow}"] < k
                and r.get(f"rank_{deep}") is not None
                and r[f"rank_{deep}"] < k
            )
            out[f"retention_{shallow}_to_{deep}@{k}"] = float(numer / denom) if denom else 0.0
    return out


def prefix_branching(index_json: Path) -> dict[str, float]:
    raw = json.loads(index_json.read_text(encoding="utf-8"))
    sids = [list(v)[:4] for v in raw.values()]
    out: dict[str, float] = {}
    for depth, name in [(1, "c1_to_c2"), (2, "c1c2_to_c3"), (3, "c1c2c3_to_c4")]:
        children: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for sid in sids:
            if len(sid) > depth:
                children[tuple(str(x) for x in sid[:depth])].add(str(sid[depth]))
        vals = [len(v) for v in children.values()]
        out[f"avg_branch_{name}"] = float(np.mean(vals)) if vals else 0.0
        out[f"max_branch_{name}"] = float(np.max(vals)) if vals else 0.0
    return out


def rank_prefixes(pred_sids: list[list[str]], target_sid: list[str]) -> dict[str, int | None]:
    prefixes = {
        "c1": tuple(target_sid[:1]),
        "c1c2": tuple(target_sid[:2]),
        "c1c2c3": tuple(target_sid[:3]),
        "full": tuple(target_sid[:4]),
    }
    ranks = {f"rank_{k}": None for k in prefixes}
    for rank, sid in enumerate(pred_sids):
        sid_t = tuple(sid[:4])
        for name, pref in prefixes.items():
            key = f"rank_{name}"
            if ranks[key] is None and sid_t[: len(pref)] == pref:
                ranks[key] = rank
    return ranks


class T5SeqDataset(Dataset):
    def __init__(self, inter_json: Path, index_json: Path, max_his_len: int = 20, sample_num: int = -1, seed: int = 42):
        self.inters = json.loads(inter_json.read_text(encoding="utf-8"))
        self.index = {str(k): list(v) for k, v in json.loads(index_json.read_text(encoding="utf-8")).items()}
        self.max_his_len = max_his_len
        self.rows = []
        for uid, items in self.inters.items():
            items = [str(x) for x in items]
            if len(items) < 2:
                continue
            target_item = items[-1]
            if target_item not in self.index:
                continue
            hist_items = [x for x in items[:-1] if x in self.index]
            if max_his_len > 0:
                hist_items = hist_items[-max_his_len:]
            history = "".join("".join(self.index[x]) for x in hist_items)
            target_sid = self.index[target_item]
            self.rows.append({
                "uid": uid,
                "history": history,
                "target_item": target_item,
                "target": "".join(target_sid),
                "target_sid": target_sid,
            })
        if sample_num and sample_num > 0 and sample_num < len(self.rows):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.rows), size=sample_num, replace=False)
            self.rows = [self.rows[int(i)] for i in idx]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def all_item_strings(self):
        return {"".join(v) for v in self.index.values()}

    def sid_by_string(self):
        return {"".join(v): list(v) for v in self.index.values()}


class T5Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):
        inputs = self.tokenizer(
            [b["history"] for b in batch],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
        )
        return inputs, batch


def run_t5_style(args, method: str, ckpt_path: Path, index_json: Path, inter_json: Path, base_model: Path, out_dir: Path):
    sys.path.insert(0, str(Path(args.letter_root) / "LETTER-TIGER"))
    from generation_trie import Trie
    from utils import prefix_allowed_tokens_fn
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(str(ckpt_path), model_max_length=512)
    # For CHORD/LETTER final checkpoints, from_pretrained works because the saved class is T5-compatible for generation.
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt_path), low_cpu_mem_usage=True).to(device)
    model.eval()

    ds = T5SeqDataset(inter_json, index_json, max_his_len=args.max_his_len, sample_num=args.sample_num, seed=args.seed)
    all_items = ds.all_item_strings()
    sid_by_str = ds.sid_by_string()
    trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    allowed = prefix_allowed_tokens_fn(trie)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=T5Collator(tokenizer))
    records = []
    invalid = 0
    total_pred = 0
    with torch.no_grad():
        for inputs, batch in tqdm(loader, desc=f"{method} generate"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=allowed,
                num_beams=args.num_beams,
                num_return_sequences=args.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            seqs = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
            scores = output["sequences_scores"].detach().cpu().tolist()
            bsz = len(batch)
            for b in range(bsz):
                pairs = []
                for j in range(args.num_beams):
                    p = clean_sid(seqs[b * args.num_beams + j])
                    s = float(scores[b * args.num_beams + j])
                    total_pred += 1
                    if p not in all_items:
                        invalid += 1
                        s = -1e9
                    pairs.append((p, s))
                pairs.sort(key=lambda x: x[1], reverse=True)
                pred_sids = [sid_by_str[p] for p, s in pairs if p in sid_by_str]
                rec = {
                    "method": method,
                    "uid": batch[b]["uid"],
                    "target_item": batch[b]["target_item"],
                    "target": batch[b]["target"],
                    "target_sid": batch[b]["target_sid"],
                    "top1": pairs[0][0] if pairs else "",
                    "invalid_predictions_in_beam": sum(1 for p, _ in pairs if p not in sid_by_str),
                }
                rec.update(rank_prefixes(pred_sids, batch[b]["target_sid"]))
                records.append(rec)
    summary = summarize_prefix_records(records, ks=(1, 5, 10, 20))
    summary.update({"method": method, "invalid_prediction_ratio": invalid / max(total_pred, 1), "checkpoint": str(ckpt_path), "index": str(index_json), "sample_num": len(records)})
    summary.update(prefix_branching(index_json))
    return records, summary


def run_tiger(args, ckpt_path: Path, code_path: Path, dataset_path: Path, out_dir: Path):
    sys.path.insert(0, str(Path(args.tiger_repo) / "model"))
    from main import TIGER
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader

    config = {
        "num_layers": 4, "num_decoder_layers": 4, "d_model": 128, "d_ff": 1024,
        "num_heads": 6, "d_kv": 64, "dropout_rate": 0.1, "vocab_size": 1025,
        "pad_token_id": 0, "eos_token_id": 0, "feed_forward_proj": "relu",
    }
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    model = TIGER(config)
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    ds = GenRecDataset(str(dataset_path / "test.parquet"), str(code_path), mode="evaluation", max_len=args.max_his_len)
    if args.sample_num and args.sample_num > 0 and args.sample_num < len(ds):
        rng = np.random.default_rng(args.seed)
        keep = set(map(int, rng.choice(len(ds), size=args.sample_num, replace=False)))
        ds.data = [x for i, x in enumerate(ds.data) if i in keep]
    loader = GenRecDataLoader(ds, batch_size=args.batch_size, shuffle=False)
    records = []
    with torch.no_grad():
        cursor = 0
        for batch in tqdm(loader, desc="TIGER generate"):
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].cpu().numpy().tolist()
            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=args.num_beams)
            preds = preds[:, 1:].detach().cpu().numpy().reshape(input_ids.shape[0], args.num_beams, -1).tolist()
            for i, target in enumerate(labels):
                pred_sids = [[str(int(x)) for x in p[:4]] for p in preds[i]]
                target_sid = [str(int(x)) for x in target[:4]]
                rec = {"method": "TIGER", "uid": str(cursor), "target_item": "", "target": " ".join(target_sid), "target_sid": target_sid, "top1": " ".join(pred_sids[0])}
                rec.update(rank_prefixes(pred_sids, target_sid))
                records.append(rec)
                cursor += 1
    summary = summarize_prefix_records(records, ks=(1, 5, 10, 20))
    summary.update({"method": "TIGER", "invalid_prediction_ratio": 0.0, "checkpoint": str(ckpt_path), "index": str(code_path), "sample_num": len(records)})
    return records, summary


def write_outputs(out_dir: Path, all_records: list[dict[str, Any]], summaries: list[dict[str, Any]]):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(out_dir / "prediction_time_prefix_gain_records.csv", index=False)
    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "prediction_time_prefix_gain_summary.csv", index=False)
    (out_dir / "prediction_time_prefix_gain_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    cols = [
        "method", "sample_num",
        "c1_hit@10", "c1c2_hit@10", "c1c2c3_hit@10", "full_hit@10",
        "c1_ndcg@10", "c1c2_ndcg@10", "c1c2c3_ndcg@10", "full_ndcg@10",
        "retention_c1_to_c1c2@10", "retention_c1c2_to_c1c2c3@10", "retention_c1c2c3_to_full@10",
        "retention_c1_to_c1c2@20", "retention_c1c2_to_c1c2c3@20", "retention_c1c2c3_to_full@20",
        "avg_branch_c1_to_c2", "avg_branch_c1c2_to_c3", "avg_branch_c1c2c3_to_c4",
    ]
    cols = [c for c in cols if c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    table = "\n".join([header, sep] + rows)
    lines = ["# Beauty Prediction-Time Prefix Retention", "", "This report measures whether generated beams contain the target prefix at increasing SID depths.", "", table]
    lines += ["", "Interpretation: c1/c1c2/c1c2c3 are prefix-level hits in the model's generated beam; full is exact SID hit. This is prediction-time evidence, unlike static centroid localization."]
    (out_dir / "prediction_time_prefix_gain_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_beams", type=int, default=20)
    ap.add_argument("--max_his_len", type=int, default=20)
    ap.add_argument("--sample_num", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--methods", default="CHORD")
    ap.add_argument("--out_dir", default=str(PROJECT / "results/prefix_retention_beauty"))
    ap.add_argument("--inter_json", default=str(PROJECT / "data/Beauty/Beauty.inter.json"))
    ap.add_argument("--letter_root", default=str(PROJECT / "runtime_root/LETTER-master"))
    ap.add_argument("--tiger_repo", default="")
    ap.add_argument("--tiger_ckpt", default="")
    ap.add_argument("--tiger_code", default="")
    ap.add_argument("--tiger_data", default="")
    ap.add_argument("--letter_ckpt", default="")
    ap.add_argument("--letter_index", default="")
    ap.add_argument("--chord_ckpt", default="")
    ap.add_argument("--chord_index", default="")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    inter = Path(args.inter_json)
    records_all: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    methods = {m.strip().upper() for m in args.methods.split(",") if m.strip()}
    if "TIGER" in methods:
        required = [args.tiger_repo, args.tiger_ckpt, args.tiger_code, args.tiger_data]
        if not all(required):
            raise SystemExit("TIGER requires --tiger_repo/--tiger_ckpt/--tiger_code/--tiger_data")
        r, s = run_tiger(
            args,
            Path(args.tiger_ckpt),
            Path(args.tiger_code),
            Path(args.tiger_data),
            out_dir,
        )
        records_all.extend(r); summaries.append(s); write_outputs(out_dir, records_all, summaries)
    if "LETTER" in methods:
        if not args.letter_ckpt or not args.letter_index:
            raise SystemExit("LETTER requires --letter_ckpt and --letter_index")
        r, s = run_t5_style(
            args, "LETTER",
            Path(args.letter_ckpt),
            Path(args.letter_index),
            inter,
            Path(args.letter_root) / "LETTER-TIGER/ckpt/TIGER",
            out_dir,
        )
        records_all.extend(r); summaries.append(s); write_outputs(out_dir, records_all, summaries)
    if "CHORD" in methods:
        if not args.chord_ckpt or not args.chord_index:
            raise SystemExit("CHORD requires --chord_ckpt and --chord_index")
        r, s = run_t5_style(
            args, "CHORD",
            Path(args.chord_ckpt),
            Path(args.chord_index),
            inter,
            Path(args.letter_root) / "LETTER-TIGER/ckpt/TIGER",
            out_dir,
        )
        records_all.extend(r); summaries.append(s); write_outputs(out_dir, records_all, summaries)
    print(json.dumps({"out_dir": str(out_dir), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
