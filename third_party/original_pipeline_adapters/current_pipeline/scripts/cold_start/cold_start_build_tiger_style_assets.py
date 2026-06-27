#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _PathForImport
PIPELINE_SCRIPT_ROOT = _PathForImport(__file__).resolve().parents[1]
TIGER_ROOT = _PathForImport("/home/huangxin/llmNrec/Letter/LETTER-master/LETTER-TIGER")
for _p in (PIPELINE_SCRIPT_ROOT, TIGER_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from project_paths import NEW_BASE, save_json, load_json

COLD_BASE = NEW_BASE / 'results/pls_sd128_dpos_pcsc/cold_start'

def ratio_tag(r):
    return f"cold{int(round(float(r)*100)):02d}"

def pkey(sid, n):
    return '|'.join(sid[:n])

def stats(vals):
    if not vals:
        return {'p50': 0.0, 'p95': 0.0, 'max': 0}
    a=np.asarray(vals, dtype=np.int64)
    return {'p50': float(np.percentile(a,50)), 'p95': float(np.percentile(a,95)), 'max': int(a.max())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['Beauty','Instruments','Yelp'])
    ap.add_argument('--cold_ratio', type=float, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cold_seed', type=int, default=42)
    ap.add_argument('--prefix_len', type=int, default=3)
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    split_key=f"{args.dataset}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    split_dir=COLD_BASE/split_key
    manifest_path=split_dir/'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'manifest not found: {manifest_path}')
    manifest=load_json(manifest_path)
    warm=[str(x) for x in load_json(split_dir/'warm_items.json')]
    cold=[str(x) for x in load_json(split_dir/'cold_items.json')]
    warm_set=set(warm); cold_set=set(cold)
    index=load_json(Path(manifest['index_json']))
    raw=load_json(Path(manifest['raw_codes'])) if Path(manifest['raw_codes']).exists() else {}
    out=split_dir/'tiger_style'
    out.mkdir(parents=True, exist_ok=True)
    warm_index={i:index[i] for i in warm if i in index}
    if len(warm_index)!=len(warm):
        raise RuntimeError(f'warm index coverage mismatch {len(warm_index)} vs {len(warm)}')
    cold_item_to_sid={i:index[i] for i in cold if i in index}
    if len(cold_item_to_sid)!=len(cold):
        raise RuntimeError(f'cold index coverage mismatch {len(cold_item_to_sid)} vs {len(cold)}')
    save_json(warm_index, out/'warm_only.index.json')
    save_json(cold_item_to_sid, out/'cold_item_to_sid.json')
    for n in [1,2,3]:
        mp=defaultdict(list)
        for item,sid in cold_item_to_sid.items():
            mp[pkey(sid,n)].append(item)
        for k in mp:
            mp[k]=sorted(mp[k], key=lambda x:int(x) if str(x).isdigit() else str(x))
        save_json(dict(sorted(mp.items())), out/f'cold_prefix{n}_to_items.json')
    prefix3=load_json(out/'cold_prefix3_to_items.json')
    bucket_sizes=[len(v) for v in prefix3.values()]
    s=stats(bucket_sizes)
    summary={
        'dataset': args.dataset,
        'split_key': split_key,
        'cold_ratio': args.cold_ratio,
        'seed': args.seed,
        'cold_seed': args.cold_seed,
        'prefix_len': args.prefix_len,
        'warm_item_count': len(warm),
        'cold_item_count': len(cold),
        'warm_only_index_count': len(warm_index),
        'cold_item_to_sid_count': len(cold_item_to_sid),
        'raw_codes_count': len(raw),
        'prefix3_cold_bucket_count': len(prefix3),
        'prefix3_cold_bucket_p50': s['p50'],
        'prefix3_cold_bucket_p95': s['p95'],
        'max_prefix3_cold_bucket_size': s['max'],
        'outputs': {k:str(out/k) for k in ['warm_only.index.json','cold_prefix1_to_items.json','cold_prefix2_to_items.json','cold_prefix3_to_items.json','cold_item_to_sid.json']},
    }
    save_json(summary, out/'tiger_style_asset_summary.json')
    print(json.dumps(summary, indent=2))
if __name__=='__main__':
    main()
