#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PROJECT = ROOT / 'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
COLD_BASE = PROJECT / 'results/pls_sd128_dpos_pcsc/cold_start'
EXP_BASE = PROJECT / 'results/pls_sd128_dpos_pcsc/order_ablation_cold_start'


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def ratio_tag(r: float) -> str:
    return f"cold{int(round(float(r) * 100)):02d}"


def swap_sid(sid):
    if len(sid) != 4:
        raise ValueError(f'Expected 4-token SID, got {sid}')
    c2 = int(str(sid[2]).split('_', 1)[1].rstrip('>'))
    c3 = int(str(sid[1]).split('_', 1)[1].rstrip('>'))
    return [sid[0], f'<b_{c2}>', f'<c_{c3}>', sid[3]]


def swap_raw(raw):
    out = {}
    for k, v in raw.items():
        rec = dict(v)
        rec['c2'], rec['c3'] = int(v['c3']), int(v['c2'])
        rec['order_transform'] = 'swap_c2_c3'
        out[k] = rec
    return out


def pkey(sid, n):
    return '|'.join(sid[:n])


def prefix_maps(index, cold_items):
    cold_set = set(map(str, cold_items))
    cold_item_to_sid = {str(i): index[str(i)] for i in cold_items if str(i) in index}
    maps = {}
    for n in [1, 2, 3]:
        mp = defaultdict(list)
        for item, sid in cold_item_to_sid.items():
            mp[pkey(sid, n)].append(str(item))
        maps[n] = {k: sorted(v, key=lambda x: int(x) if str(x).isdigit() else x) for k, v in sorted(mp.items())}
    return cold_item_to_sid, maps


def bucket_stats(index):
    counts = Counter(tuple(v[:3]) for v in index.values())
    sizes = list(counts.values())
    singleton = sum(1 for x in sizes if x == 1) / max(len(sizes), 1)
    return {
        'prefix3_unique': len(counts),
        'prefix3_singleton_ratio': singleton,
        'max_prefix3_bucket': max(sizes) if sizes else 0,
    }


def write_data_alias(alias, inter, item_json, index, split_meta):
    dst = EXP_BASE / 'data' / alias
    dst.mkdir(parents=True, exist_ok=True)
    save_json({str(k): [int(x) for x in v] for k, v in inter.items()}, dst / f'{alias}.inter.json')
    save_json(item_json, dst / f'{alias}.item.json')
    save_json(index, dst / f'{alias}.index.json')
    save_json(split_meta, dst / 'dataset_meta.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cold_seed', type=int, default=42)
    ap.add_argument('--cold_ratio', type=float, default=0.05)
    ap.add_argument('--order', choices=['cf_first', 'sem_first'], required=True)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    split_key = f"{args.dataset}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    split_dir = COLD_BASE / split_key
    manifest_path = split_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Cold split manifest not found: {manifest_path}')
    manifest = load_json(manifest_path)
    src_index = load_json(Path(manifest['index_json']))
    src_raw = load_json(Path(manifest['raw_codes']))
    warm_items = [str(x) for x in load_json(split_dir / 'warm_items.json')]
    cold_items = [str(x) for x in load_json(split_dir / 'cold_items.json')]
    warm_inter = load_json(split_dir / 'warm_train.inter.json')
    cold_eval_inter = load_json(split_dir / 'cold_eval.inter.json')
    item_json = load_json(ROOT / 'data' / args.dataset / f'{args.dataset}.item.json')

    run_static = f"{args.dataset}_chord_{args.order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    out = EXP_BASE / 'assets' / split_key / args.order
    if out.exists() and any(out.iterdir()) and not args.force:
        print((out / 'asset_summary.json').read_text(encoding='utf-8'))
        return
    out.mkdir(parents=True, exist_ok=True)

    if args.order == 'cf_first':
        index = {str(k): list(v) for k, v in src_index.items()}
        raw = {str(k): dict(v) for k, v in src_raw.items()}
        pcsc_mapping = {
            'h2': 'cf_residual', 'h3': 'semantic_residual',
            'fusion_h1_h2': 'cf_embedding_and_semantic_base',
            'fusion_h1_h3': 'unused',
        }
    else:
        index = {str(k): swap_sid(v) for k, v in src_index.items()}
        raw = swap_raw(src_raw)
        pcsc_mapping = {
            'h2': 'semantic_residual', 'h3': 'cf_residual',
            'fusion_h1_h2': 'st5_embedding_and_semantic_base',
            'fusion_h1_h3': 'cf_embedding',
        }

    if len(index) != len(src_index):
        raise RuntimeError('item count changed after order transform')
    dup = len(index) - len({tuple(v) for v in index.values()})
    if dup:
        raise RuntimeError(f'duplicate SID after {args.order}: {dup}')
    if len(index) != 12101 and args.dataset == 'Beauty':
        raise RuntimeError(f'Beauty item_count expected 12101, got {len(index)}')

    warm_index = {str(i): index[str(i)] for i in warm_items if str(i) in index}
    cold_item_to_sid, maps = prefix_maps(index, cold_items)
    if len(warm_index) != len(warm_items):
        raise RuntimeError('warm index coverage mismatch')
    if len(cold_item_to_sid) != len(cold_items):
        raise RuntimeError('cold index coverage mismatch')

    save_json(index, out / 'index.json')
    save_json(raw, out / 'raw_codes.json')
    save_json(warm_index, out / 'warm_only.index.json')
    save_json(cold_item_to_sid, out / 'cold_item_to_sid.json')
    for n, mp in maps.items():
        save_json(mp, out / f'cold_prefix{n}_to_items.json')

    train_alias = f"{args.dataset}_chord_{args.order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}_warm_train"
    eval_alias = f"{args.dataset}_chord_{args.order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}_cold_eval"
    meta_base = {
        'dataset': args.dataset, 'method': 'CHORD', 'order': args.order,
        'cold_ratio': args.cold_ratio, 'seed': args.seed, 'cold_seed': args.cold_seed,
        'source_manifest': str(manifest_path), 'static_run': run_static,
    }
    write_data_alias(train_alias, warm_inter, item_json, index, {**meta_base, 'split': 'warm_train'})
    write_data_alias(eval_alias, cold_eval_inter, item_json, index, {**meta_base, 'split': 'cold_eval', 'target_filter': 'cold_only'})

    # Copy resource manifest paths rather than duplicating large arrays.
    resource_dir = Path(manifest['resource_dir'])
    summary = {
        **meta_base,
        'run_static': run_static,
        'train_alias': train_alias,
        'eval_alias': eval_alias,
        'asset_dir': str(out),
        'index_json': str(out / 'index.json'),
        'raw_codes': str(out / 'raw_codes.json'),
        'warm_only_index': str(out / 'warm_only.index.json'),
        'resource_dir': str(resource_dir),
        'item_order': str(resource_dir / f'{args.dataset}_item_id_order.json'),
        'cf_emb': str(resource_dir / f'{args.dataset}_coldstart_cf_svd.npy'),
        'cf_res': str(resource_dir / f'{args.dataset}_coldstart_cf_residual.npy'),
        'sem_base': str(resource_dir / f'{args.dataset}_coldstart_semantic_base.npy'),
        'sem_res_raw': str(resource_dir / f'{args.dataset}_coldstart_semantic_residual.npy'),
        'st5_emb': str(PROJECT / 'results/../../results/plain_st5_rqvae/input'),
        'pcsc_mapping': pcsc_mapping,
        'duplicate_sid_count': dup,
        'item_count': len(index),
        'warm_item_count': len(warm_items),
        'cold_item_count': len(cold_items),
        **bucket_stats(index),
    }
    save_json(summary, out / 'asset_summary.json')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
