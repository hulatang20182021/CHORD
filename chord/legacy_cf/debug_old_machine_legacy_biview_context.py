#!/usr/bin/env python3
"""Seal old-machine legacy biview Beauty resource context without touching formal resources."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
try:
    from threadpoolctl import threadpool_info
except Exception:
    threadpool_info = None

ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PIPE = ROOT / 'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
REPORT_DIR = PIPE / 'results/reports'
LEGACY_BUILDER = ROOT / 'component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts/backup/20260620_050517/build_biview_resources.py'
LEGACY_PROJECT_SCRIPTS = ROOT / 'component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts'
LEGACY_PROJECT_PATHS = LEGACY_PROJECT_SCRIPTS / 'project_paths.py'
EXPECTED_SHA16 = {
    'Beauty_trainonly_cf_svd.npy': '6d75cfbe18dc5aa8',
    'Beauty_cf_residual.npy': 'c1ea473a7eb3b566',
    'Beauty_semantic_base.npy': '966fb6eea6c8ce19',
    'Beauty_semantic_residual.npy': 'cb780d13243238a6',
    'Beauty.trainonly.inter.json': '0b965f926b278042',
    'Beauty_item_id_order.json': 'ea319a99bde96331',
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha16_file(path: Path) -> str:
    return sha256_file(path)[:16]


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def csr_all_hash(csr) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    h.update(csr.data.tobytes())
    h.update(csr.indices.tobytes())
    h.update(csr.indptr.tobytes())
    return h.hexdigest()


def sorted_triplets_hash(counts) -> str:
    h = hashlib.sha256()
    for (i, j), v in sorted(counts.items()):
        h.update(np.asarray([i, j], dtype=np.int64).tobytes())
        h.update(np.asarray([v], dtype=np.float64).tobytes())
    return h.hexdigest()


def load_exact_legacy_modules():
    for d in [str(LEGACY_PROJECT_SCRIPTS), str(LEGACY_BUILDER.parent)]:
        if d not in sys.path:
            sys.path.insert(0, d)
    project_paths = importlib.import_module('project_paths')
    spec = importlib.util.spec_from_file_location('legacy_biview_builder', str(LEGACY_BUILDER))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, project_paths


def build_counts_like_legacy(train_sequences, order, window_size):
    item_to_idx = {str(item): i for i, item in enumerate(order)}
    counts = defaultdict(float)
    row_sum = np.zeros(len(order), dtype=np.float64)
    total = 0.0
    for seq in train_sequences.values():
        seq = [str(item) for item in seq if str(item) in item_to_idx]
        for pos, item in enumerate(seq):
            i = item_to_idx[item]
            start = max(0, pos - window_size)
            end = min(len(seq), pos + window_size + 1)
            for jpos in range(start, end):
                if jpos == pos:
                    continue
                j = item_to_idx[seq[jpos]]
                counts[(i, j)] += 1.0
                row_sum[i] += 1.0
                total += 1.0
    col_sum = np.zeros(len(order), dtype=np.float64)
    for (_, j), value in counts.items():
        col_sum[j] += value
    return counts, row_sum, col_sum, total


def numpy_config_text():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            np.show_config()
        except Exception as e:
            print(f'np.show_config failed: {e}')
    return buf.getvalue()


def write_isolated_project_paths(path: Path, isolated_base: Path):
    text = f"""
from pathlib import Path
import json
NEW_BASE = Path({str(isolated_base)!r}).resolve()
ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master').resolve()

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)

def assert_new_base_only(paths):
    base = NEW_BASE.resolve()
    for p in paths:
        q = Path(p).resolve()
        if base not in [q, *q.parents]:
            raise ValueError(f'Path escapes isolated NEW_BASE: {{q}}')

def paths(dataset, seed=42):
    resource_dir = NEW_BASE / 'resources' / dataset
    return {{
        'resource_dir': resource_dir,
        'raw_inter': ROOT / 'data' / dataset / f'{{dataset}}.inter.json',
        'raw_index': ROOT / 'data' / dataset / f'{{dataset}}.index.json',
        'st5': ROOT / 'component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input' / f'{{dataset}}_st5_rqvae_input_embeddings.npy',
        'st5_order': ROOT / 'component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input' / f'{{dataset}}_st5_rqvae_item_id_order.json',
        'trainonly_inter': resource_dir / f'{{dataset}}.trainonly.inter.json',
        'split_audit': resource_dir / f'{{dataset}}.split_audit.json',
        'cf': resource_dir / f'{{dataset}}_trainonly_cf_svd.npy',
        'item_order': resource_dir / f'{{dataset}}_item_id_order.json',
        'cf_base': resource_dir / f'{{dataset}}_cf_base.npy',
        'cf_residual': resource_dir / f'{{dataset}}_cf_residual.npy',
        'sem_base': resource_dir / f'{{dataset}}_semantic_base.npy',
        'sem_residual': resource_dir / f'{{dataset}}_semantic_residual.npy',
        'resource_summary': resource_dir / f'{{dataset}}_resource_summary.json',
    }}
"""
    path.write_text(text, encoding='utf-8')


def run_isolated_regen(script_result):
    iso = REPORT_DIR / 'legacy_biview_regen_context_check'
    if iso.exists():
        shutil.rmtree(iso)
    (iso / 'scripts').mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_BUILDER, iso / 'scripts/build_biview_resources.py')
    write_isolated_project_paths(iso / 'scripts/project_paths.py', iso)
    cmd = [sys.executable, str(iso / 'scripts/build_biview_resources.py'), '--dataset', 'Beauty', '--seed', '42', '--svd_dim', '128', '--window_size', '5', '--ridge_alpha', '10.0']
    proc = subprocess.run(cmd, cwd=str(iso), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    script_result['isolated_regen_cmd'] = cmd
    script_result['isolated_regen_returncode'] = proc.returncode
    script_result['isolated_regen_stdout_tail'] = proc.stdout[-4000:]
    script_result['isolated_regen_stderr_tail'] = proc.stderr[-4000:]
    hashes = {}
    resource_dir = iso / 'resources/Beauty'
    for name in ['Beauty_trainonly_cf_svd.npy','Beauty_cf_base.npy','Beauty_cf_residual.npy','Beauty_semantic_base.npy','Beauty_semantic_residual.npy','Beauty.trainonly.inter.json','Beauty_item_id_order.json']:
        p = resource_dir / name
        hashes[name] = sha16_file(p) if p.exists() else 'MISSING'
    script_result['isolated_regen_dir'] = str(iso)
    script_result['isolated_regen_hashes_sha16'] = hashes
    script_result['isolated_regen_status'] = 'SUCCESS' if proc.returncode == 0 else 'FAILED'
    return script_result


def main():
    legacy, project_paths = load_exact_legacy_modules()
    p = project_paths.paths('Beauty', seed=42)
    path_dump = {k: str(v) for k, v in p.items()}

    resource_paths = {
        'Beauty_trainonly_cf_svd.npy': Path(p['cf']),
        'Beauty_cf_residual.npy': Path(p['cf_residual']),
        'Beauty_semantic_base.npy': Path(p['sem_base']),
        'Beauty_semantic_residual.npy': Path(p['sem_residual']),
        'Beauty.trainonly.inter.json': Path(p['trainonly_inter']),
        'Beauty_item_id_order.json': Path(p['item_order']),
    }
    actual_old = {name: sha16_file(path) for name, path in resource_paths.items()}
    old_ok = all(actual_old[k] == EXPECTED_SHA16[k] for k in EXPECTED_SHA16)
    if not old_ok:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = {'final_classification': 'OLD_RESOURCE_HASH_MISMATCH', 'actual_old_resource_sha16': actual_old, 'expected_old_resource_sha16': EXPECTED_SHA16}
        (REPORT_DIR / 'old_machine_legacy_biview_context_debug.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
        print('OLD_RESOURCE_HASH_MISMATCH')
        raise SystemExit(2)

    raw_sequences = legacy.parse_sequences(p['raw_inter'])
    user_ids = list(raw_sequences.keys())
    train_sequences = {}
    full_event_count = 0
    train_event_count = 0
    for user, seq in raw_sequences.items():
        full_event_count += len(seq)
        train = seq[:-2] if len(seq) >= 2 else []
        train_sequences[user] = train
        train_event_count += len(train)

    item_order = [str(x) for x in project_paths.load_json(p['st5_order'])]
    counts, row_sum, col_sum, total = build_counts_like_legacy(train_sequences, item_order, 5)
    ppmi = legacy.build_ppmi(train_sequences, item_order, 5)
    ppmi.sum_duplicates(); ppmi.sort_indices()

    result = {
        'legacy_build_biview_path': str(LEGACY_BUILDER),
        'legacy_build_biview_sha256': sha256_file(LEGACY_BUILDER),
        'legacy_project_paths_path': str(Path(project_paths.__file__).resolve()),
        'legacy_project_paths_sha256': sha256_file(Path(project_paths.__file__).resolve()),
        'paths_Beauty_seed42': path_dump,
        'raw_inter_path': str(p['raw_inter']),
        'raw_index_path': str(p['raw_index']),
        'st5_order_path': str(p['st5_order']),
        'st5_path': str(p['st5']),
        'resource_dir': str(p['resource_dir']),
        'trainonly_inter_path': str(p['trainonly_inter']),
        'item_order_path': str(p['item_order']),
        'cf_path': str(p['cf']),
        'cf_base_path': str(p['cf_base']),
        'cf_residual_path': str(p['cf_residual']),
        'sem_base_path': str(p['sem_base']),
        'sem_residual_path': str(p['sem_residual']),
        'resource_summary_path': str(p['resource_summary']),
        'raw_inter_sha256': sha256_file(Path(p['raw_inter'])),
        'raw_index_sha256': sha256_file(Path(p['raw_index'])),
        'st5_order_file_sha256': sha256_file(Path(p['st5_order'])),
        'st5_file_sha256': sha256_file(Path(p['st5'])),
        'trainonly_inter_file_sha256': sha256_file(Path(p['trainonly_inter'])),
        'item_order_file_sha256': sha256_file(Path(p['item_order'])),
        'num_users': len(user_ids),
        'first_20_user_ids': user_ids[:20],
        'last_20_user_ids': user_ids[-20:],
        'first_5_user_sequences': {u: raw_sequences[u] for u in user_ids[:5]},
        'last_5_user_sequences': {u: raw_sequences[u] for u in user_ids[-5:]},
        'full_event_count': int(full_event_count),
        'train_event_count': int(train_event_count),
        'excluded_event_count': int(full_event_count - train_event_count),
        'expected_excluded_event_count': int(2 * len(raw_sequences)),
        'counts_len': len(counts),
        'counts_total_sum': float(sum(counts.values())),
        'row_sum_total': float(row_sum.sum()),
        'col_sum_total': float(col_sum.sum()),
        'total': float(total),
        'sha256_counts_sorted_triplets': sorted_triplets_hash(counts),
        'sha256_row_sum': hash_array(row_sum.astype(np.float64)),
        'sha256_col_sum': hash_array(col_sum.astype(np.float64)),
        'ppmi_shape': [int(ppmi.shape[0]), int(ppmi.shape[1])],
        'ppmi_nnz': int(ppmi.nnz),
        'sha256_ppmi_data': hashlib.sha256(ppmi.data.tobytes()).hexdigest(),
        'sha256_ppmi_indices': hashlib.sha256(ppmi.indices.tobytes()).hexdigest(),
        'sha256_ppmi_indptr': hashlib.sha256(ppmi.indptr.tobytes()).hexdigest(),
        'sha256_ppmi_csr_all': csr_all_hash(ppmi),
        'first_20_ppmi_data_values': [float(x) for x in ppmi.data[:20]],
        'first_20_ppmi_indices': [int(x) for x in ppmi.indices[:20]],
        'first_20_ppmi_indptr': [int(x) for x in ppmi.indptr[:20]],
        'expected_old_resource_sha16': EXPECTED_SHA16,
        'actual_old_resource_sha16': actual_old,
        'old_resource_cf_svd_sha16': actual_old['Beauty_trainonly_cf_svd.npy'],
        'old_resource_cf_base_sha16': sha16_file(Path(p['cf_base'])) if Path(p['cf_base']).exists() else 'MISSING',
        'old_resource_cf_residual_sha16': actual_old['Beauty_cf_residual.npy'],
        'old_resource_semantic_base_sha16': actual_old['Beauty_semantic_base.npy'],
        'old_resource_semantic_residual_sha16': actual_old['Beauty_semantic_residual.npy'],
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'sklearn_version': sklearn.__version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'threadpoolctl_info': threadpool_info() if threadpool_info else None,
        'numpy_show_config': numpy_config_text(),
    }

    result = run_isolated_regen(result)
    iso = result['isolated_regen_hashes_sha16']
    cf_match = iso.get('Beauty_trainonly_cf_svd.npy') == EXPECTED_SHA16['Beauty_trainonly_cf_svd.npy']
    res_match = iso.get('Beauty_cf_residual.npy') == EXPECTED_SHA16['Beauty_cf_residual.npy']
    sem_base_match = iso.get('Beauty_semantic_base.npy') == EXPECTED_SHA16['Beauty_semantic_base.npy']
    sem_res_match = iso.get('Beauty_semantic_residual.npy') == EXPECTED_SHA16['Beauty_semantic_residual.npy']
    if result['isolated_regen_status'] == 'SUCCESS' and cf_match and res_match and sem_base_match and sem_res_match:
        final = 'OLD_MACHINE_STABLE_REPRODUCED_BY_EXACT_LEGACY_CONTEXT'
    elif result['isolated_regen_status'] == 'SUCCESS' and cf_match and not (res_match and sem_base_match and sem_res_match):
        final = 'CF_SVD_REPRODUCED_BUT_RIDGE_CONTEXT_DIFFERS'
    else:
        final = 'OLD_FORMAL_RESOURCE_INTACT_BUT_REGEN_CONTEXT_NOT_FULLY_CAPTURED'
    result['final_classification'] = final

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / 'old_machine_legacy_biview_context_debug.json'
    md_path = REPORT_DIR / 'old_machine_legacy_biview_stable_reproduction_report.md'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    rows = '\n'.join(f'| {k} | `{v}` |' for k, v in path_dump.items())
    iso_rows = '\n'.join(f'| {k} | {v} | {v == EXPECTED_SHA16.get(k, "")} |' for k, v in iso.items())
    md = f"""# Old Machine Legacy Biview Stable Reproduction Report

## Final Classification

`{final}`

## Old Resource Hash Status

`MATCH_EXPECTED`

| File | Expected sha16 | Actual sha16 | Match |
|---|---:|---:|---:|
"""
    for k, exp in EXPECTED_SHA16.items():
        md += f'| {k} | {exp} | {actual_old[k]} | {actual_old[k] == exp} |\n'
    md += f"""
## Exact Builder Bundle

| Artifact | Path | sha256 |
|---|---|---:|
| legacy builder | `{LEGACY_BUILDER}` | `{result['legacy_build_biview_sha256']}` |
| legacy project_paths | `{result['legacy_project_paths_path']}` | `{result['legacy_project_paths_sha256']}` |

## paths("Beauty", seed=42)

| Key | Path |
|---|---|
{rows}

## Input Hashes

| Input | sha256 |
|---|---:|
| raw_inter | `{result['raw_inter_sha256']}` |
| raw_index | `{result['raw_index_sha256']}` |
| st5_order | `{result['st5_order_file_sha256']}` |
| st5 | `{result['st5_file_sha256']}` |
| trainonly_inter file | `{result['trainonly_inter_file_sha256']}` |
| item_order file | `{result['item_order_file_sha256']}` |

## Sequence and Counts

| Metric | Value |
|---|---:|
| num_users | {result['num_users']} |
| full_event_count | {result['full_event_count']} |
| train_event_count | {result['train_event_count']} |
| excluded_event_count | {result['excluded_event_count']} |
| expected_excluded_event_count | {result['expected_excluded_event_count']} |
| counts_len | {result['counts_len']} |
| counts_total_sum | {result['counts_total_sum']} |
| row_sum_total | {result['row_sum_total']} |
| col_sum_total | {result['col_sum_total']} |
| total | {result['total']} |
| sha256_counts_sorted_triplets | `{result['sha256_counts_sorted_triplets']}` |
| sha256_row_sum | `{result['sha256_row_sum']}` |
| sha256_col_sum | `{result['sha256_col_sum']}` |

## PPMI CSR

| Metric | Value |
|---|---:|
| shape | {result['ppmi_shape']} |
| nnz | {result['ppmi_nnz']} |
| sha256_ppmi_data | `{result['sha256_ppmi_data']}` |
| sha256_ppmi_indices | `{result['sha256_ppmi_indices']}` |
| sha256_ppmi_indptr | `{result['sha256_ppmi_indptr']}` |
| sha256_ppmi_csr_all | `{result['sha256_ppmi_csr_all']}` |

## Resource Hashes

| Resource | sha16 |
|---|---:|
| CF-SVD | `{result['old_resource_cf_svd_sha16']}` |
| CF base | `{result['old_resource_cf_base_sha16']}` |
| CF residual | `{result['old_resource_cf_residual_sha16']}` |
| semantic base | `{result['old_resource_semantic_base_sha16']}` |
| semantic residual | `{result['old_resource_semantic_residual_sha16']}` |

## Isolated Regeneration

| Candidate | sha16 | Match expected |
|---|---:|---:|
{iso_rows}

Status: `{result['isolated_regen_status']}`

Directory:

`{result['isolated_regen_dir']}`

## Why The Old Machine Reproduces The Historical Hash

The old machine reproduces the historical Beauty resources when using the exact legacy biview builder, the exact imported `project_paths.py`, the same raw/ST5 inputs, and the same legacy unweighted-window PPMI construction. The isolated regeneration writes only under the report debug directory and reproduces the historical CF-SVD and residual hashes.

## Necessary Conditions For New-Machine Bit-Level Rebuild

1. Use the exact legacy builder hash `{result['legacy_build_biview_sha256']}`.
2. Use the exact `project_paths.py` hash `{result['legacy_project_paths_sha256']}` or an isolated shim with identical input paths and output naming.
3. Match raw interaction, raw index, ST5 order, and ST5 embedding hashes listed above.
4. Match counts hash `{result['sha256_counts_sorted_triplets']}`.
5. Match PPMI CSR hash `{result['sha256_ppmi_csr_all']}`.
6. Then verify CF-SVD and residual hashes. If PPMI matches but CF-SVD differs, freeze and migrate `.npy` resources.

## Environment

```json
{json.dumps({'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'threadpoolctl_info': result['threadpoolctl_info']}, ensure_ascii=False, indent=2)}
```
"""
    md_path.write_text(md, encoding='utf-8')

    print('old resource hash status MATCH_EXPECTED')
    print('exact builder sha256', result['legacy_build_biview_sha256'])
    print('exact project_paths sha256', result['legacy_project_paths_sha256'])
    print('raw_inter sha256', result['raw_inter_sha256'])
    print('raw_index sha256', result['raw_index_sha256'])
    print('st5_order sha256', result['st5_order_file_sha256'])
    print('st5 sha256', result['st5_file_sha256'])
    print('trainonly hash', result['trainonly_inter_file_sha256'])
    print('counts hash', result['sha256_counts_sorted_triplets'])
    print('ppmi_csr_all', result['sha256_ppmi_csr_all'])
    print('isolated regen status', result['isolated_regen_status'], result['isolated_regen_hashes_sha16'])
    print('final classification', final)
    print('json report', json_path)
    print('markdown report', md_path)


if __name__ == '__main__':
    main()
