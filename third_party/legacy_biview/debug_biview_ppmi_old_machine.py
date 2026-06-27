#!/usr/bin/env python3
"""Read-only PPMI debug for historical Beauty legacy biview resources."""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
try:
    from threadpoolctl import threadpool_info
except Exception:  # pragma: no cover
    threadpool_info = None

ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PIPE = ROOT / 'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
LEGACY_BUILDER = ROOT / 'component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts/backup/20260620_050517/build_biview_resources.py'
REPORT_DIR = PIPE / 'results/reports'
NEW_MACHINE_PPMI = '5bad114055b8a6bc1f76921dc7216d60671a8facc44580282aa267f51c48ed8f'
NEW_MACHINE_CF_SHA16 = '4ac176b0e1291413'

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


def canonical_json_sha256(obj) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def load_legacy_builder(path: Path):
    legacy_dir = str(path.parent)
    project_scripts = str(ROOT / 'component_relation_sid/rqvae_supervision/res/biview_shared_private_project/scripts')
    for import_dir in [legacy_dir, project_scripts]:
        if import_dir not in sys.path:
            sys.path.insert(0, import_dir)
    spec = importlib.util.spec_from_file_location('legacy_biview_builder', str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def csr_all_hash(ppmi) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(ppmi.shape, dtype=np.int64).tobytes())
    h.update(ppmi.data.tobytes())
    h.update(ppmi.indices.tobytes())
    h.update(ppmi.indptr.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_first', type=int, default=20)
    args = parser.parse_args()

    raw_inter = ROOT / 'data/Beauty/Beauty.inter.json'
    raw_index = ROOT / 'data/Beauty/Beauty.index.json'
    st5_order = ROOT / 'component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_item_id_order.json'
    st5 = ROOT / 'component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input/Beauty_st5_rqvae_input_embeddings.npy'
    resource_dir = PIPE / 'results/resources/Beauty'

    resource_paths = {name: resource_dir / name for name in EXPECTED_SHA16}
    resource_hashes = {name: sha16_file(path) for name, path in resource_paths.items()}
    old_resource_hash_status = all(resource_hashes[name] == expected for name, expected in EXPECTED_SHA16.items())
    if not old_resource_hash_status:
        result = {
            'status': 'OLD_RESOURCE_HASH_MISMATCH',
            'expected_sha16': EXPECTED_SHA16,
            'actual_sha16': resource_hashes,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORT_DIR / 'old_machine_biview_ppmi_debug.json'
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('OLD_RESOURCE_HASH_MISMATCH')
        print(json.dumps(result, indent=2))
        raise SystemExit(2)

    legacy = load_legacy_builder(LEGACY_BUILDER)
    raw_sequences = legacy.parse_sequences(raw_inter)
    train_sequences = {}
    full_event_count = 0
    train_event_count = 0
    for user, seq in raw_sequences.items():
        full_event_count += len(seq)
        train = seq[:-2] if len(seq) >= 2 else []
        train_sequences[user] = train
        train_event_count += len(train)

    item_order = [str(x) for x in json.loads(st5_order.read_text(encoding='utf-8'))]
    raw_index_order = [str(x) for x in json.loads(raw_index.read_text(encoding='utf-8'))]
    st5_order_aligned = (item_order == raw_index_order)

    ppmi = legacy.build_ppmi(train_sequences, item_order, 5)
    ppmi.sum_duplicates()
    ppmi.sort_indices()

    trainonly_inter_path = resource_dir / 'Beauty.trainonly.inter.json'
    item_order_path = resource_dir / 'Beauty_item_id_order.json'
    try:
        trainonly_obj = json.loads(trainonly_inter_path.read_text(encoding='utf-8'))
        item_order_obj = json.loads(item_order_path.read_text(encoding='utf-8'))
    except Exception:
        trainonly_obj = None
        item_order_obj = None

    data_n = min(args.sample_first, len(ppmi.data))
    ind_n = min(args.sample_first, len(ppmi.indices))
    ptr_n = min(args.sample_first, len(ppmi.indptr))

    old_ppmi_hash = csr_all_hash(ppmi)
    if old_ppmi_hash != NEW_MACHINE_PPMI:
        classification = 'PPMI_VALUE_DIFFERENCE'
        interpretation = 'Old and new machines diverge before SVD: PPMI CSR values/indices/inputs or legacy builder context differ.'
    elif resource_hashes['Beauty_trainonly_cf_svd.npy'] != NEW_MACHINE_CF_SHA16:
        classification = 'TRUNCATEDSVD_NUMERICAL_ENVIRONMENT'
        interpretation = 'PPMI CSR is identical but TruncatedSVD output differs across machines/environments.'
    else:
        classification = 'FULL_MATCH_OR_UNEXPECTED'
        interpretation = 'PPMI and CF-SVD hashes do not show the expected mismatch pattern.'

    result = {
        'status': 'OK',
        'classification': classification,
        'interpretation': interpretation,
        'legacy_builder_path': str(LEGACY_BUILDER),
        'raw_inter_path': str(raw_inter),
        'raw_index_path': str(raw_index),
        'st5_order_path': str(st5_order),
        'st5_path': str(st5),
        'raw_inter_sha256': sha256_file(raw_inter),
        'raw_index_sha256': sha256_file(raw_index),
        'st5_order_file_sha256': sha256_file(st5_order),
        'st5_file_sha256': sha256_file(st5),
        'trainonly_inter_file_sha256': sha256_file(trainonly_inter_path),
        'item_order_file_sha256': sha256_file(item_order_path),
        'trainonly_inter_sha256': canonical_json_sha256(trainonly_obj) if trainonly_obj is not None else None,
        'item_order_sha256': canonical_json_sha256(item_order_obj) if item_order_obj is not None else None,
        'st5_order_aligned_with_raw_index': st5_order_aligned,
        'full_event_count': int(full_event_count),
        'train_event_count': int(train_event_count),
        'excluded_event_count': int(full_event_count - train_event_count),
        'expected_excluded_event_count': int(2 * len(raw_sequences)),
        'ppmi_shape': [int(ppmi.shape[0]), int(ppmi.shape[1])],
        'ppmi_nnz': int(ppmi.nnz),
        'sha256_ppmi_data': hashlib.sha256(ppmi.data.tobytes()).hexdigest(),
        'sha256_ppmi_indices': hashlib.sha256(ppmi.indices.tobytes()).hexdigest(),
        'sha256_ppmi_indptr': hashlib.sha256(ppmi.indptr.tobytes()).hexdigest(),
        'sha256_ppmi_csr_all': old_ppmi_hash,
        'new_machine_sha256_ppmi_csr_all': NEW_MACHINE_PPMI,
        'ppmi_matches_new_machine': old_ppmi_hash == NEW_MACHINE_PPMI,
        'first_20_data_values': [float(x) for x in ppmi.data[:data_n]],
        'first_20_indices': [int(x) for x in ppmi.indices[:ind_n]],
        'first_20_indptr': [int(x) for x in ppmi.indptr[:ptr_n]],
        'old_resource_hash_status': 'MATCH_EXPECTED' if old_resource_hash_status else 'OLD_RESOURCE_HASH_MISMATCH',
        'expected_old_resource_sha16': EXPECTED_SHA16,
        'actual_old_resource_sha16': resource_hashes,
        'old_resource_cf_svd_sha16': resource_hashes['Beauty_trainonly_cf_svd.npy'],
        'old_resource_cf_residual_sha16': resource_hashes['Beauty_cf_residual.npy'],
        'old_resource_semantic_base_sha16': resource_hashes['Beauty_semantic_base.npy'],
        'old_resource_semantic_residual_sha16': resource_hashes['Beauty_semantic_residual.npy'],
        'new_machine_cf_svd_sha16': NEW_MACHINE_CF_SHA16,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'sklearn_version': sklearn.__version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'threadpoolctl_info': threadpool_info() if threadpool_info is not None else None,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / 'old_machine_biview_ppmi_debug.json'
    md_path = REPORT_DIR / 'old_machine_biview_ppmi_debug_report.md'
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    md = f"""# Old Machine Legacy Biview PPMI Debug

## Old Resource Hash Status

`{result['old_resource_hash_status']}`

| File | Expected sha16 | Actual sha16 | Match |
|---|---:|---:|---:|
"""
    for name, expected in EXPECTED_SHA16.items():
        actual = resource_hashes[name]
        md += f"| {name} | {expected} | {actual} | {actual == expected} |\n"
    md += f"""
## Legacy Builder

`{LEGACY_BUILDER}`

## Input Paths and Hashes

| Input | Path | sha256 |
|---|---|---:|
| raw_inter | `{raw_inter}` | `{result['raw_inter_sha256']}` |
| raw_index | `{raw_index}` | `{result['raw_index_sha256']}` |
| st5_order | `{st5_order}` | `{result['st5_order_file_sha256']}` |
| st5 | `{st5}` | `{result['st5_file_sha256']}` |

## Split Summary

| Metric | Value |
|---|---:|
| full_event_count | {full_event_count} |
| train_event_count | {train_event_count} |
| excluded_event_count | {full_event_count - train_event_count} |
| expected_excluded_event_count | {2 * len(raw_sequences)} |
| st5_order_aligned_with_raw_index | {st5_order_aligned} |

## PPMI CSR

| Metric | Value |
|---|---:|
| shape | {result['ppmi_shape']} |
| nnz | {result['ppmi_nnz']} |
| sha256_ppmi_data | `{result['sha256_ppmi_data']}` |
| sha256_ppmi_indices | `{result['sha256_ppmi_indices']}` |
| sha256_ppmi_indptr | `{result['sha256_ppmi_indptr']}` |
| sha256_ppmi_csr_all | `{old_ppmi_hash}` |

## Old vs New Machine

| Item | Old machine | New machine | Match |
|---|---:|---:|---:|
| PPMI CSR all | `{old_ppmi_hash}` | `{NEW_MACHINE_PPMI}` | {old_ppmi_hash == NEW_MACHINE_PPMI} |
| CF-SVD sha16 | `{resource_hashes['Beauty_trainonly_cf_svd.npy']}` | `{NEW_MACHINE_CF_SHA16}` | {resource_hashes['Beauty_trainonly_cf_svd.npy'] == NEW_MACHINE_CF_SHA16} |

## Classification

`{classification}`

{interpretation}

## Conclusion

- If exact historical CHORD results are needed, migrate the old `.npy` resources directly.
- Legacy builder cross-machine bit-level rebuild is {'not supported because PPMI already differs' if classification == 'PPMI_VALUE_DIFFERENCE' else 'blocked by TruncatedSVD numerical/environment differences' if classification == 'TRUNCATEDSVD_NUMERICAL_ENVIRONMENT' else 'not determined by this run'}.

## Environment

```json
{json.dumps({'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'threadpoolctl_info': result['threadpoolctl_info']}, ensure_ascii=False, indent=2)}
```
"""
    md_path.write_text(md, encoding='utf-8')

    print('old_machine sha256_ppmi_csr_all', old_ppmi_hash)
    print('old_machine ppmi_nnz', ppmi.nnz)
    print('old_machine cf_svd sha16', resource_hashes['Beauty_trainonly_cf_svd.npy'])
    print('new_machine sha256_ppmi_csr_all =', NEW_MACHINE_PPMI)
    print('new_machine cf_svd sha16 =', NEW_MACHINE_CF_SHA16)
    print('classification', classification)
    print('json_report', json_path)
    print('markdown_report', md_path)


if __name__ == '__main__':
    main()


