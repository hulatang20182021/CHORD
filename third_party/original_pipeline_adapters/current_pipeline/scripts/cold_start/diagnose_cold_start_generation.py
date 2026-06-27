#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _PathForImport
PIPELINE_SCRIPT_ROOT = _PathForImport(__file__).resolve().parents[1]
TIGER_ROOT = _PathForImport("/home/huangxin/llmNrec/Letter/LETTER-master/LETTER-TIGER")
for _p in (PIPELINE_SCRIPT_ROOT, TIGER_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
TIGER = ROOT / 'LETTER-TIGER'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TIGER) not in sys.path:
    sys.path.insert(0, str(TIGER))

from project_paths import NEW_BASE, save_json  # noqa: E402
from collator import TestCollator  # noqa: E402
from data import SeqRecDataset  # noqa: E402
from generation_trie import Trie  # noqa: E402
from utils import prefix_allowed_tokens_fn, set_seed  # noqa: E402

RESULT_BASE = NEW_BASE / 'results/pls_sd128_dpos_pcsc'
COLD_BASE = RESULT_BASE / 'cold_start'
REPORT_DIR = COLD_BASE / 'reports'

DEFAULT_STATIC_RUN = 'Beauty_coldstart_plssd128_c4_dpos_cold05_seed42_cseed42'
DEFAULT_TRAIN_RUN = 'Beauty_coldstart_plssd128_c4_dpos_cold05_seed42_cseed42_warm_train_hard_pcsc_down60_beam20_cold5'


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def prefix_tuple(index: Dict[str, List[str]], item: str, n: int) -> Tuple[str, ...]:
    return tuple(index[str(item)][:n])


def hit_ndcg(rank: int | None, k: int) -> Tuple[float, float]:
    if rank is None or rank > k:
        return 0.0, 0.0
    return 1.0, 1.0 / math.log2(rank + 1)


def metrics_from_ranked(ranked: Sequence[str], target: str, ks: Sequence[int]) -> Dict[str, float]:
    seen, used = [], set()
    for x in ranked:
        if x in used:
            continue
        used.add(x)
        seen.append(x)
    pos = None
    for i, x in enumerate(seen, start=1):
        if x == target:
            pos = i
            break
    out = {}
    for k in ks:
        h, n = hit_ndcg(pos, k)
        out[f'HR@{k}'] = h
        out[f'NDCG@{k}'] = n
    return out


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    return {k: float(sum(r.get(k, 0.0) for r in rows) / len(rows)) for k in keys}


def coverage_stats(index, warm_items, cold_targets):
    warm_prefixes = {n: {prefix_tuple(index, i, n) for i in warm_items} for n in [1, 2, 3, 4]}
    out = {}
    for n, label in [(1, 'c1'), (2, 'c1c2'), (3, 'c1c2c3'), (4, 'full_sid')]:
        covered = sum(1 for i in cold_targets if prefix_tuple(index, i, n) in warm_prefixes[n])
        out[f'cold_target_{label}_in_warm_count'] = int(covered)
        out[f'cold_target_{label}_in_warm_ratio'] = float(covered / max(len(cold_targets), 1))
    return out


def build_args(eval_alias: str, data_root: Path, sample_num: int, batch_size: int, seed: int):
    return SimpleNamespace(
        seed=seed,
        base_model='./ckpt/TIGER',
        output_dir='./ckpt',
        data_path=str(data_root),
        tasks='seqrec',
        dataset=eval_alias,
        index_file='.index.json',
        max_his_len=20,
        add_prefix=False,
        his_sep=', ',
        only_train_response=False,
        train_prompt_sample_num='1',
        train_data_sample_num='-1',
        valid_prompt_id=0,
        sample_valid=True,
        valid_prompt_sample_num=2,
        filter_items=True,
        results_file='unused.json',
        test_batch_size=batch_size,
        num_beams=100,
        sample_num=sample_num,
        gpu_id=0,
        test_prompt_ids='0',
        metrics='hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10',
        test_task='SeqRec',
    )


def decode_generation(model, tokenizer, dataset, args, all_items, device, beam_size: int, batch_size: int):
    collator = TestCollator(args, tokenizer)
    trie = Trie([[0] + tokenizer.encode(candidate) for candidate in all_items])
    prefix_fn = prefix_allowed_tokens_fn(trie)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=0)
    rows = []
    started = time.time()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            inputs = batch[0].to(device)
            targets = batch[1]
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_fn,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )
            pred_texts = [x.strip().replace(' ', '') for x in tokenizer.batch_decode(output['sequences'], skip_special_tokens=True)]
            scores = [float(x) for x in output['sequences_scores'].detach().cpu().tolist()]
            for b, target in enumerate(targets):
                lo, hi = b * beam_size, (b + 1) * beam_size
                rows.append({'target': target.strip().replace(' ', ''), 'predictions': pred_texts[lo:hi], 'scores': scores[lo:hi]})
            if step == 1 or step % 10 == 0:
                elapsed = time.time() - started
                print(f'[diagnose] generated batch {step}/{len(loader)} users={len(rows)} elapsed={elapsed:.1f}s', flush=True)
    return rows


def prediction_distribution(rows, all_items, cold_sids, warm_sids, ks):
    out = {}
    total_by_k = {k: 0 for k in ks}
    cold_by_k = {k: 0 for k in ks}
    warm_by_k = {k: 0 for k in ks}
    invalid_by_k = {k: 0 for k in ks}
    for r in rows:
        for k in ks:
            top = r['predictions'][:k]
            total_by_k[k] += len(top)
            cold_by_k[k] += sum(1 for p in top if p in cold_sids)
            warm_by_k[k] += sum(1 for p in top if p in warm_sids)
            invalid_by_k[k] += sum(1 for p in top if p not in all_items)
    for k in ks:
        denom = max(total_by_k[k], 1)
        out[f'top{k}_cold_prediction_ratio'] = float(cold_by_k[k] / denom)
        out[f'top{k}_warm_prediction_ratio'] = float(warm_by_k[k] / denom)
        out[f'top{k}_invalid_sid_ratio'] = float(invalid_by_k[k] / denom)
        out[f'top{k}_cold_prediction_count'] = int(cold_by_k[k])
        out[f'top{k}_warm_prediction_count'] = int(warm_by_k[k])
        out[f'top{k}_invalid_sid_count'] = int(invalid_by_k[k])
    out['generated_user_count'] = len(rows)
    return out


def prefix_hit(rows, sid_to_tokens, ks):
    metric_rows = []
    for r in rows:
        target_tokens = sid_to_tokens.get(r['target'])
        if target_tokens is None:
            continue
        one = {}
        for n in [1, 2, 3, 4]:
            target_prefix = target_tokens[:n]
            pred_prefixes = []
            for p in r['predictions']:
                toks = sid_to_tokens.get(p)
                pred_prefixes.append(toks[:n] if toks is not None else None)
            pos = None
            for i, pfx in enumerate(pred_prefixes, start=1):
                if pfx == target_prefix:
                    pos = i
                    break
            label = f'prefix{n}' if n < 4 else 'full_sid'
            for k in ks:
                h, nd = hit_ndcg(pos, k)
                one[f'{label}_hit@{k}'] = h
                one[f'{label}_ndcg@{k}'] = nd
        metric_rows.append(one)
    return mean_dict(metric_rows)


def cold_only_oracle(rows, sid_to_tokens, cold_sids, ks=(10,)):
    cold_by_prefix = {n: defaultdict(list) for n in [1, 2, 3]}
    for sid in cold_sids:
        toks = sid_to_tokens[sid]
        for n in [1, 2, 3]:
            cold_by_prefix[n][toks[:n]].append(sid)
    metric_rows = []
    for r in rows:
        scored = {}
        for rank, (pred, score) in enumerate(zip(r['predictions'], r['scores']), start=1):
            toks = sid_to_tokens.get(pred)
            if toks is None:
                continue
            for n, bonus in [(3, 0.003), (2, 0.002), (1, 0.001)]:
                for cand in cold_by_prefix[n].get(toks[:n], []):
                    val = float(score) + bonus - rank * 1e-6
                    if cand not in scored or val > scored[cand]:
                        scored[cand] = val
        ranked = [x for x, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)]
        metric_rows.append(metrics_from_ranked(ranked, r['target'], ks))
    return mean_dict(metric_rows)


def prefix_expansion_eval(rows, sid_to_tokens, cold_sids, epsilons):
    cold_by_prefix3 = defaultdict(list)
    for sid in cold_sids:
        cold_by_prefix3[sid_to_tokens[sid][:3]].append(sid)
    out = {}
    for eps in epsilons:
        metric_rows = []
        max_cold = max(1, int(math.ceil(10 * eps))) if eps > 0 else 0
        for r in rows:
            ranked, used = [], set()
            cold_added = 0
            for pred in r['predictions']:
                if pred not in used:
                    ranked.append(pred)
                    used.add(pred)
                toks = sid_to_tokens.get(pred)
                if toks is None:
                    continue
                for cand in cold_by_prefix3.get(toks[:3], []):
                    if cold_added >= max_cold:
                        break
                    if cand in used:
                        continue
                    ranked.append(cand)
                    used.add(cand)
                    cold_added += 1
                if len(ranked) >= 10 and cold_added >= max_cold:
                    break
            metric_rows.append(metrics_from_ranked(ranked, r['target'], [10]))
        mean = mean_dict(metric_rows)
        tag = str(eps).replace('.', 'p')
        out[f'epsilon_{tag}_HR@10'] = mean.get('HR@10', 0.0)
        out[f'epsilon_{tag}_NDCG@10'] = mean.get('NDCG@10', 0.0)
    return out


def write_report(result, md_path: Path):
    lines = ['# Cold-start Generation Diagnosis', '']
    lines.append(f"- static run: `{result['static_run']}`")
    lines.append(f"- train run: `{result['train_run']}`")
    lines.append(f"- checkpoint: `{result['checkpoint']}`")
    lines.append(f"- generated users: {result['generation_distribution']['generated_user_count']}")
    sections = [
        ('A. Data Checks', result['data_checks']),
        ('B. Training Coverage', result['training_coverage']),
        ('C. Generation Distribution', result['generation_distribution']),
        ('D. Prefix Hit', result['prefix_hit']),
        ('E. Cold-only Oracle', result['cold_only_oracle']),
        ('F. Prefix Expansion Evaluation', result['prefix_expansion_eval']),
    ]
    for title, data in sections:
        lines += ['', f'## {title}', '', '| Metric | Value |', '|---|---:|']
        for k, v in data.items():
            if isinstance(v, float):
                lines.append(f'| {k} | {v:.8f} |')
            else:
                lines.append(f'| {k} | {v} |')
    lines += ['', '## Judgment', '']
    for item in result['judgment']:
        lines.append(f'- {item}')
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Diagnose strict cold-start generation without retraining.')
    ap.add_argument('--static_run', default=DEFAULT_STATIC_RUN)
    ap.add_argument('--train_run', default=DEFAULT_TRAIN_RUN)
    ap.add_argument('--sample_users', type=int, default=1024, help='Use -1 for all cold eval users.')
    ap.add_argument('--beam_size', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    manifests = list(COLD_BASE.glob('*/manifest.json'))
    manifest_path = None
    for path in manifests:
        m = load_json(path)
        if m.get('static_run') == args.static_run:
            manifest_path = path
            manifest = m
            break
    if manifest_path is None:
        raise FileNotFoundError(f'No manifest found for static_run={args.static_run} under {COLD_BASE}')
    split_dir = manifest_path.parent
    run_dir = COLD_BASE / 'runs' / args.train_run
    ckpt = run_dir / 'checkpoints'
    cold_eval_metrics = run_dir / 'cold_eval_metrics.json'
    if not ckpt.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')

    warm_items = [str(x) for x in load_json(split_dir / 'warm_items.json')]
    cold_items = [str(x) for x in load_json(split_dir / 'cold_items.json')]
    cold_set = set(cold_items)
    cold_eval = {str(k): [str(x) for x in v] for k, v in load_json(split_dir / 'cold_eval.inter.json').items()}
    index_path = Path(manifest['index_json'])
    raw_path = Path(manifest['raw_codes'])
    index = load_json(index_path)
    raw_codes = load_json(raw_path) if raw_path.exists() else {}
    metrics = load_json(cold_eval_metrics) if cold_eval_metrics.exists() else {}

    targets = [seq[-1] for seq in cold_eval.values() if seq]
    hist_cold_leak = sum(1 for seq in cold_eval.values() if any(x in cold_set for x in seq[:-1]))
    target_cold = sum(1 for x in targets if x in cold_set)
    target_missing = sum(1 for x in targets if x not in index)

    sid_to_items = defaultdict(list)
    for item, sid in index.items():
        sid_to_items[''.join(sid)].append(str(item))
    dup_count = sum(len(v) - 1 for v in sid_to_items.values() if len(v) > 1)
    sid_to_tokens = {''.join(sid): tuple(sid) for sid in index.values()}
    item_to_sid = {str(item): ''.join(sid) for item, sid in index.items()}
    cold_sids = {item_to_sid[i] for i in cold_items}
    warm_sids = {item_to_sid[i] for i in warm_items}
    all_sids = set(item_to_sid.values())

    data_checks = {
        'cold_item_count': len(cold_items),
        'cold_eval_users': len(cold_eval),
        'target_cold_coverage_count': target_cold,
        'target_cold_coverage_ratio': target_cold / max(len(targets), 1),
        'history_cold_leakage_count': hist_cold_leak,
        'cold_target_missing_in_index_count': target_missing,
        'duplicate_sid_count': int(dup_count),
        'raw_codes_count': len(raw_codes),
        'reported_cold_eval_hit@10': metrics.get('mean_results', {}).get('hit@10'),
        'reported_cold_eval_ndcg@10': metrics.get('mean_results', {}).get('ndcg@10'),
    }
    training_coverage = coverage_stats(index, warm_items, targets)

    data_root = Path(manifest['data_root'])
    eval_alias = manifest['eval_alias']
    sample_num = args.sample_users if args.sample_users and args.sample_users > 0 else -1
    ds_args = build_args(eval_alias, data_root, sample_num, args.batch_size, args.seed)

    tokenizer = T5Tokenizer.from_pretrained(str(TIGER / 'ckpt/TIGER'), model_max_length=512, local_files_only=True)
    token_dataset = SeqRecDataset(ds_args, mode='train', prompt_sample_num=1, sample_num=-1)
    tokenizer.add_tokens(token_dataset.get_new_tokens())
    test_dataset = SeqRecDataset(ds_args, mode='test', sample_num=sample_num)
    test_dataset.set_prompt(0)
    all_items = test_dataset.get_all_items()
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt), low_cpu_mem_usage=True).to(device)

    rows = decode_generation(model, tokenizer, test_dataset, ds_args, all_items, device, args.beam_size, args.batch_size)
    dist = prediction_distribution(rows, all_sids, cold_sids, warm_sids, [10, 20, 50, 100])
    p_hit = prefix_hit(rows, sid_to_tokens, [10, 20, 50])
    oracle = cold_only_oracle(rows, sid_to_tokens, cold_sids, [10])
    expansion = prefix_expansion_eval(rows, sid_to_tokens, cold_sids, [0.05, 0.1, 0.2, 0.5, 1.0])

    is_data_bug = data_checks['history_cold_leakage_count'] > 0 or data_checks['cold_target_missing_in_index_count'] > 0 or data_checks['duplicate_sid_count'] > 0 or data_checks['target_cold_coverage_ratio'] < 0.999
    cold_beam_ratio = dist.get('top20_cold_prediction_ratio', 0.0)
    prefix3_signal = p_hit.get('prefix3_hit@50', 0.0)
    judgment = []
    if is_data_bug:
        judgment.append('数据/index 存在异常，需要先修复 split/index 后再解释模型结果。')
    else:
        judgment.append('未发现 target/index/duplicate/history 泄露级别的数据 bug；主要问题是生成分布偏向 warm SID。')
    judgment.append(f"cold items in top20 beam ratio = {cold_beam_ratio:.8f}；{'几乎不进入 beam' if cold_beam_ratio < 1e-6 else '已经进入 beam，但比例偏低'}。")
    judgment.append(f"prefix3 hit@50 = {prefix3_signal:.8f}；该值用于判断是否有可被 prefix expansion 利用的弱信号。")
    judgment.append('strict exact-SID cold-start 当前不建议作为论文主 cold-start 结果，除非后续引入 cold item scoring/bridge 后指标不再退化为全 0。')
    judgment.append('建议补 content reranker、prefix expansion 或 soft/content bridge；其中 prefix expansion 只能作为补充评估，不应替代 exact-SID 主指标。')

    result = {
        'static_run': args.static_run,
        'train_run': args.train_run,
        'checkpoint': str(ckpt),
        'manifest_path': str(manifest_path),
        'index_path': str(index_path),
        'raw_codes_path': str(raw_path),
        'cold_eval_metrics_path': str(cold_eval_metrics),
        'sample_users_requested': args.sample_users,
        'beam_size': args.beam_size,
        'data_checks': data_checks,
        'training_coverage': training_coverage,
        'generation_distribution': dist,
        'prefix_hit': p_hit,
        'cold_only_oracle': oracle,
        'prefix_expansion_eval': expansion,
        'judgment': judgment,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / 'cold_start_generation_diagnosis.json'
    md_path = REPORT_DIR / 'cold_start_generation_diagnosis.md'
    save_json(result, json_path)
    write_report(result, md_path)
    print(json.dumps({'json': str(json_path), 'markdown': str(md_path), 'judgment': judgment}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()


