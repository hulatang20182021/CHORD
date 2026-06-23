#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PROJECT = ROOT / 'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
EXP_BASE = PROJECT / 'results/pls_sd128_dpos_pcsc/order_ablation_cold_start'
CONDA = Path('/home/huangxin/anaconda3/bin/conda')
TIGER = ROOT / 'LETTER-TIGER'
TEST_WRAPPER = ROOT / 'component_relation_sid/scripts/run_letter_script_patience_override.py'
ST5_DIR = ROOT / 'component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def ratio_tag(r):
    return f"cold{int(round(float(r) * 100)):02d}"


def run(cmd, log, env, cwd=TIGER, quiet=False, stage='command'):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[stage] {stage}: START {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print('[run]', ' '.join(map(str, cmd)), flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.Popen([str(x) for x in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line); f.flush()
            if not quiet:
                print(line, end='', flush=True)
        code = p.wait()
    if code:
        raise RuntimeError(f'Command failed at {stage}; see {log}')
    print(f"[stage] {stage}: DONE {datetime.now().isoformat(timespec='seconds')}", flush=True)


def extract_metrics(raw_path, prefix=''):
    data = load_json(raw_path)
    mean = data.get('mean_results', data)
    out = {}
    for k, v in mean.items():
        kk = k.replace('hit@', 'HR@').replace('ndcg@', 'NDCG@')
        out[prefix + kk] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cold_seed', type=int, default=42)
    ap.add_argument('--cold_ratio', type=float, default=0.05)
    ap.add_argument('--orders', default='cf_first,sem_first')
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--num_beams', type=int, default=20)
    ap.add_argument('--run_suffix', default='order_ablation')
    ap.add_argument('--train_batch_size', type=int, default=256)
    ap.add_argument('--test_batch_size', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip_train', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT / 'scripts'), str(TIGER)])
    env['PYTHONUNBUFFERED'] = '1'
    env['WANDB_DISABLED'] = 'true'
    split_key = f"{args.dataset}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    logs = EXP_BASE / 'logs'
    runs = EXP_BASE / 'runs'

    for order in [x for x in args.orders.split(',') if x]:
        run([
            CONDA, 'run', '--no-capture-output', '-n', 'emotion_ml1m', 'python',
            PROJECT / 'scripts/order_ablation_build_assets.py',
            '--dataset', args.dataset, '--seed', args.seed, '--cold_seed', args.cold_seed,
            '--cold_ratio', args.cold_ratio, '--order', order, '--force'
        ], logs / f'{split_key}_{order}.build_assets.log', env, cwd=PROJECT, quiet=args.quiet, stage=f'{order}:build_assets')
        asset = load_json(EXP_BASE / 'assets' / split_key / order / 'asset_summary.json')
        run_name = f"{args.dataset}_{order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}_down{args.epochs}_beam{args.num_beams}_{args.run_suffix}"
        run_dir = runs / run_name
        ckpt = run_dir / 'checkpoints'
        metrics_path = run_dir / 'metrics_warm.json'
        if metrics_path.exists() and not args.force:
            print(f'[skip] completed {run_name}')
            continue
        if run_dir.exists() and any(run_dir.iterdir()) and not (args.force or args.skip_train):
            raise SystemExit(f'Refusing non-empty incomplete run: {run_dir}. Use --force or --skip_train.')
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json({'run_name': run_name, 'method': 'CHORD', 'order': order, 'asset': asset, 'epochs': args.epochs, 'beam': args.num_beams, 'seed': args.seed}, run_dir / 'run_config.json')
        if not args.skip_train:
            run([
                CONDA, 'run', '--no-capture-output', '-n', 'emotion_ml1m', 'python',
                PROJECT / 'scripts/order_ablation_static_intersection_downstream_finetune.py',
                '--output_dir', ckpt,
                '--dataset', asset['train_alias'], '--data_path', EXP_BASE / 'data',
                '--per_device_batch_size', args.train_batch_size,
                '--learning_rate', '5e-4', '--epochs', args.epochs,
                '--gradient_accumulation_steps', '1', '--logging_step', '50',
                '--train_data_sample_num', '-1', '--valid_prompt_sample_num', '1',
                '--save_and_eval_strategy', 'epoch', '--index_file', '.index.json',
                '--temperature', '1.0', '--seed', args.seed,
                '--index', asset['index_json'], '--item_order', asset['item_order'],
                '--cf_emb', asset['cf_emb'], '--sem_emb', ST5_DIR / f"{args.dataset}_st5_rqvae_input_embeddings.npy",
                '--cf_res', asset['cf_res'], '--sem_base', asset['sem_base'], '--sem_res_raw', asset['sem_res_raw'],
                '--chord_order', order, '--pcsc_aux', '--pcsc_max_factor', '1.0', '--pcsc_schedule_type', 'warmup_hold_decay',
                '--lambda_cf', '1.0', '--lambda_cfres', '1.0', '--lambda_base', '1.0', '--lambda_res', '1.0', '--lambda_comp', '1.0',
                '--training_metrics', run_dir / 'training_metrics.jsonl', '--run_summary', run_dir / 'run_summary.json'
            ], logs / f'{run_name}.train.log', env, cwd=TIGER, quiet=args.quiet, stage=f'{run_name}:train')
        # warm/full test
        warm_raw = run_dir / 'warm_metrics_raw.json'
        run([
            CONDA, 'run', '--no-capture-output', '-n', 'emotion_ml1m', 'python', TEST_WRAPPER, './test.py',
            '--gpu_id', '0', '--ckpt_path', ckpt, '--dataset', asset['train_alias'], '--data_path', EXP_BASE / 'data',
            '--results_file', warm_raw, '--test_batch_size', args.test_batch_size, '--num_beams', args.num_beams,
            '--sample_num', '-1', '--test_prompt_ids', '0', '--index_file', '.index.json',
            '--metrics', 'hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10', '--seed', args.seed
        ], logs / f'{run_name}.warm_eval.log', env, cwd=TIGER, quiet=args.quiet, stage=f'{run_name}:warm_eval')
        warm_metrics = extract_metrics(warm_raw)
        save_json(warm_metrics, run_dir / 'metrics_warm.json')
        # strict cold exact
        cold_raw = run_dir / 'cold_exact_metrics_raw.json'
        run([
            CONDA, 'run', '--no-capture-output', '-n', 'emotion_ml1m', 'python', TEST_WRAPPER, './test.py',
            '--gpu_id', '0', '--ckpt_path', ckpt, '--dataset', asset['eval_alias'], '--data_path', EXP_BASE / 'data',
            '--results_file', cold_raw, '--test_batch_size', args.test_batch_size, '--num_beams', args.num_beams,
            '--sample_num', '-1', '--test_prompt_ids', '0', '--index_file', '.index.json',
            '--metrics', 'hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10', '--seed', args.seed
        ], logs / f'{run_name}.cold_exact_eval.log', env, cwd=TIGER, quiet=args.quiet, stage=f'{run_name}:cold_exact_eval')
        cold_metrics = extract_metrics(cold_raw)
        save_json(cold_metrics, run_dir / 'metrics_cold_exact.json')
        print(json.dumps({'run_name': run_name, 'warm': warm_metrics, 'cold_exact': cold_metrics}, indent=2), flush=True)

if __name__ == '__main__':
    main()
