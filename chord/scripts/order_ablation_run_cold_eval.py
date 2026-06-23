#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PROJECT = ROOT / 'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
EXP_BASE = PROJECT / 'results/pls_sd128_dpos_pcsc/order_ablation_cold_start'
CONDA = Path('/home/huangxin/anaconda3/bin/conda')
TIGER = ROOT / 'LETTER-TIGER'


def ratio_tag(r): return f"cold{int(round(float(r)*100)):02d}"
def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def run(cmd, log, env, cwd=PROJECT):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(map(str, cmd)), flush=True)
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.Popen([str(x) for x in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line); f.flush(); print(line, end='', flush=True)
        code = p.wait()
    if code:
        raise RuntimeError(f'Command failed; see {log}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cold_seed', type=int, default=42)
    ap.add_argument('--cold_ratio', type=float, default=0.05)
    ap.add_argument('--orders', default='cf_first,sem_first')
    ap.add_argument('--epsilons', default='0.1,0.5,1.0')
    ap.add_argument('--prefix_lens', default='3,2,1')
    ap.add_argument('--gpu', default='2')
    ap.add_argument('--export_beams', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--num_beams', type=int, default=20)
    ap.add_argument('--run_suffix', default='order_ablation')
    ap.add_argument('--test_batch_size', type=int, default=8)
    args=ap.parse_args()
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    env['PYTHONPATH']=os.pathsep.join([str(PROJECT/'scripts'), str(TIGER)]); env['PYTHONUNBUFFERED']='1'; env['WANDB_DISABLED']='true'
    split_key=f"{args.dataset}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    logs=EXP_BASE/'logs'
    for order in [x for x in args.orders.split(',') if x]:
        asset_path=EXP_BASE/'assets'/split_key/order/'asset_summary.json'
        if not asset_path.exists():
            raise FileNotFoundError(f'asset summary not found: {asset_path}')
        asset=load_json(asset_path)
        run_name=f"{args.dataset}_{order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}_down{args.epochs}_beam{args.num_beams}_{args.run_suffix}"
        ckpt=EXP_BASE/'runs'/run_name/'checkpoints'
        if not ckpt.exists():
            raise FileNotFoundError(f'checkpoint not found: {ckpt}')
        out=EXP_BASE/'tiger_style_eval'/f"{args.dataset}_{order}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
        out.mkdir(parents=True, exist_ok=True)
        beams=out/'warm_beams.jsonl'
        run([
            CONDA,'run','--no-capture-output','-n','emotion_ml1m','python', PROJECT/'scripts/export_cold_start_warm_beams.py',
            '--dataset',args.dataset,'--split_key',split_key,'--checkpoint',ckpt,'--warm_index',asset['warm_only_index'],
            '--eval_alias',asset['eval_alias'],'--data_root',EXP_BASE/'data','--num_beams',args.export_beams,
            '--test_batch_size',args.test_batch_size,'--gpu','0','--output',beams,'--seed',args.seed
        ], logs/f'{run_name}.export_beams.log', env, cwd=PROJECT)
        for prefix_len in [int(x) for x in args.prefix_lens.split(',') if x]:
            eps = args.epsilons if prefix_len == 3 else '1.0'
            run([
                CONDA,'run','--no-capture-output','-n','emotion_ml1m','python', PROJECT/'scripts/eval_cold_start_tiger_style.py',
                '--beams_jsonl',beams,'--cold_prefix_map',EXP_BASE/'assets'/split_key/order/f'cold_prefix{prefix_len}_to_items.json',
                '--cold_item_to_sid',EXP_BASE/'assets'/split_key/order/'cold_item_to_sid.json',
                '--warm_items',PROJECT/'results/pls_sd128_dpos_pcsc/cold_start'/split_key/'warm_items.json',
                '--cold_items',PROJECT/'results/pls_sd128_dpos_pcsc/cold_start'/split_key/'cold_items.json',
                '--k_list','1,5,10','--epsilons',eps,'--prefix_len',prefix_len,
                '--output_metrics',out/f'metrics_prefix{prefix_len}.json','--output_details',out/f'details_prefix{prefix_len}.jsonl'
            ], logs/f'{run_name}.eval_prefix{prefix_len}.log', env, cwd=PROJECT)

if __name__=='__main__': main()
