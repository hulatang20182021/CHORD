#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

ROOT=Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PROJECT=ROOT/'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
EXP_BASE=PROJECT/'results/pls_sd128_dpos_pcsc/order_ablation_cold_start'
COLD_BASE=PROJECT/'results/pls_sd128_dpos_pcsc/cold_start'

def save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def ratio_tag(r): return f"cold{int(round(float(r)*100)):02d}"

def locate_letter_index(dataset):
    candidates=[
        ROOT/'data'/dataset/f'{dataset}.index.json',
        ROOT/'data'/dataset/'index.json',
        ROOT/'data'/f'{dataset}.index.json',
    ]
    for p in candidates:
        if p.exists(): return p
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='Beauty')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--cold_seed',type=int,default=42)
    ap.add_argument('--cold_ratio',type=float,default=0.05)
    args=ap.parse_args()
    out=EXP_BASE/'letter_baseline'
    out.mkdir(parents=True,exist_ok=True)
    idx=locate_letter_index(args.dataset)
    if idx is None:
        save_json({'status':'missing_resources','message':'LETTER baseline resources not found. Need user to provide LETTER SID/index path.','searched':['data/<dataset>/<dataset>.index.json','data/<dataset>/index.json','data/<dataset>.index.json']}, out/'LETTER_RESOURCE_MISSING.json')
        print('LETTER baseline resources not found. Need user to provide LETTER SID/index path.')
        return
    split_key=f"{args.dataset}_{ratio_tag(args.cold_ratio)}_seed{args.seed}_cseed{args.cold_seed}"
    split_dir=COLD_BASE/split_key
    warm=load_json(split_dir/'warm_items.json'); cold=load_json(split_dir/'cold_items.json')
    index=load_json(idx)
    missing=[str(i) for i in list(warm)+list(cold) if str(i) not in index]
    if missing:
        save_json({'status':'index_coverage_failed','index':str(idx),'missing_count':len(missing),'sample_missing':missing[:20]}, out/'LETTER_RESOURCE_MISSING.json')
        print(f'LETTER index coverage failed: missing {len(missing)} items')
        return
    save_json({'status':'found_but_training_not_implemented','index':str(idx),'message':'LETTER index was located, but matched cold-start train wrapper is not enabled automatically in this script to avoid using an incompatible tokenizer.'}, out/'LETTER_RESOURCE_MISSING.json')
    print(f'LETTER index found at {idx}, but automatic training is disabled pending tokenizer compatibility check.')
if __name__=='__main__': main()
