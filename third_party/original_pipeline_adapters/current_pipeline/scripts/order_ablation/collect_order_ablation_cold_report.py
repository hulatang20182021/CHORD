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
from pathlib import Path

PROJECT = Path('/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline')
DEFAULT_EXP = PROJECT / 'results/pls_sd128_dpos_pcsc/order_ablation_cold_start'


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception:
        return None


def ratio_tag(r): return f"cold{int(round(float(r)*100)):02d}"

def metric(m, k):
    if not m: return ''
    return m.get(k, '')

def write_tsv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write('\t'.join(cols)+'\n')
        for r in rows:
            f.write('\t'.join(str(r.get(c,'')) for c in cols)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp_base', default=str(DEFAULT_EXP))
    ap.add_argument('--dataset', default='Beauty')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cold_seed', type=int, default=42)
    ap.add_argument('--cold_ratio', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--num_beams', type=int, default=20)
    ap.add_argument('--run_suffix', default='order_ablation')
    args=ap.parse_args()
    exp=Path(args.exp_base)
    rep=exp/'reports'; rep.mkdir(parents=True, exist_ok=True)
    rt=ratio_tag(args.cold_ratio)
    orders=['cf_first','sem_first']
    raw=[]; tiger=[]; letter=[]
    for order in orders:
        run_name=f"{args.dataset}_{order}_{rt}_seed{args.seed}_cseed{args.cold_seed}_down{args.epochs}_beam{args.num_beams}_{args.run_suffix}"
        run_dir=exp/'runs'/run_name
        warm=load_json(run_dir/'metrics_warm.json')
        cold=load_json(run_dir/'metrics_cold_exact.json')
        raw.append({
            'Method':'CHORD','Order':order,'Dataset':args.dataset,'Seed':args.seed,
            'Warm HR@1':metric(warm,'HR@1'),'Warm HR@5':metric(warm,'HR@5'),'Warm HR@10':metric(warm,'HR@10'),
            'Warm NDCG@1':metric(warm,'NDCG@1'),'Warm NDCG@5':metric(warm,'NDCG@5'),'Warm NDCG@10':metric(warm,'NDCG@10'),
            'Exact Cold HR@1':metric(cold,'HR@1'),'Exact Cold HR@5':metric(cold,'HR@5'),'Exact Cold HR@10':metric(cold,'HR@10'),
            'Exact Cold NDCG@1':metric(cold,'NDCG@1'),'Exact Cold NDCG@5':metric(cold,'NDCG@5'),'Exact Cold NDCG@10':metric(cold,'NDCG@10'),
            'Run Dir':str(run_dir),
        })
        eval_dir=exp/'tiger_style_eval'/f"{args.dataset}_{order}_{rt}_seed{args.seed}_cseed{args.cold_seed}"
        for prefix in [1,2,3]:
            m=load_json(eval_dir/f'metrics_prefix{prefix}.json')
            if not m: continue
            eps = ['0.1','0.5','1'] if prefix==3 else ['1']
            for e in eps:
                tiger.append({
                    'Method':'CHORD','Order':order,'Dataset':args.dataset,'Seed':args.seed,
                    'Prefix Len':prefix,'Epsilon':e,
                    'HR@1':m.get(f'epsilon_{e}_HR@1',''),'HR@5':m.get(f'epsilon_{e}_HR@5',''),'HR@10':m.get(f'epsilon_{e}_HR@10',''),
                    'NDCG@1':m.get(f'epsilon_{e}_NDCG@1',''),'NDCG@5':m.get(f'epsilon_{e}_NDCG@5',''),'NDCG@10':m.get(f'epsilon_{e}_NDCG@10',''),
                })
    # LETTER rows if present.
    ldir=exp/'letter_baseline'
    if ldir.exists():
        for p in ldir.glob('**/*.json'):
            if p.name.startswith('metrics'):
                m=load_json(p) or {}
                letter.append({'Method':'LETTER','Dataset':args.dataset,'Seed':args.seed,'Setting':p.stem,'Epsilon':m.get('epsilon',''),'HR@10':m.get('HR@10',''),'NDCG@10':m.get('NDCG@10',''),'Path':str(p)})
    raw_cols=['Method','Order','Dataset','Seed','Warm HR@1','Warm HR@5','Warm HR@10','Warm NDCG@1','Warm NDCG@5','Warm NDCG@10','Exact Cold HR@1','Exact Cold HR@5','Exact Cold HR@10','Exact Cold NDCG@1','Exact Cold NDCG@5','Exact Cold NDCG@10','Run Dir']
    tiger_cols=['Method','Order','Dataset','Seed','Prefix Len','Epsilon','HR@1','HR@5','HR@10','NDCG@1','NDCG@5','NDCG@10']
    letter_cols=['Method','Dataset','Seed','Setting','Epsilon','HR@10','NDCG@10','Path']
    write_tsv(rep/'order_ablation_cold_raw.tsv', raw, raw_cols)
    write_tsv(rep/'order_ablation_cold_tiger.tsv', tiger, tiger_cols)
    write_tsv(rep/'order_ablation_cold_letter.tsv', letter, letter_cols)
    write_tsv(rep/'order_ablation_cold_summary.tsv', raw + tiger + letter, sorted(set(raw_cols+tiger_cols+letter_cols)))

    def find_order(rows, order, field):
        for r in rows:
            if r.get('Order')==order:
                try: return float(r.get(field,''))
                except Exception: return None
        return None
    cf_warm=find_order(raw,'cf_first','Warm HR@10'); sem_warm=find_order(raw,'sem_first','Warm HR@10')
    cf_cold=find_order(raw,'cf_first','Exact Cold HR@10'); sem_cold=find_order(raw,'sem_first','Exact Cold HR@10')
    def tval(order,prefix,eps,field='HR@10'):
        for r in tiger:
            if r.get('Order')==order and str(r.get('Prefix Len'))==str(prefix) and str(r.get('Epsilon'))==str(eps):
                try: return float(r.get(field,''))
                except Exception: return None
        return None
    lines=['# CHORD Order Ablation + LETTER Cold-start Baseline Report','',f'- Dataset: `{args.dataset}`','- Method name: CHORD: Consensus and Hierarchical Orthogonal Residual Decoupling for Generative Recommendation',f'- Seed/cold seed/cold ratio: `{args.seed}` / `{args.cold_seed}` / `{args.cold_ratio}`','']
    lines += ['## CHORD Order Ablation Main Table','', '| Method | Order | Warm HR@10 | Warm NDCG@10 | Exact Cold HR@10 | Exact Cold NDCG@10 |','|---|---|---:|---:|---:|---:|']
    for r in raw:
        lines.append(f"| {r['Method']} | {r['Order']} | {r.get('Warm HR@10','')} | {r.get('Warm NDCG@10','')} | {r.get('Exact Cold HR@10','')} | {r.get('Exact Cold NDCG@10','')} |")
    lines += ['', '## TIGER-style / Diagnostic Cold Table','', '| Method | Order | Prefix Len | Epsilon | HR@10 | NDCG@10 |','|---|---|---:|---:|---:|---:|']
    for r in tiger:
        lines.append(f"| {r['Method']} | {r['Order']} | {r['Prefix Len']} | {r['Epsilon']} | {r.get('HR@10','')} | {r.get('NDCG@10','')} |")
    lines += ['', '## LETTER Matched Cold-start Baseline','']
    if letter:
        lines += ['| Method | Setting | Epsilon | HR@10 | NDCG@10 |','|---|---|---:|---:|---:|']
        for r in letter:
            lines.append(f"| LETTER | {r['Setting']} | {r.get('Epsilon','')} | {r.get('HR@10','')} | {r.get('NDCG@10','')} |")
    else:
        lines.append('LETTER cold-start baseline not completed because resource path is missing or the run has not been executed yet.')
    lines += ['', '## Automatic Judgement','']
    if sem_warm is not None and cf_warm is not None:
        lines.append(f"- sem_first warm/full {'improves' if sem_warm>cf_warm else 'does not improve'} HR@10 versus cf_first ({sem_warm:.6f} vs {cf_warm:.6f}).")
    if sem_cold is not None and cf_cold is not None:
        lines.append(f"- sem_first strict exact cold {'improves' if sem_cold>cf_cold else 'does not improve'} HR@10 versus cf_first ({sem_cold:.6f} vs {cf_cold:.6f}).")
    cf_p2=tval('cf_first',2,'1'); sem_p2=tval('sem_first',2,'1')
    if cf_p2 is not None and sem_p2 is not None:
        lines.append(f"- sem_first prefix2 diagnostic {'improves' if sem_p2>cf_p2 else 'does not improve'} cold-start HR@10 ({sem_p2:.6f} vs {cf_p2:.6f}).")
    cf_p3=tval('cf_first',3,'1'); sem_p3=tval('sem_first',3,'1')
    if cf_p3 is not None and sem_p3 is not None:
        lines.append(f"- prefix3 TIGER-style comparable result favors {'sem_first' if sem_p3>cf_p3 else 'cf_first'} at epsilon=1.0 ({sem_p3:.6f} vs {cf_p3:.6f}).")
    lines += ['', f'Raw TSV: `{rep / "order_ablation_cold_raw.tsv"}`', f'TIGER TSV: `{rep / "order_ablation_cold_tiger.tsv"}`', f'Summary TSV: `{rep / "order_ablation_cold_summary.tsv"}`', '']
    (rep/'order_ablation_cold_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(rep/'order_ablation_cold_report.md')

if __name__=='__main__': main()
