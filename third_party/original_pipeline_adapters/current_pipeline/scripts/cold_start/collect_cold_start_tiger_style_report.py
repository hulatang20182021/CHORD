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
from pathlib import Path
from project_paths import NEW_BASE
COLD_BASE=NEW_BASE/'results/pls_sd128_dpos_pcsc/cold_start'

def load(p):
    try: return json.loads(Path(p).read_text(encoding='utf-8'))
    except Exception: return None

def fmt(v): return '' if v is None else f'{float(v):.6f}' if isinstance(v,(float,int)) else str(v)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--eval_dir', required=True); ap.add_argument('--train_run', required=True); args=ap.parse_args()
    ed=Path(args.eval_dir); reports=COLD_BASE/'reports'; reports.mkdir(parents=True, exist_ok=True)
    rows=[]
    for n in [3,1,2]:
        m=load(ed/f'metrics_prefix{n}.json')
        if m:
            for k,v in m.items():
                if k.startswith('epsilon_') and ('HR@' in k or 'NDCG@' in k):
                    parts=k.split('_'); eps=parts[1]; metric='_'.join(parts[2:])
                    rows.append({'prefix_len':n,'setting':'TIGER-style comparable' if n==3 else 'CHORD diagnostic','epsilon':eps,'metric':metric,'value':v})
    strict=load(COLD_BASE/'runs'/args.train_run/'cold_eval_metrics.json') or {}
    diag=load(COLD_BASE/'reports/cold_start_generation_diagnosis.json') or {}
    summary=reports/'cold_start_tiger_style_summary.tsv'
    with summary.open('w', encoding='utf-8') as f:
        f.write('prefix_len\tsetting\tepsilon\tmetric\tvalue\n')
        for r in rows:
            f.write(f"{r['prefix_len']}\t{r['setting']}\t{r['epsilon']}\t{r['metric']}\t{fmt(r['value'])}\n")
    md=reports/'cold_start_tiger_style_report.md'
    lines=['# CHORD Cold-start TIGER-style Prefix Expansion Report','',f'- eval_dir: `{ed}`',f'- train_run: `{args.train_run}`','', '## Strict exact-SID cold-start','']
    mean=strict.get('mean_results',{})
    for k in ['hit@1','hit@5','hit@10','ndcg@1','ndcg@5','ndcg@10']:
        lines.append(f'- {k}: {fmt(mean.get(k))}')
    lines += ['', '## Generation Diagnosis', '']
    gd=diag.get('generation_distribution',{}); ph=diag.get('prefix_hit',{})
    for k in ['top10_cold_prediction_ratio','top20_cold_prediction_ratio','top50_cold_prediction_ratio','top100_cold_prediction_ratio']:
        if k in gd: lines.append(f'- {k}: {fmt(gd[k])}')
    for k in ['prefix1_hit@50','prefix2_hit@50','prefix3_hit@50','full_sid_hit@50']:
        if k in ph: lines.append(f'- {k}: {fmt(ph[k])}')
    lines += ['', '## Prefix Expansion Metrics', '', '| prefix | setting | epsilon | HR@1 | HR@5 | HR@10 | NDCG@1 | NDCG@5 | NDCG@10 |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for n in [3,1,2]:
        m=load(ed/f'metrics_prefix{n}.json') or {}
        epsilons=sorted({x.split('_')[1] for x in m if x.startswith('epsilon_')}, key=lambda x:float(x))
        for e in epsilons:
            setting='TIGER-style comparable' if n==3 else 'CHORD diagnostic'
            vals=[m.get(f'epsilon_{e}_{name}') for name in ['HR@1','HR@5','HR@10','NDCG@1','NDCG@5','NDCG@10']]
            lines.append(f"| {n} | {setting} | {e} | " + ' | '.join(fmt(v) for v in vals) + ' |')
    lines += ['', '## Conclusion', '', 'strict exact-SID evaluates zero-shot full SID generation; TIGER-style evaluates prefix-based cold candidate expansion; prefix1/2 are CHORD diagnostic variants and should not be directly compared with TIGER unless explicitly stated.']
    p3=load(ed/'metrics_prefix3.json') or {}; p1=load(ed/'metrics_prefix1.json') or {}
    if (p3.get('epsilon_1_HR@10',0) or p3.get('epsilon_1.0_HR@10',0)) < (p1.get('epsilon_1_HR@10',0) or p1.get('epsilon_1.0_HR@10',0)):
        lines.append("CHORD's shared layer carries stronger transferable cold-start signal than its full private-residual prefix. Therefore, TIGER-style prefix3 expansion underestimates CHORD's cold-start potential, and a CHORD-specific cold bridge should use shared-prefix recall plus content-aware reranking.")
    md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    local=ed/'tiger_style_eval_report.md'; local.write_text(md.read_text(encoding='utf-8'), encoding='utf-8')
    print(md); print(summary)
if __name__=='__main__': main()
