#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def key(sid,n): return '|'.join(sid[:n])
def hr_ndcg(rank,k):
    if rank is None or rank>k: return 0.0,0.0
    return 1.0,1.0/math.log2(rank+1)
def eps_tag(e): return str(e).rstrip('0').rstrip('.') if '.' in str(e) else str(e)

def ordered_candidates(beams, prefix_map, cold_sid, prefix_len, tie_break):
    scores={}; meta={}
    order=0
    for b in beams:
        sid=b.get('sid') or []
        score=float(b.get('score', -1e9))
        item=b.get('item_id')
        if item is not None:
            old=scores.get(item)
            if old is None or score>old:
                scores[item]=score; meta[item]={'is_cold':False,'order':order}
        if len(sid)>=prefix_len:
            for c in prefix_map.get(key(sid,prefix_len), []):
                old=scores.get(c)
                if old is None or score>old:
                    scores[c]=score; meta[c]={'is_cold':True,'order':order}
        order+=1
    def sortkey(item):
        is_cold=meta[item]['is_cold']
        if tie_break=='cold_first': tb=0 if is_cold else 1
        elif tie_break=='item_id': tb=int(item) if str(item).isdigit() else str(item)
        else: tb=0 if not is_cold else 1
        return (-scores[item], tb, meta[item]['order'], int(item) if str(item).isdigit() else str(item))
    return sorted(scores, key=sortkey), meta, scores

def limit_topk(ranked, meta, k, eps):
    max_cold=math.ceil(float(eps)*k)
    out=[]; cold=0
    for item in ranked:
        is_cold=meta[item]['is_cold']
        if is_cold and cold>=max_cold: continue
        out.append(item)
        if is_cold: cold+=1
        if len(out)>=k: break
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--beams_jsonl', required=True)
    ap.add_argument('--cold_prefix_map', required=True)
    ap.add_argument('--cold_item_to_sid', required=True)
    ap.add_argument('--warm_items', required=True)
    ap.add_argument('--cold_items', required=True)
    ap.add_argument('--k_list', default='1,5,10')
    ap.add_argument('--epsilons', default='0.05,0.1,0.2,0.5,1.0')
    ap.add_argument('--prefix_len', type=int, default=3)
    ap.add_argument('--tie_break', choices=['warm_first','cold_first','item_id'], default='warm_first')
    ap.add_argument('--output_metrics', required=True)
    ap.add_argument('--output_details', required=True)
    args=ap.parse_args()
    prefix_map={k:[str(x) for x in v] for k,v in load_json(args.cold_prefix_map).items()}
    cold_sid={str(k):v for k,v in load_json(args.cold_item_to_sid).items()}
    cold_set=set(map(str,load_json(args.cold_items)))
    warm_set=set(map(str,load_json(args.warm_items)))
    ks=[int(x) for x in args.k_list.split(',') if x]
    eps=[float(x) for x in args.epsilons.split(',') if x]
    sums={}; num=0; prefix_hits=0; expanded_total=0
    for e in eps:
        for k in ks:
            for m in ['HR','NDCG']:
                sums[f'epsilon_{eps_tag(e)}_{m}@{k}']=0.0
    outd=Path(args.output_details); outd.parent.mkdir(parents=True, exist_ok=True)
    with open(args.beams_jsonl, encoding='utf-8') as f, outd.open('w', encoding='utf-8') as g:
        for line in f:
            rec=json.loads(line); tgt=str(rec['target_item'])
            if tgt not in cold_set: continue
            ranked,meta,scores=ordered_candidates(rec['beams'], prefix_map, cold_sid, args.prefix_len, args.tie_break)
            expanded=[x for x in ranked if meta[x]['is_cold']]
            expanded_total+=len(expanded)
            target_prefix=key(cold_sid[tgt], args.prefix_len)
            hit_prefix=any(key((b.get('sid') or []), args.prefix_len)==target_prefix for b in rec['beams'] if len(b.get('sid') or [])>=args.prefix_len)
            prefix_hits+=int(hit_prefix)
            detail={'user_id':rec.get('user_id'), 'target_item':tgt, f'target_prefix{args.prefix_len}':target_prefix,
                    'hit_prefix_generated':bool(hit_prefix), 'rank_by_epsilon':{}, 'num_expanded_cold_candidates':len(expanded)}
            for e in eps:
                for k in ks:
                    top=limit_topk(ranked, meta, k, e)
                    rank=None
                    for i,it in enumerate(top, start=1):
                        if it==tgt: rank=i; break
                    h,n=hr_ndcg(rank,k)
                    sums[f'epsilon_{eps_tag(e)}_HR@{k}']+=h
                    sums[f'epsilon_{eps_tag(e)}_NDCG@{k}']+=n
                    if k==10: detail['rank_by_epsilon'][eps_tag(e)]=rank
                    detail[f'top{k}_items_epsilon_{eps_tag(e)}']=top
            g.write(json.dumps(detail, ensure_ascii=False)+'\n')
            num+=1
    metrics={'eval_type':'tiger_style_prefix_expansion' if args.prefix_len==3 else 'chord_diagnostic_prefix_expansion',
             'prefix_len': args.prefix_len, 'num_eval_users': num, 'cold_item_count': len(cold_set), 'warm_item_count': len(warm_set),
             'beam_export_size': sum(1 for _ in open(args.beams_jsonl, encoding='utf-8')), 'prefix_generated_ratio': prefix_hits/max(num,1),
             'avg_expanded_cold_candidates': expanded_total/max(num,1), 'tie_break': args.tie_break}
    for k,v in sums.items(): metrics[k]=v/max(num,1)
    Path(args.output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metrics).write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
if __name__=='__main__': main()
