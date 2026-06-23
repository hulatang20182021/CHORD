#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT=Path('/home/huangxin/llmNrec/Letter/LETTER-master')
PROJECT=ROOT/'component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline'
TIGER=ROOT/'LETTER-TIGER'
sys.path.insert(0, str(PROJECT/'scripts'))
sys.path.insert(0, str(TIGER))
from project_paths import load_json
from collator import TestCollator
from data import SeqRecDataset
from generation_trie import Trie
from utils import prefix_allowed_tokens_fn, set_seed

def sid_text(sid): return ''.join(sid)
def parse_ratio_tag(r): return f"cold{int(round(float(r)*100)):02d}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split_key', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--warm_index', required=True)
    ap.add_argument('--eval_alias', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--num_beams', type=int, default=100)
    ap.add_argument('--test_batch_size', type=int, default=32)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args=ap.parse_args()
    set_seed(args.seed)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt=Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt}')
    warm_index=load_json(Path(args.warm_index))
    sid_to_item={sid_text(v):str(k) for k,v in warm_index.items()}
    all_warm_sids=set(sid_to_item)
    ns=SimpleNamespace(seed=args.seed, base_model='./ckpt/TIGER', output_dir='./ckpt', data_path=args.data_root,
        tasks='seqrec', dataset=args.eval_alias, index_file='.index.json', max_his_len=20, add_prefix=False,
        his_sep=', ', only_train_response=False, train_prompt_sample_num='1', train_data_sample_num='-1',
        valid_prompt_id=0, sample_valid=True, valid_prompt_sample_num=2, filter_items=True, results_file='unused',
        test_batch_size=args.test_batch_size, num_beams=args.num_beams, sample_num=-1, gpu_id=0,
        test_prompt_ids='0', metrics='hit@1,hit@5,hit@10,ndcg@1,ndcg@5,ndcg@10', test_task='SeqRec')
    tokenizer=T5Tokenizer.from_pretrained(str(ckpt), model_max_length=512, local_files_only=True)
    test_data=SeqRecDataset(ns, mode='test', sample_num=-1)
    test_data.set_prompt(0)
    eval_index=load_json(Path(args.data_root)/args.eval_alias/f'{args.eval_alias}.index.json')
    eval_inter=load_json(Path(args.data_root)/args.eval_alias/f'{args.eval_alias}.inter.json')
    trie=Trie([[0]+tokenizer.encode(x) for x in all_warm_sids])
    prefix_fn=prefix_allowed_tokens_fn(trie)
    model=T5ForConditionalGeneration.from_pretrained(str(ckpt), low_cpu_mem_usage=True).to(device)
    model.eval()
    loader=DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=TestCollator(ns, tokenizer), shuffle=False, num_workers=0)
    out=Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    started=time.time(); n=0
    with out.open('w', encoding='utf-8') as f, torch.no_grad():
        for step,batch in enumerate(loader, start=1):
            inputs=batch[0].to(device); targets=batch[1]
            gen=model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_fn, num_beams=args.num_beams, num_return_sequences=args.num_beams,
                output_scores=True, return_dict_in_generate=True, early_stopping=True)
            texts=[x.strip().replace(' ','') for x in tokenizer.batch_decode(gen['sequences'], skip_special_tokens=True)]
            scores=[float(x) for x in gen['sequences_scores'].detach().cpu().tolist()]
            offset=(step-1)*args.test_batch_size
            users=list(test_data.inters.keys())[offset:offset+len(targets)]
            for b,uid in enumerate(users):
                seq=[str(x) for x in eval_inter[str(uid)]]
                target_item=seq[-1]
                lo=b*args.num_beams; hi=(b+1)*args.num_beams
                beams=[]
                for text,score in zip(texts[lo:hi], scores[lo:hi]):
                    item=sid_to_item.get(text)
                    beams.append({'sid': warm_index[item] if item is not None else [], 'sid_text': text, 'score': score, 'item_id': item, 'is_warm': item is not None})
                rec={'user_id': str(uid), 'target_item': target_item, 'target_sid': eval_index[target_item],
                     'history': seq[:-1], 'beams': beams}
                f.write(json.dumps(rec, ensure_ascii=False)+'\n'); n+=1
            if step==1 or step%10==0:
                print(f'[export] batch {step}/{len(loader)} users={n} elapsed={time.time()-started:.1f}s', flush=True)
    print(json.dumps({'output': str(out), 'users': n, 'num_beams': args.num_beams}, indent=2))
if __name__=='__main__': main()
