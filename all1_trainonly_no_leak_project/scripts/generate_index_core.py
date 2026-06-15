#!/usr/bin/env python3
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np, torch
from tokenizer_core import Model

def dump(x,p): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+"\n",encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--st5_emb",required=True);p.add_argument("--item_order",required=True);p.add_argument("--output_dir",required=True);p.add_argument("--run_name",required=True);p.add_argument("--device",default="cuda:1");a=p.parse_args()
 out=Path(a.output_dir); paths=[out/f"{a.run_name}.index.json",out/f"{a.run_name}_raw_codes.json",out/f"{a.run_name}_build_summary.json"]
 if any(x.exists() for x in paths): raise SystemExit("Refusing overwrite")
 ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False); cfg=ck["config"];m=Model(cfg["input_dim"],cfg["cf_dim"],cfg["latent_dim"],cfg["codebook_size"],cfg["num_quantizers"]).to(a.device);m.load_state_dict(ck["model_state_dict"]);m.eval()
 x=np.load(a.st5_emb).astype(np.float32);order=[str(v) for v in json.load(open(a.item_order,encoding="utf-8"))];rows=[]
 with torch.no_grad():
  for s in range(0,len(x),2048): rows.append(m(torch.from_numpy(x[s:s+2048]).to(a.device))[2].cpu())
 codes=torch.cat(rows).numpy().astype(int); raw={i:[int(v) for v in c] for i,c in zip(order,codes)}
 buckets=defaultdict(list)
 for item,c in raw.items(): buckets[tuple(c)].append(item)
 index={}
 for c,items in buckets.items():
  for pos,item in enumerate(sorted(items,key=lambda z:int(z) if z.isdigit() else z)): index[item]=[f"<a_{c[0]}>",f"<b_{c[1]}>",f"<c_{c[2]}>",f"<d_{pos}>"]
 dup=sum(v-1 for v in Counter(map(tuple,index.values())).values() if v>1);p2=len({tuple(v[:2]) for v in index.values()});p3=len(buckets)
 summary={"num_items":len(index),"duplicate_sid_count":dup,"unique_sid_count":len(set(map(tuple,index.values()))),"max_c4":max(len(v)-1 for v in buckets.values()),
 "c1_unique":len(set(codes[:,0])),"c2_unique":len(set(codes[:,1])),"c3_unique":len(set(codes[:,2])),"c4_unique":max(len(v) for v in buckets.values()),
 "p2_unique":p2,"p3_unique":p3,"prefix3_singleton_ratio":sum(len(v)==1 for v in buckets.values())/p3}
 if dup: raise SystemExit(f"duplicate SID: {dup}")
 dump(index,paths[0]);dump(raw,paths[1]);dump(summary,paths[2]);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
