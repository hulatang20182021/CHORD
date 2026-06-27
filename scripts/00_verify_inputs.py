#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.hash_utils import exists_sha256, sha256_file
from chord.io_utils import save_json
from chord.paths import load_config
EXPECTED={'inter':'8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5','index':'be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82','item':'74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330'}
LEGACY='753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff'
def version(name):
    try:
        m=importlib.import_module(name); return getattr(m,'__version__','UNKNOWN')
    except Exception as e: return f'MISSING: {e!r}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/beauty_new_machine.yaml'); ap.add_argument('--output'); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; cfg=load_config(root/args.config)
    data={}
    for suffix,exp in EXPECTED.items():
        row=exists_sha256(cfg.data_file(suffix)); row['expected_sha256']=exp; row['match']=row['sha256']==exp; data[cfg.data_file(suffix).name]=row
    model_required=['config.json','modules.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors','2_Dense/model.safetensors']
    model={x:(cfg.model_path/x).exists() for x in model_required}
    legacy=root/'third_party/legacy_biview/legacy_backup_20260620_050517/build_biview_resources.py'
    lh=sha256_file(legacy) if legacy.exists() else None
    report={'data':data,'model_path':str(cfg.model_path),'model_required':model,'legacy_builder':{'path':str(legacy),'sha256':lh,'match':lh==LEGACY},'versions':{m:version(m) for m in ['numpy','scipy','sklearn','torch','transformers','tokenizers','yaml']}}
    out=Path(args.output) if args.output else cfg.output_root/'verify_inputs_report.json'; save_json(report,out); print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
