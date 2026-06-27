#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chord.io_utils import save_json
from chord.paths import load_config

REQUIRED_BASE_FILES = [
    'base_build_summary.json',
    'base_config.json',
    'base_raw_codes.json',
    'item_order.json',
    'z_shared.npy',
    'z_cfres.npy',
    'z_semres.npy',
    'c1.npy',
    'c2.npy',
    'c3.npy',
    'kmeans_c1_centers.npy',
    'kmeans_c2_centers.npy',
    'kmeans_c3_centers.npy',
]

def nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0

def missing_required(output_dir: Path) -> list[str]:
    return [name for name in REQUIRED_BASE_FILES if not nonempty(output_dir/name)]

def main():
    ap=argparse.ArgumentParser(description='Build or plan CHORD PLS shared/private resources.')
    ap.add_argument('--config',default='configs/beauty_new_machine.yaml')
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/args.config)
    script=root/'chord/pls_resources/build_pls_shared_private_resources.py'
    if not script.exists():
        raise SystemExit(f'No PLS builder found: {script}')
    st5_dir=cfg.output_root/'st5'/cfg.dataset
    pls=cfg.raw.get('pls',{})
    resource_dir=cfg.output_root/'resources'/cfg.dataset
    output_dir=cfg.output_root/'base'/f'{cfg.dataset}_chord_seed{cfg.seed}'
    force=bool(cfg.raw.get('force', False)) or os.environ.get('FORCE') == '1'
    cmd=[sys.executable,str(script),'--dataset',cfg.dataset,'--seed',str(cfg.seed),'--shared_dim',str(pls.get('shared_dim',128)),'--private_dim',str(pls.get('private_dim',64)),'--k1',str(pls.get('k1',256)),'--k2',str(pls.get('k2',256)),'--k3',str(pls.get('k3',256)),'--resource_dir',str(resource_dir),'--st5_dir',str(st5_dir),'--output_dir',str(output_dir)]
    if force:
        cmd.append('--force')

    missing=missing_required(output_dir)
    complete=not missing
    plan={
        'status':'ready_to_run' if args.run else 'planned_only',
        'builder_source':str(script),
        'runner_script':str(script),
        'st5_dir':str(st5_dir),
        'command':cmd,
        'resource_dir':str(resource_dir),
        'output_dir':str(output_dir),
        'force':force,
        'required_files':REQUIRED_BASE_FILES,
        'complete_existing_base':complete,
        'missing_files':missing,
    }
    save_json(plan,cfg.output_root/'reports'/f'{cfg.dataset}_pls_plan.json')
    print(json.dumps(plan,indent=2))
    if args.run:
        if complete and not force:
            summary={
                'status':'reused_existing',
                'output_dir':str(output_dir),
                'base_build_summary':str(output_dir/'base_build_summary.json'),
                'force':False,
                'required_files':REQUIRED_BASE_FILES,
            }
            print(f'SKIP existing complete PLS base: {output_dir}')
            save_json(summary,cfg.output_root/'reports'/f'{cfg.dataset}_pls_summary.json')
            save_json(summary,output_dir/'pls_wrapper_summary.json')
            return
        if output_dir.exists() and missing and not force:
            summary={
                'status':'incomplete_existing_base',
                'output_dir':str(output_dir),
                'base_build_summary':str(output_dir/'base_build_summary.json'),
                'force':False,
                'missing_files':missing,
                'hint':'Set FORCE=1 to rebuild, or clean the incomplete output_dir manually.',
            }
            save_json(summary,cfg.output_root/'reports'/f'{cfg.dataset}_pls_summary.json')
            raise SystemExit(
                'Incomplete existing PLS base at '
                f'{output_dir}; missing files: {", ".join(missing)}. '
                'Set FORCE=1 to rebuild, or clean the directory.'
            )
        subprocess.check_call(cmd,cwd=str(root),env={**os.environ,'PYTHONPATH':str(root)+':'+os.environ.get('PYTHONPATH','')})
        post_missing=missing_required(output_dir)
        status='regenerated' if not post_missing else 'regenerated_incomplete'
        summary={
            'status':status,
            'output_dir':str(output_dir),
            'base_build_summary':str(output_dir/'base_build_summary.json'),
            'force':force,
            'required_files':REQUIRED_BASE_FILES,
            'missing_files':post_missing,
        }
        save_json(summary,cfg.output_root/'reports'/f'{cfg.dataset}_pls_summary.json')
        if output_dir.exists():
            save_json(summary,output_dir/'pls_wrapper_summary.json')
        if post_missing:
            raise SystemExit(f'PLS builder finished but output is incomplete; missing: {", ".join(post_missing)}')
if __name__=='__main__': main()
