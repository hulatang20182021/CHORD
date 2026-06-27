#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/beauty_new_machine.yaml'); ap.add_argument('--run',action='store_true'); args=ap.parse_args(); root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/'scripts/02_build_legacy_cf_ppmi_svd.py'),'--config',args.config]
    if args.run: cmd.append('--run')
    print('Residual resources are produced by the legacy builder.'); print(' '.join(cmd))
    if args.run: subprocess.check_call(cmd)
if __name__=='__main__': main()
