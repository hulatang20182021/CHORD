# CHORD Run Order

Use `scripts/utils/example.sh` for the normal paper pipeline. The same stages can also be
run individually for debugging.

## 0. Verify Inputs

```bash
python scripts/00_verify_inputs.py --config configs/beauty_new_machine.yaml
```

Checks raw data files, Sentence-T5 model files, package versions, and required CHORD pipeline files.

## 1. ST5 Embeddings

```bash
python scripts/01_build_st5_embeddings.py --config configs/beauty_new_machine.yaml --run
```

Builds item-order-aligned, normalized Sentence-T5 embeddings.

## 2. Train-Only CF Resources

```bash
python scripts/02_build_legacy_cf_ppmi_svd.py --config configs/beauty_new_machine.yaml --run
```

Builds train-only co-occurrence, PPMI, CF-SVD, semantic base/residual, and related resource summaries.

## 3. Residual Resources

```bash
python scripts/03_build_residual_resources.py --config configs/beauty_new_machine.yaml --run
```

Ensures residual resources exist for downstream PLS/PCSC stages.

## 4. PLS Shared/Private Representations

```bash
python scripts/04_build_pls_shared_private.py --config configs/beauty_new_machine.yaml --run
```

Builds shared consensus, CF-private residual, and semantic-private residual representations.

## 5. SID Index

```bash
python scripts/05_optional_build_sid_index.py --config configs/beauty_new_machine.yaml --run
```

Builds the static CHORD SID index. The paper default uses `C4_MODE=dpos`.

## 6. Downstream Training/Evaluation

Recommended launcher:

```bash
DATASET=Beauty SEED=42 GPU=0 EPOCHS=60 NUM_BEAMS=20 RUN_DOWNSTREAM=1 \
bash scripts/utils/example.sh
```

The paper default uses hard SID only, `PCSC_MODE=legacy5`, and final-checkpoint evaluation.

## 7. Audit Report

When `RUN_AUDIT=1`, `scripts/run_chord_pipeline.sh` writes a compact reproduction audit under
`$RESULT_BASE/reports/` and `$RESULT_BASE/audit_report.json`.
