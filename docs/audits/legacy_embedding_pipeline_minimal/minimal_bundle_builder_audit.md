# Minimal Bundle Builder Audit

## Summary

- tar.gz hash matches: `True`
- tar.gz list ok: `True`
- MANIFEST_ALL_FILES_SHA256 passed: `True`
- MANIFEST_MISSING_FILES empty: `True`
- exact legacy builder found: `True`
- classifications: `MINIMAL_BUNDLE_OK, LEGACY_BUILDER_FOUND_AT_DIFFERENT_PATH`

## Builder

Expected direct path exists: `False` (`code/02_legacy_biview_trainonly_cf/build_biview_resources.py`)

| path | sha256 | matches_expected |
| --- | --- | --- |
| code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py | 753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff | True |

## Legacy Directory Files

- `code/02_legacy_biview_trainonly_cf/Historical_Beauty_CF_SVD_origin_trace.md`
- `code/02_legacy_biview_trainonly_cf/debug_biview_ppmi_old_machine.py`
- `code/02_legacy_biview_trainonly_cf/debug_old_machine_legacy_biview_context.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/audit_biview_checkpoints.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_dataset_alias.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/collect_biview_report.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/collect_static_intersection_best_ablation_report.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/collect_static_intersection_downstream_report.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/collect_static_intersection_sid_report.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/collect_static_ridge_sid_report.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/finetune_biview_trainonly.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/generate_biview_dsnloss_index.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/generate_biview_dsnloss_v2_index.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/generate_biview_shared_private_index.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/prepare_static_intersection_downstream_candidates.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/prepare_static_intersection_pcsc_off_top2.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/prepare_static_ridge_downstream_candidates.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/probe_biview_id_q_alignment.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/probe_static_intersection_id_alignment.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/probe_static_ridge_sid_id_alignment.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_biview_beauty_down60.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_biview_dsnloss_beauty_down60.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_biview_dsnloss_v2_beauty_down60.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_biview_id_q_probe.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_one_biview_downstream.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_static_intersection_best_ablation_grid.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_static_intersection_downstream_grid.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_static_intersection_sid_sweep.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/run_static_ridge_sid_sweep.sh`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_best_finetune.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_best_make_swapped_index.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_best_modeling.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_best_run_one.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_downstream_run_one.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_intersection_sid_build.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/static_ridge_sid_build.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/train_biview_dsnloss_tokenizer.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/train_biview_dsnloss_v2_tokenizer.py`
- `code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/train_biview_shared_private_tokenizer.py`
- `code/02_legacy_biview_trainonly_cf/old_machine_legacy_biview_stable_reproduction_report.md`
- `code/02_legacy_biview_trainonly_cf/project_paths.py`

## Content-Level Legacy Logic Candidates

| path | sha256 | features |
| --- | --- | --- |
| code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py | 753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff | {'def build_ppmi': True, 'bidirectional_window': True, 'counts_increment': True, 'TruncatedSVD': True, 'train_test_split': True, 'seq[:-2]': True} |

## Manifest

- builder mentions: `['753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff  ./code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py']`
- expected hash mentions: `['753d4eeb902f28deb621fa7f7cf3eb31f5fc8a334f741e8ce179da50057b6bff  ./code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/build_biview_resources.py']`
- missing file size: `1`

## project_paths.py

| path | sha256 | matches_expected |
| --- | --- | --- |
| code/02_legacy_biview_trainonly_cf/project_paths.py | 96014dc9079b83c72d04961d2e47de9a3119f9911fd0d21482958409f43166f2 | True |

## Sentence-T5

- model.safetensors exists: `True` size=219268776
- 2_Dense/model.safetensors exists: `True` size=2359416

## Beauty Data

| path | sha256 | expected_sha256 | matches_expected |
| --- | --- | --- | --- |
| /home/huangxin/llmNrec/data/Beauty/Beauty.inter.json | 8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5 | 8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5 | True |
| /home/huangxin/llmNrec/data/Beauty/Beauty.index.json | be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82 | be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82 | True |
| /home/huangxin/llmNrec/data/Beauty/Beauty.item.json | 74aeddf911548aadfcd1dab326c16678ed4ac74457e874da300d63a94bdb2330 | None | None |

## Recommendation

Exact builder exists under code/02_legacy_biview_trainonly_cf/legacy_backup_20260620_050517/. If scripts expect the shallower path, create a copy or symlink to code/02_legacy_biview_trainonly_cf/build_biview_resources.py after approval; do not need to re-transfer the tarball.
