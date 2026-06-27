# Build Repo Audit

- readiness: `CHORD_NEW_MACHINE_REPO_READY`
- target repo: `/home/huangxin/llmNrec/chord_new_machine_repro`
- legacy builder hash match: True
- wrapper scripts generated: True
- missing files count: 0
- forbidden dirs count: 0
- forbidden generated files count: 0
- model regular weight files in repo scan count: 0
- model handling: symlinked_for_local_use_weights_gitignored -> `/home/huangxin/llmNrec/repro_bundles/Beauty_new_machine_full_pipeline/models/Sentence-T5/sentence-t5-base`
- data handling: not_copied_path_configured_only

## Key Scripts
- verify: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/00_verify_inputs.py` exists=True
- st5: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/01_build_st5_embeddings.py` exists=True
- legacy_cf: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/02_build_legacy_cf_ppmi_svd.py` exists=True
- residual: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/03_build_residual_resources.py` exists=True
- pls: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/04_build_pls_shared_private.py` exists=True
- sid: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/05_optional_build_sid_index.py` exists=True
- downstream: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/06_optional_downstream_train_eval.sh` exists=True
- audit: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/audit_reproduction.py` exists=True
- legacy_builder: `/home/huangxin/llmNrec/chord_new_machine_repro/third_party/legacy_biview/legacy_backup_20260620_050517/build_biview_resources.py` exists=True
