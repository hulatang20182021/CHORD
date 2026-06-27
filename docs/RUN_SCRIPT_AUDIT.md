# Run Script Audit

- final classification: `RUN_SCRIPT_READY_WITH_TODOS`
- run script: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/run_beauty_new_machine_pipeline.sh`
- example script: `/home/huangxin/llmNrec/chord_new_machine_repro/scripts/utils/example_beauty_new_machine.sh`
- bash -n passed: True
- dry run passed: True
- pyyaml dependency recorded: True
- README updated: True
- RUN_ORDER updated: True

## Parameter Override

run_beauty_new_machine_pipeline.sh writes $OUTPUT_ROOT/runtime_config.yaml from shell variables and passes it through --config.

## TODOs
- scripts/04_build_pls_shared_private.py currently emits a plan and candidate adapters; execute only after confirming adapter path assumptions.
- scripts/05_optional_build_sid_index.py is optional/planning by default.
- scripts/06_optional_downstream_train_eval.sh intentionally does not launch training/eval by default.
