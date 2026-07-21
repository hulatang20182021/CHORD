# SID Component-Order Ablation

This directory contains the Beauty K1024 controls used to isolate the ordering of the
three learned CHORD components. The default paper mainline remains:

```text
shared -> semantic gap -> CF gap -> DPOS
```

The two controls are:

```text
shared -> CF gap -> semantic gap -> DPOS
semantic gap -> shared -> CF gap -> DPOS
```

The index builder only permutes the first three assignments from the audited mainline
index. It does not refit codebooks, and it preserves each item's DPOS suffix. Training
uses role-aware hidden-state routing so that changing a component's position does not
silently change its supervision target:

```text
h_shared                 -> semantic base
h_semres                 -> semantic residual
h_cfres                  -> CF residual
h_shared + h_semres      -> full semantic representation
h_shared + h_cfres       -> full CF representation
```

Run the fixed-tensor routing audit before launching the experiments:

```bash
PYTHONPATH=/path/to/LETTER-TIGER:chord/downstream/scripts \
python experiments/order_ablation/audit_pcsc_role_alignment_equivalence.py
```

Then run one variant:

```bash
bash experiments/order_ablation/run_beauty_k1024_order_ablation.sh shared_cfres_semres
bash experiments/order_ablation/run_beauty_k1024_order_ablation.sh semres_shared_cfres
```

Set `RESULT_BASE`, `LETTER_ROOT`, `DATA_ROOT`, and `FORMAL_PYTHON` when the artifacts or
environment are outside the repository defaults. The optional serial queue accepts
`BLOCKING_PID=<pid>` when it must wait for an existing GPU job.

Both variants use seed 42, K1024, a 100-epoch learning-rate schedule stopped at epoch 60,
and one fixed epoch-60 test. They should be compared with the role-aligned main-order A1
control, not with the cross-view compositional main method, because the purpose here is to
isolate SID order under invariant component-role supervision.
