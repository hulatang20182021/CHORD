# Shared-Anchor Epoch Sweeps

After the Beauty ablation queue completes, each dataset is trained in one uninterrupted
trajectory through epoch 70 under the fixed 100-epoch schedule. Checkpoints
50/55/60/65/70 are preserved and evaluated on the test split.

These test sweeps are diagnostics only. They must not be used to select the checkpoint
reported in the formal main table; formal reporting keeps the predeclared epoch-60 policy.
