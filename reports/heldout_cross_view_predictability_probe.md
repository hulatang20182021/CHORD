# Held-Out Cross-View Predictability Probe

## Purpose

This experiment tests whether the residual energy in the MLP-sem-first tokenizer mainly
reflects cross-view-specific variation or insufficient cross-view prediction. It uses the
raw pre-quantization Sentence-T5 semantic representation and train-interaction-only CF
representation entering the tokenizer.

For five fixed item-level 80/20 splits (seeds 42--46), each direction compares a training
target mean, standardized Ridge, the exact mainline MLP configuration, and an MLP trained
after shuffling target-item correspondence. Every scaler and predictor is fitted using the
80% training items only. All metrics are evaluated on held-out items.

The primary R2 is variance weighted, equivalent to the global explained-variance ratio.
Uniform-per-dimension R2 is retained in the complete result artifact because low-variance
dimensions can otherwise be hidden. The source arrays and item-order files match exactly
for every dataset.

## Main Results

| Dataset | Direction | Ridge R2 | MLP R2 | Shuffled MLP R2 | MLP cosine | MLP nRMSE | MLP train-test R2 gap |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | semantic to CF | 0.0609 +/- 0.0066 | **0.1098 +/- 0.0056** | -0.0107 +/- 0.0008 | 0.3937 +/- 0.0046 | 0.9432 +/- 0.0031 | 0.0785 +/- 0.0140 |
| Beauty | CF to semantic | **0.1275 +/- 0.0047** | 0.1171 +/- 0.0059 | -0.0191 +/- 0.0042 | 0.7894 +/- 0.0009 | 0.9376 +/- 0.0018 | 0.0646 +/- 0.0114 |
| Instruments | semantic to CF | 0.0689 +/- 0.0028 | **0.1247 +/- 0.0030** | -0.0108 +/- 0.0009 | 0.4501 +/- 0.0022 | 0.9351 +/- 0.0015 | 0.0653 +/- 0.0099 |
| Instruments | CF to semantic | **0.1371 +/- 0.0024** | 0.1218 +/- 0.0036 | -0.0209 +/- 0.0023 | 0.8280 +/- 0.0016 | 0.9351 +/- 0.0051 | 0.0715 +/- 0.0063 |
| Yelp | semantic to CF | 0.0070 +/- 0.0016 | **0.0345 +/- 0.0021** | -0.0087 +/- 0.0006 | 0.3005 +/- 0.0029 | 0.9826 +/- 0.0007 | 0.0328 +/- 0.0060 |
| Yelp | CF to semantic | **0.0383 +/- 0.0015** | 0.0308 +/- 0.0018 | -0.0123 +/- 0.0022 | 0.6886 +/- 0.0010 | 0.9849 +/- 0.0018 | 0.0540 +/- 0.0059 |

MLP minus Ridge held-out R2 is `+0.0489/+0.0558/+0.0274` for semantic-to-CF on
Beauty/Instruments/Yelp, but `-0.0104/-0.0153/-0.0075` for CF-to-semantic.

## Conclusions

1. The original views have reproducible, held-out cross-view structure. Every learned
   predictor beats the mean or shuffled control in global R2, and shuffled-pair R2 is
   consistently negative.
2. Cross-view predictability is partial rather than dominant. The mainline MLP explains
   about 11--12% of held-out variance in Beauty/Instruments and about 3% in Yelp. Therefore,
   high residual energy cannot be identified entirely as view-private signal; predictor
   error and noise remain material alternatives.
3. Nonlinearity is direction dependent. The MLP clearly helps semantic-to-CF prediction,
   but standardized Ridge is slightly better for CF-to-semantic prediction on all three
   datasets. The evidence does not support claiming that nonlinear predictors are necessary
   in both directions.
4. Raw cosine is not sufficient evidence. The mean-only CF-to-semantic baseline already
   has cosine 0.676--0.801 because semantic embeddings are anisotropic. Centered R2 and the
   shuffled control should be the primary evidence.
5. Yelp has substantially weaker cross-view predictability than Beauty/Instruments. Any
   claim that semantic and collaborative views share a uniformly strong relation across
   datasets should be avoided.

The defensible wording is: semantic and collaborative item representations contain a
generalizable but limited paired component, with substantial directional prediction gaps.
Those gaps may contain view-specific variation, noise, and predictor error and should not
be called pure private factors.

## Artifacts

- Complete table: `results/heldout_cross_view_predictability_probe_v2/report.md`
- Per-split metrics: `results/heldout_cross_view_predictability_probe_v2/per_split_results.json`
- Machine-readable summary: `results/heldout_cross_view_predictability_probe_v2/summary.json`
- Reproducibility manifest: `results/heldout_cross_view_predictability_probe_v2/manifest.json`
- Runner: `scripts/run_heldout_cross_view_predictability_probe.py`
