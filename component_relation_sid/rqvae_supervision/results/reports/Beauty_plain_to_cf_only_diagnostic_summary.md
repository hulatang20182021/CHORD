# Beauty Plain to CF-only Diagnostic Summary

## Plain failure diagnosis

Diagnosis class: `embedding_source_problem`

Implementation bug found: `false`

Reason: raw ST5 cosine similarity has very low rank correlation with CF-SVD cosine similarity. The plain index generation is internally consistent: layered token namespaces are used, the recomputed total vocab matches the audit/build summary, no token naming collision was found, and raw code prefixes match the generated index prefixes.

## Plain metrics

| method | duplicate | vocab | prefix1 mean | prefix2 mean | prefix3 mean | exposure <=5 | prefix1 lift | prefix2 lift | prefix3 lift | reconstruction loss | code usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plain | 0 | 377 | 115.2476 | 1.9796 | 1.0928 | 0.002653 | 8.0562 | 42.1786 | 70.6667 | 0.000261 | 232/256, 250/256, 242/256 |

## CF-only sweep

Each CF-only run used 50 epochs, batch size 256, temperature 0.1, and only `L_rec + L_vq + lambda_cf_global * L_CF_global`.

| method | duplicate | vocab | prefix1 mean | prefix2 mean | prefix3 mean | exposure <=5 | prefix1 lift | prefix2 lift | prefix3 lift | reconstruction loss | CF loss | code usage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| plain | 0 | 377 | 115.2476 | 1.9796 | 1.0928 | 0.002653 | 8.0562 | 42.1786 | 70.6667 | 0.000261 | NA | 232/256, 250/256, 242/256 |
| CF-only lambda 0.01 | 0 | 677 | 79.6118 | 1.3102 | 1.0142 | 0.001477 | 26.6992 | 254.8571 | 159.0000 | 0.000433 | 1.1372 | 152/256, 256/256, 256/256 |
| CF-only lambda 0.05 | 0 | 590 | 142.3647 | 1.6111 | 1.0201 | 0.000000 | 16.0091 | 167.0714 | NA | 0.000433 | 1.5395 | 85/256, 244/256, 256/256 |
| CF-only lambda 0.10 | 0 | 499 | 198.3770 | 2.2347 | 1.0380 | 0.002004 | 11.7550 | 79.6607 | 326.0000 | 0.000430 | 1.8217 | 61/256, 170/256, 256/256 |

`lambda=0.05` has `prefix3_lift = null` in the audit because the random prefix3 sharing denominator was zero for that sampled baseline, so it should be treated as undefined rather than failed.

## Deltas against plain

| method | delta prefix1 lift | delta prefix2 lift | delta prefix3 lift | delta reconstruction loss | delta exposure <=5 | delta vocab |
|---|---:|---:|---:|---:|---:|---:|
| CF-only lambda 0.01 | +18.6430 | +212.6786 | +88.3333 | +0.000172 | -0.001175 | +300 |
| CF-only lambda 0.05 | +7.9528 | +124.8929 | NA | +0.000172 | -0.002653 | +213 |
| CF-only lambda 0.10 | +3.6988 | +37.4821 | +255.3333 | +0.000169 | -0.000649 | +122 |

## Alignment

| method | NMI c1-CF | NMI c2-CF | NMI c3-CF | NMI c1-product | NMI c2-product | NMI c3-product | NMI c1-original | NMI c2-original | NMI c3-original |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CF-only lambda 0.01 | 0.5480 | 0.3455 | 0.3232 | 0.2822 | 0.1775 | 0.1638 | 0.4548 | 0.3311 | 0.3189 |
| CF-only lambda 0.05 | 0.4874 | 0.3511 | 0.3235 | 0.2687 | 0.1796 | 0.1636 | 0.4506 | 0.3259 | 0.3199 |
| CF-only lambda 0.10 | 0.4501 | 0.3292 | 0.3273 | 0.2642 | 0.1664 | 0.1667 | 0.4473 | 0.2851 | 0.3231 |

## Interpretation

CF-only diagnostic is effective relative to plain: all runs keep duplicate at 0 and exposure <=5 below 1.5%, and they improve behavior lift substantially. Reconstruction loss worsens from 0.000261 to about 0.000430-0.000433, but it does not collapse.

Best lambda depends on the target:

- `lambda=0.01` is best for prefix1 and prefix2 lift. It pushes prefix2 lift to 254.86, clearly above V2-ST5's 102.84, while prefix3 lift improves to 159.00 but remains below V2-ST5's 322.67.
- `lambda=0.10` is best for prefix3 lift. It reaches 326.00, roughly matching or slightly exceeding V2-ST5's 322.67, but prefix2 lift is only 79.66 and c1/c2 code usage is more compressed.
- `lambda=0.05` is intermediate for prefix2 lift but has an undefined prefix3 lift under the current random baseline sample.

## Recommendation

Proceed to `CF + c1 product_type alignment`, using CF-only as a validated diagnostic signal rather than jumping to full supervised CR-LETTER. The most useful next starting point is `lambda_cf_global=0.01` if prefix2 behavior sharing is the priority, or `lambda_cf_global=0.10` if prefix3 sharing is the priority. A follow-up should also stabilize lift estimation by increasing or stratifying random-pair sampling, because very small random prefix3 sharing can make prefix3 lift noisy or undefined.
