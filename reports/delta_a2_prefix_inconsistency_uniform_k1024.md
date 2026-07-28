# Tokenizer-Level Consistency: Uniform K1024

## Protocol

This rerun uses the seed-42 K1024 CHORD tokenizer resources for Beauty,
Instruments, and Yelp. TIGER and LETTER retain their traceable true-reproduction
tokenizer artifacts. No downstream checkpoint or PCSC state is involved.

- Layer: L2 (`c2` grouped by preceding `c1`)
- Statistic: assignment-frequency-weighted eta-squared of item-level `Delta A2`
- Minimum prefix-group support: `S=3`
- Null: token-local permutation preserving each token's contribution multiset
  and prefix-group sizes
- Permutations: 200
- Base permutation seed: 424242
- RNG scope: deterministic per
  `(dataset, method, layer, metric, support, token)`
- Empirical p-value: `(1 + #null >= observed) / 201`

## Uniform-K1024 Results

| Dataset | Method | Eligible tokens | Assignments | Observed eta2 | Shuffled eta2 | Excess eta2 | Empirical p |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | TIGER | 233 | 6,080 | 0.4323 | 0.1743 | 0.2580 | 0.004975 |
| Beauty | LETTER | 251 | 8,395 | 0.2742 | 0.1584 | 0.1158 | 0.004975 |
| Beauty | CHORD K1024 | 217 | 3,049 | 0.2453 | 0.1394 | **0.1059** | 0.004975 |
| Instruments | TIGER | 134 | 4,497 | 0.4194 | 0.1759 | 0.2434 | 0.004975 |
| Instruments | LETTER | 238 | 7,555 | 0.4081 | 0.1202 | 0.2879 | 0.004975 |
| Instruments | CHORD K1024 | 202 | 2,731 | 0.2154 | 0.1291 | **0.0863** | 0.004975 |
| Yelp | TIGER | 246 | 8,958 | 0.4883 | 0.1572 | 0.3312 | 0.004975 |
| Yelp | LETTER | 256 | 18,085 | 0.3762 | 0.0937 | 0.2825 | 0.004975 |
| Yelp | CHORD K1024 | 289 | 6,264 | 0.1766 | 0.1045 | **0.0722** | 0.004975 |

CHORD has the lowest excess eta-squared on all three datasets. The result
supports reduced, not eliminated, prefix-conditioned contribution variation.
The empirical p-values compare each method with its own shuffled null; they are
not pairwise significance tests between methods.

## Change From The Mixed-K Audit

| Dataset | Previous CHORD K | Previous excess eta2 | New CHORD K | New excess eta2 |
|---|---:|---:|---:|---:|
| Beauty | 1024 | 0.1066 | 1024 | 0.1059 |
| Instruments | 256 | 0.1269 | 1024 | 0.0863 |
| Yelp | 256 | 0.1166 | 1024 | 0.0722 |

The small Beauty change is solely due to replacing the old global RNG stream
with deterministic token-local RNG streams. Its observed eta-squared is
unchanged. Instruments and Yelp additionally change tokenizer capacity from
K256 to K1024.

## CHORD Artifact Provenance

| Dataset | Tokenizer seed | Index MD5 |
|---|---:|---|
| Beauty | 42 | `e412c9264f6dc3af5e8c871b710bda4a` |
| Instruments | 42 | `079cce4724fe2c09ffc2bdb750f8f39e` |
| Yelp | 42 | `77b884bb2db7dedc699bc0ac26ca6897` |

The exact component paths and hashes are recorded in:

`results/reconstruction_role_mixture_uniform_k1024_seed42_stable_rng/manifest.json`

The primary machine-readable table is:

`results/reconstruction_role_mixture_uniform_k1024_seed42_stable_rng/weighted_excess_eta2.csv`

## Paper-Ready Text

Tokenizer-Level Consistency. Using the diagnostic in Figure 1(b), we report
excess eta-squared, the observed value minus its token-local shuffled control.
Under a uniform K1024 CHORD configuration, CHORD obtains the lowest values
across Beauty, Instruments, and Yelp (0.072-0.106), compared with TIGER
(0.243-0.331) and LETTER (0.116-0.288), indicating reduced rather than
eliminated prefix-conditioned variation in reused codewords. These descriptive
values do not alone establish pairwise significance or causal recommendation
gains.
