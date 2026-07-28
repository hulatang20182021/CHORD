# K256 vs K1024 Fan-in and DOCS Collision Audit

## Scope

This audit uses the seed-42 CHORD tokenizer artifacts for all three datasets.
It compares K256 and K1024 without downstream training.

- `c2` fan-in: number of distinct `c1` prefixes using a `c2` codeword.
- `c3` fan-in: number of distinct `c1:c2` prefixes using a `c3` codeword.
- Suffix-required items: items belonging to a `c1:c2:c3` bucket of size
  greater than one.
- Singleton-item rate: items whose `c1:c2:c3` prefix uniquely identifies them.
- Singleton-bucket rate: proportion of distinct `c1:c2:c3` buckets containing
  exactly one item.

## Prefix and Collision Statistics

| Dataset | K | P1 | P2 | P3 | Collision groups | Suffix-required items | Suffix-required rate | Singleton-item rate | Singleton-bucket rate | Max bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 256 | 256 | 5,255 | 11,027 | 733 | 1,807 | 14.93% | 85.07% | 93.35% | 12 |
| Beauty | 1024 | 1,022 | 7,226 | 11,459 | 464 | 1,106 | **9.14%** | **90.86%** | **95.95%** | 11 |
| Instruments | 256 | 256 | 4,075 | 9,175 | 581 | 1,328 | 13.38% | 86.62% | 93.67% | 7 |
| Instruments | 1024 | 1,022 | 5,683 | 9,509 | 344 | 757 | **7.63%** | **92.37%** | **96.38%** | 9 |
| Yelp | 256 | 256 | 6,938 | 17,038 | 2,168 | 5,163 | 25.77% | 74.23% | 87.28% | 13 |
| Yelp | 1024 | 1,023 | 10,653 | 19,319 | 631 | 1,345 | **6.71%** | **93.29%** | **96.73%** | 7 |

Relative reductions in the number of suffix-required items from K256 to K1024
are 38.8% on Beauty, 43.0% on Instruments, and 73.9% on Yelp.

K1024 reduces collision prevalence on all three datasets. It does not
monotonically reduce the worst-case bucket size: Instruments increases from
7 to 9. Claims should therefore concern the fraction of collision cases rather
than the maximum collision size.

## Fan-in Statistics

| Dataset | K | c2 mean fan-in | c2 median | c2 max | c3 mean fan-in | c3 median | c3 max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty | 256 | 20.53 | 19.0 | 63 | 43.07 | 42.5 | 106 |
| Beauty | 1024 | **7.06** | **4.0** | 49 | **11.19** | **8.0** | 61 |
| Instruments | 256 | 15.92 | 15.0 | 62 | 35.84 | 34.0 | 86 |
| Instruments | 1024 | **5.56** | **3.0** | 43 | **9.30** | **6.0** | 77 |
| Yelp | 256 | 27.10 | 26.0 | 89 | 66.55 | 66.0 | 144 |
| Yelp | 1024 | **10.40** | **7.0** | 56 | **18.88** | **15.0** | 106 |

K1024 reduces both c2 and c3 mean/median fan-in on all three datasets. This
means each reused component codeword is shared across fewer preceding prefixes.

## DOCS/DPOS Verification

For all six dataset-capacity combinations:

- every complete four-token SID is unique;
- suffix values within each collision bucket form exactly `0...(bucket_size-1)`;
- deterministic distance ordering reproduces every exported suffix;
- exact suffix mismatch count is zero.

The implementation names this deterministic suffix procedure `dpos`. The paper
should use one name consistently if `DOCS` refers to the same procedure.

## Interpretation of Prefix Retention

The decrease from `c1:c2:c3` hit to full-SID hit occurs at the suffix prediction
step. However, every item has a suffix token. Singleton prefixes use the fixed
suffix `c4=0`, while collision buckets require the suffix to distinguish two or
more items. Therefore the complete retention drop should not be attributed only
to collision items.

A precise paper statement is:

> The remaining decrease from the three-component prefix to the full SID occurs
> at the DOCS suffix prediction step. For singleton prefixes the suffix is fixed
> to zero, whereas collision buckets require within-prefix item
> disambiguation. Using K=1024 reduces the fraction of items requiring such
> disambiguation from 14.9% to 9.1% on Beauty, 13.4% to 7.6% on Instruments,
> and 25.8% to 6.7% on Yelp.

## Artifacts

Newly computed combinations:

`results/fanin_collision_k256_k1024_seed42/`

Previously computed matching combinations:

`results/aaai27_crossview_compositional/fanin_collision/`
