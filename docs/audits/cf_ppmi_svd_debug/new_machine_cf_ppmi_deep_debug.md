# New Machine CF/PPMI/SVD Deep Debug

Scope: CF/PPMI/SVD only. No ST5, no PLS, no SID index, no training, no eval.

## Hash Comparison
| level | new | old expected | match |
|---|---:|---:|---:|
| raw_inter | `8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5` | `8809584b8b78fa6561771c88911ed65bd716fc9926fca1e581e6a1ab578ccfd5` | True |
| raw_index | `be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82` | `be3fb890e59b5443824cdebc560ef349191b97f20d3aecad4727de2d6b212c82` | True |
| trainonly_inter | `0b965f926b278042af4bb0ee2dfa736841360472bbda9201497da0a60be017a3` | `0b965f926b278042af4bb0ee2dfa736841360472bbda9201497da0a60be017a3` | True |
| counts_sorted_triplets | `2a03615a2ac6d84ba9dc6dc5e95c93008bd5437384d86a4b3b848eaab2971cfb` | `519847c4f3c8e31f0d94d385f776d3862615a3df5c1a55b2224114d787a55195` | False |
| ppmi_csr_all | `6199be484601b92ed87559d08e819d06232f74fdd26754cf5ef282bd88041174` | `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a` | False |
| CF-SVD npy sha16 | `4ac176b0e1291413` | `6d75cfbe18dc5aa8` | False |

## Counts / PPMI
- num_users: 22363
- full_event_count: 198502
- train_event_count: 153776
- excluded_event_count: 44726
- counts_len: 723748
- counts_total_sum: 918284.0
- row_sum_total: 918284.0
- col_sum_total: 918284.0
- ppmi_shape: [12101, 12101]
- ppmi_nnz: 722608

## SVD
- params: `{"algorithm": "randomized", "class": "sklearn.decomposition.TruncatedSVD", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}`
- sklearn_version: `1.7.2`
- CF-SVD npy sha16: `4ac176b0e1291413`
- CF-SVD raw array sha16: `57ee66fdced918dc`

## Classification
- COOCCURRENCE_COUNTS_DIFFERENCE
