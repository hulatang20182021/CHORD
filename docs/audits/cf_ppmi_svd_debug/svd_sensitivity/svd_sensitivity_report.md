# SVD Sensitivity Report

Scope: isolated TruncatedSVD sensitivity only. No ST5, no PLS, no SID index, no training, no eval.

- PPMI hash: `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a`
- expected CF-SVD sha16: `6d75cfbe18dc5aa8`
- settings tested: 11

| run | sha16 | match 6d75 | params | error |
|---|---:|---:|---|---|
| A_default | `4ac176b0e1291413` | False | `{"n_components": 128, "random_state": 42}` | `` |
| B_fixed_threads_default | `ad01883f9b6dbbbe` | False | `{"n_components": 128, "random_state": 42}` | `` |
| C_explicit_randomized | `4ac176b0e1291413` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}` | `` |
| D_power_normalizer_auto | `4ac176b0e1291413` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}` | `` |
| D_power_normalizer_none | `5db6c785414ac6ad` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "none", "random_state": 42, "tol": 0.0}` | `` |
| D_power_normalizer_LU | `4ac176b0e1291413` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "LU", "random_state": 42, "tol": 0.0}` | `` |
| D_power_normalizer_QR | `None` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "QR", "random_state": 42, "tol": 0.0}` | `InvalidParameterError("The 'power_iteration_normalizer' parameter of TruncatedSVD must be a str among {'auto', 'LU', 'none', 'OR'}. Got 'QR' instead.")` |
| E_n_iter_5 | `4ac176b0e1291413` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}` | `` |
| E_n_iter_7 | `4b025e2694013e96` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 7, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}` | `` |
| E_n_iter_10 | `8e2c4009b6223ba0` | False | `{"algorithm": "randomized", "n_components": 128, "n_iter": 10, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}` | `` |
| F_arpack | `112bdeb4606d9773` | False | `{"algorithm": "arpack", "n_components": 128, "random_state": 42}` | `` |

## Classification
- SVD_NUMERICAL_ENV_NOT_REPRODUCED
