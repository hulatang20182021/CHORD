# SVD Attribution Report

Scope: isolated CF-SVD debug only. No ST5, no PLS, no SID index, no training, no eval.

## PPMI Input
- counts legacy hash: `519847c4f3c8e31f0d94d385f776d3862615a3df5c1a55b2224114d787a55195` match=True
- PPMI legacy CSR hash: `0627d0770a3f817011c861d3f1c63a294c76c33aa627f0a13c32fc8c3a46c63a` match=True
- PPMI shape/nnz: [12101, 12101] / 722608

## SVD
- params: `{"algorithm": "randomized", "n_components": 128, "n_iter": 5, "n_oversamples": 10, "power_iteration_normalizer": "auto", "random_state": 42, "tol": 0.0}`
- sklearn/scipy/numpy: `1.7.2` / `1.15.3` / `2.2.6`
- legacy postprocess: `{"fit_transform_astype_float32": true, "normalize": true, "np_save": true, "signs_or_flips": false}`
- output: `/home/huangxin/llmNrec/reference/legacy_embedding_pipeline_minimal/rebuild_outputs/cf_ppmi_svd_debug/svd_attribution/Beauty_trainonly_cf_svd.npy`
- CF-SVD sha16: `4ac176b0e1291413`
- matches old 6d75: False
- first 8 values: `[0.14318989217281342, -0.12404478341341019, 0.16373476386070251, -0.040133822709321976, 0.0013735407264903188, -0.04400114342570305, -0.10562986135482788, -0.05776644125580788]`

## Old Array Search
- status: OLD_CF_SVD_ARRAY_NOT_FOUND

## Classification
- TRUNCATEDSVD_NUMERICAL_ENVIRONMENT_OR_POSTPROCESSING_DIFFERENCE
