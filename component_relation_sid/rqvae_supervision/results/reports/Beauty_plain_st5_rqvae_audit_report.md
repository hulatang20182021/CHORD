# Beauty Plain ST5-RQ-VAE Static Audit

```json
{
  "methods": [
    {
      "method": "original",
      "full_sid_duplicate_count": 13,
      "total_token_vocab_size": 965,
      "c1_vocab_size": 197,
      "c2_vocab_size": 256,
      "c3_vocab_size": 256,
      "c4_vocab_size": 256,
      "compact_c4_vocab_size": 256,
      "max_prefix3_bucket_size": 27,
      "index_all_ratio_freq_le_5": 0.046632124352331605,
      "exposure_all_ratio_freq_le_5": 0.007253886010362694,
      "per_position_index_low_freq": [
        0.22842639593908629,
        0.0,
        0.0,
        0.0
      ],
      "per_position_exposure_low_freq": [
        0.03553299492385787,
        0.0,
        0.0,
        0.0
      ],
      "prefix1_mean_bucket_size": 61.4263959390863,
      "prefix2_mean_bucket_size": 1.5257848947169335,
      "prefix3_mean_bucket_size": 1.0346272229822162,
      "c1c2_singleton_ratio": 0.7176900769133779,
      "c1c2c3_singleton_ratio": 0.977000683994528,
      "prefix1_lift": 10.826020719073735,
      "prefix2_lift": 314.48148148148147,
      "prefix3_lift": 1014.0
    },
    {
      "method": "cr_sid_v0",
      "full_sid_duplicate_count": 0,
      "total_token_vocab_size": 846,
      "c1_vocab_size": 256,
      "c2_vocab_size": 256,
      "c3_vocab_size": 256,
      "c4_vocab_size": 78,
      "compact_c4_vocab_size": 78,
      "max_prefix3_bucket_size": 78,
      "index_all_ratio_freq_le_5": 0.10047281323877069,
      "exposure_all_ratio_freq_le_5": 0.001182033096926714,
      "per_position_index_low_freq": [
        0.01953125,
        0.15625,
        0.00390625,
        0.5
      ],
      "per_position_exposure_low_freq": [
        0.0,
        0.0,
        0.0,
        0.01282051282051282
      ],
      "prefix1_mean_bucket_size": 47.26953125,
      "prefix2_mean_bucket_size": 2.8928998326559885,
      "prefix3_mean_bucket_size": 1.6337248548670178,
      "c1c2_singleton_ratio": 0.6363853693521396,
      "c1c2c3_singleton_ratio": 0.8150398271904955,
      "prefix1_lift": 3.163560209424084,
      "prefix2_lift": 4.241610738255034,
      "prefix3_lift": 31.824742268041234
    },
    {
      "method": "v2_st5",
      "full_sid_duplicate_count": 0,
      "total_token_vocab_size": 786,
      "c1_vocab_size": 256,
      "c2_vocab_size": 256,
      "c3_vocab_size": 256,
      "c4_vocab_size": 18,
      "compact_c4_vocab_size": 18,
      "max_prefix3_bucket_size": 18,
      "index_all_ratio_freq_le_5": 0.1984732824427481,
      "exposure_all_ratio_freq_le_5": 0.008905852417302799,
      "per_position_index_low_freq": [
        0.00390625,
        0.5390625,
        0.03125,
        0.5
      ],
      "per_position_exposure_low_freq": [
        0.0,
        0.02734375,
        0.0,
        0.0
      ],
      "prefix1_mean_bucket_size": 47.26953125,
      "prefix2_mean_bucket_size": 1.8503058103975536,
      "prefix3_mean_bucket_size": 1.1470142180094787,
      "c1c2_singleton_ratio": 0.5571865443425077,
      "c1c2c3_singleton_ratio": 0.9014218009478673,
      "prefix1_lift": 13.650406504065039,
      "prefix2_lift": 102.83999999999999,
      "prefix3_lift": 322.66666666666663
    },
    {
      "method": "plain_st5_rqvae",
      "full_sid_duplicate_count": 0,
      "total_token_vocab_size": 377,
      "c1_vocab_size": 105,
      "c2_vocab_size": 167,
      "c3_vocab_size": 90,
      "c4_vocab_size": 15,
      "compact_c4_vocab_size": 15,
      "max_prefix3_bucket_size": 15,
      "index_all_ratio_freq_le_5": 0.07161803713527852,
      "exposure_all_ratio_freq_le_5": 0.002652519893899204,
      "per_position_index_low_freq": [
        0.0380952380952381,
        0.03592814371257485,
        0.15555555555555556,
        0.2
      ],
      "per_position_exposure_low_freq": [
        0.0,
        0.0,
        0.011111111111111112,
        0.0
      ],
      "prefix1_mean_bucket_size": 115.24761904761905,
      "prefix2_mean_bucket_size": 1.9795517749059381,
      "prefix3_mean_bucket_size": 1.0928384358349137,
      "c1c2_singleton_ratio": 0.572059545231474,
      "c1c2c3_singleton_ratio": 0.9353382100605075,
      "prefix1_lift": 8.056208053691275,
      "prefix2_lift": 42.17857142857142,
      "prefix3_lift": 70.66666666666666
    }
  ],
  "label_alignment": {
    "c1_vs_product_type": {
      "nmi": 0.39951497592750757,
      "purity": 0.4346748202627882,
      "entropy_mean": 1.8361263970227317
    },
    "c1_vs_attr_core": {
      "bucket_top_label_overlap": 0.6407129957462053
    },
    "c2_vs_product_type": {
      "nmi": 0.17963526370254787,
      "purity": 0.25948268738120817,
      "entropy_mean": 2.522187544539791
    },
    "c2_vs_attr_core": {
      "bucket_top_label_overlap": 0.5715524671407024
    },
    "c3_vs_product_type": {
      "nmi": 0.09355165175766865,
      "purity": 0.23807949756218494,
      "entropy_mean": 2.5690591729742343
    },
    "c3_vs_attr_core": {
      "bucket_top_label_overlap": 0.5
    },
    "plain_vs_original_code_nmi": {
      "c1": 0.5239515275550283,
      "c2": 0.26662267449810545,
      "c3": 0.15957662865851363
    }
  },
  "gate": {
    "duplicate_ok": true,
    "vocab_ok": true,
    "exposure_ok": true,
    "prefix2_mean_ok": true,
    "prefix_lift_ok": false,
    "recommend_next_supervised_rqvae": false
  }
}
```
