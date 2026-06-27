# Troubleshooting

## Model Missing

Set `paths.model_path` in `configs/beauty_new_machine.yaml` to a complete local Sentence-T5 directory.

## Data Missing

Set `paths.data_root` to a directory containing `Beauty/Beauty.inter.json`, `Beauty/Beauty.index.json`, and `Beauty/Beauty.item.json`.

## ST5 Hash Differs

ST5 generation depends on text construction, model files, device, and library versions. This repo targets new-machine pipeline reproduction, not old ST5 bit reproduction.

## CF-SVD Hash Differs From Old Historical `6d75`

This is expected on the new machine. PPMI can match, while `TruncatedSVD` remains numerical-environment dependent.
