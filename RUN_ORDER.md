# CHORD Paper Run Order

The paper entry point is `scripts/run_paper_main.sh`. It performs the following
stages without overwriting existing artifacts:

1. Verify the released dataset files and item universe.
2. Build normalized Sentence-T5 item embeddings.
3. Build equal-weight, train-only co-occurrence and PPMI-SVD features.
4. Fit the two directional MLP predictors and construct prediction gaps.
5. fit PLS consensus and three independent KMeans codebooks.
6. Generate the collision-free `consensus, sem-gap, CF-gap, DOCS` index.
7. Build downstream sequence data.
8. Train the shared-anchor five-objective model to the predeclared epoch.
9. Evaluate the fixed checkpoint once with trie-constrained beam-20 decoding.

Setup:

```bash
bash scripts/setup/download_chord_data.sh
bash scripts/setup/download_sentence_t5.sh
bash scripts/setup/download_letter_runtime.sh
conda env create -f environment.yml
conda activate chord-paper
```

Main runs:

```bash
for dataset in Beauty Instruments Yelp; do
  for seed in 42 1000 2026; do
    DATASET="$dataset" SEED="$seed" GPU=0 bash scripts/run_paper_main.sh
  done
done
```

The dataset-specific formal epochs are declared in
`configs/paper_k1024.env`. Diagnostic epoch sweeps must use a different
`RUN_SUFFIX`; they must not overwrite or select the formal test checkpoint.
