# Beauty Downstream Efficiency Microbenchmark

## Protocol

- Hardware: NVIDIA GeForce RTX 5070 12 GB
- Software: PyTorch 2.11.0+cu128, CUDA 12.8, FP32
- Dataset: Beauty, the same train-only split and item universe
- Seed: 42
- Batch size: 256
- DataLoader workers: 12
- Measurement: four full training epochs with no validation or checkpoint selection;
  epoch 1 is excluded as warm-up and epochs 2--4 are averaged
- Scope: fixed-work downstream training throughput, not time-to-convergence

TIGER and LETTER use their independently reproduced K256 tokenizers. CHORD-K256 is
the capacity-controlled comparison, while CHORD-K1024 is the current Beauty mainline.

## Results

| Method | Capacity per learned level | Training-active params | Inference-active params | Mean sec/epoch | Mean samples/s |
|---|---:|---:|---:|---:|---:|
| TIGER | 256 | 4,590,848 | 4,590,848 | 23.06 | 5,699.85 |
| LETTER | 256 | 8,673,920 | 8,673,920 | 27.88 | 4,713.12 |
| CHORD-K256 | 256 | 9,720,192 | 8,668,288 | 46.67 | 2,815.99 (derived) |
| CHORD-K1024 | 1024 | 10,014,720 | 8,962,816 | 46.67 | 2,815.99 (derived) |

Each epoch contains 131,413 training examples and 514 optimizer steps. CHORD
throughput is derived from the measured epoch time because the Hugging Face Trainer's
reported sample rate uses the configured 100-epoch schedule even when the benchmark
stops at epoch 4.

## Formal Epoch Context

The microbenchmark above measures fixed work per epoch. It does not imply that the
three methods use the same number of epochs in the manuscript.

| Method | Beauty manuscript checkpoint | Configured budget | Epochs reached by the source run |
|---|---:|---:|---:|
| TIGER | 87 | 200 | 89 |
| LETTER | best validation-loss checkpoint 54 | 60 | 60 |
| CHORD-K1024 current formal setting | 60 | schedule total 100, train to 60 | 60 |

TIGER and LETTER epochs above were recovered by exact metric matching against the
RTX 3090 artifacts. Full three-dataset provenance is recorded in
`reports/tiger_letter_manuscript_epoch_audit.md`. Consequently, sec/epoch should be
reported separately from selected checkpoint epoch and configured training budget.

## Parameter Breakdown

| CHORD setting | Core used by inference | PCSC training-only heads | Inactive soft-injection params | All trainable params instantiated |
|---|---:|---:|---:|---:|
| K256 | 8,668,288 | 1,051,904 | 17,280 | 9,737,472 |
| K1024 | 8,962,816 | 1,051,904 | 17,280 | 10,032,000 |

The training-active count excludes the 17,280 soft-injection parameters because the
formal hard-SID downstream does not execute that path. PCSC heads remain present in
the training checkpoint but are not called during autoregressive inference.

## Interpretation

- TIGER is the smallest model because it uses a compact 1,025-token offset vocabulary.
  TIGER and LETTER have the same 4,459,648 non-embedding parameters; LETTER's larger
  total is almost entirely its 32k SentencePiece vocabulary plus SID tokens.
- At matched K256, CHORD and LETTER have essentially the same inference-active parameter
  count. CHORD adds 1,051,904 active parameters during training for PCSC readout heads.
- K1024 increases CHORD inference-active parameters by 294,528 relative to K256, but
  had no measurable effect on the rounded per-epoch time in this run.
- CHORD took about 1.67 times the LETTER epoch time and 2.02 times the TIGER epoch time.
  This is training overhead, not inference overhead.

## Claim Boundary

The benchmark supports:

> CHORD adds training-only cross-view supervision while retaining the standard
> autoregressive inference path.

It does not support claims that CHORD is smaller or faster. The wall-clock comparison
also includes implementation differences: TIGER and LETTER were timed with a minimal
training loop, whereas CHORD used its formal Hugging Face Trainer with PCSC accounting.
For a strict systems claim, all methods would need the same training harness and an
additional inference-throughput benchmark with identical beam and vocabulary capacity.
