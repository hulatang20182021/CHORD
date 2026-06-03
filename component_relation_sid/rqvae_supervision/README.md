# Component-Relation RQ-VAE Supervision

This directory contains the P0 tokenizer rebuild stage for Component-Relation SID.

The current stage moves from post-hoc SID rewriting to tokenizer training:

1. Prepare Beauty Sentence-T5 full-text embeddings as RQ-VAE input.
2. Prepare head/product_type/attribute/relation labels for later supervised RQ-VAE.
3. Train a plain ST5-RQ-VAE tokenizer.
4. Generate a Beauty SID index from the plain tokenizer.
5. Run static audits against project original SID, CR-SID V0, and V2-ST5.

It does not train LETTER-TIGER and does not modify `RQ-VAE/`.

Run:

```bash
cd /home/huangxin/llmNrec/Letter/LETTER-master
bash component_relation_sid/rqvae_supervision/scripts/run_beauty_plain_st5_rqvae_pipeline.sh emotion_ml1m 2 2024
```

If GPU2 is occupied, the training launcher exits and asks before using GPU3.
