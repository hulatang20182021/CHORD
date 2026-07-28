# Baseline Provenance

The baseline source trees are not vendored. Reproduction used the following
public revisions:

| Baseline | Repository | Revision |
|---|---|---|
| TIGER | https://github.com/XiaoLongtaoo/TIGER | `f1e30af675a8ffec47220f323581e4aed1aabafa` |
| LETTER | https://github.com/HonghuiBao2000/LETTER | `8d0154e28de37dbb6e24871c508ad8ddb1921cda` |
| CoFiRec | https://github.com/YennNing/CoFiRec | `2c2c0ceef06fb85b958444db0497cf416f590a5f` |
| ETEGRec | https://github.com/BishopLiu/ETEGRec | `58d9736afc28e03e190a107a5fa22f5241be6088` |
| SASRec | https://github.com/pmixer/SASRec.pytorch | `fde8a9c08febc2c898dc269b1d98ab762c50d2be` |
| BERT4Rec | https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch | `f66f2534ebfd937778c7174b5f9f216efdebe5de` |

All baselines use the same released user sequences, item universe, and
chronological train/validation/test targets. No baseline uses the old
untraceable `CHORD_dpos_pcsc5_dev/data/*.index.json` as a tokenizer.

TIGER indexes come from newly trained TIGER tokenizers. LETTER uses independently
trained SASRec-32d collaborative embeddings and the official
`alpha=0.01, beta=0.0001` tokenizer setting. The semantic-ID methods are
evaluated by valid-SID generation; SASRec and BERT4Rec directly score catalog
items. Paper tables must label this scoring distinction.

The exact adapted baseline launchers and generated checkpoints remain separate
from CHORD because their upstream licenses and environments differ. The
paper/code appendix should include the protocol patches or command manifests
used for any baseline number claimed as a reproduction.
