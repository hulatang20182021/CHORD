# Third-Party Notices

CHORD uses or adapts components from the following research repositories.

## LETTER

- Project: Learnable Item Tokenization for Generative Recommendation
- Repository: https://github.com/HonghuiBao2000/LETTER
- Audited upstream revision: `8d0154e28de37dbb6e24871c508ad8ddb1921cda`
- Used for: released data format, T5-style downstream runtime, collator, and
  trie-constrained generation interfaces.

The upstream repository did not contain an explicit license file at the
audited revision. This repository does not relicense LETTER-originated code.
Users should consult the upstream authors and repository terms before uses
beyond research reproduction.

## TIGER

- Project: unofficial PyTorch implementation of Recommender Systems with
  Generative Retrieval
- Repository: https://github.com/XiaoLongtaoo/TIGER
- Audited upstream revision: `f1e30af675a8ffec47220f323581e4aed1aabafa`
- License: MIT
- Used for: independently reproduced TIGER baseline artifacts and protocol
  comparison. TIGER source is not vendored in this repository.

## Sentence-T5

- Model: `sentence-transformers/sentence-t5-base`
- Used for: semantic item embeddings.
- The downloaded model retains its model-card and upstream licensing terms.

## Datasets

Beauty, Instruments, and Yelp are existing datasets. The release archive
contains the preprocessed files used by this project following LETTER's
published data format and splits. It does not alter ownership or licensing of
the underlying review and Yelp data.
