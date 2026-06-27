# New-Machine Reproduction Notes

This repository targets CHORD new-machine pipeline-level reproduction. It is not intended to reproduce the old machine's CF-SVD bytes.

Confirmed boundary:

- Raw Beauty data hashes match the sealed context.
- Train-only sequence construction matches.
- Co-occurrence counts and PPMI CSR can be reproduced bit-identically on the new machine.
- `TruncatedSVD` is environment-dependent. The new machine produces `4ac176...` for `Beauty_trainonly_cf_svd.npy`, while the old historical artifact is `6d75...`.

Historical bit-level reproduction requires migrating the old CF-SVD/resource artifact.
