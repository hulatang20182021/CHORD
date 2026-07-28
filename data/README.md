# Data

Download the preprocessed Beauty, Instruments, and Yelp files used by the paper:

```bash
bash scripts/setup/download_chord_data.sh
```

The script verifies SHA256
`c21bd634f16b93732cccf0de74d8170485bff857d83ec910821cc3b817304c08`
and extracts:

```text
data/Beauty/{Beauty.inter.json,Beauty.item.json,Beauty.index.json}
data/Instruments/{Instruments.inter.json,Instruments.item.json,Instruments.index.json}
data/Yelp/{Yelp.inter.json,Yelp.item.json,Yelp.index.json}
```

These are existing datasets released in the format and split used by LETTER.
The final interaction per user is test, the preceding interaction is
validation, and earlier interactions are training. CHORD constructs every
collaborative resource only from the training portion.

The released `<Dataset>.index.json` fixes the item universe and mapping. It is
not an RQ/TIGER/LETTER baseline tokenizer artifact. CHORD always writes a new
SID index under `RESULT_BASE/index/`.

The archive does not change ownership or licensing of the underlying datasets.
See `THIRD_PARTY_NOTICES.md` and cite the original datasets and LETTER in
derived publications.
