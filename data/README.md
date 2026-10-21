# Data

`amazon_splits.zip` contains the leave-one-out splits of four Amazon review datasets (5-core):
Baby; Sports and Outdoors; Clothing, Shoes and Jewelry; Electronics.

- Columns: `user_id`, `item_id` (re-indexed integers), `rating`, `timestamp` (day resolution).
- Per user: the last interaction is in `test.csv`, the second last in `val.csv`, the rest in `train.csv`.
  Interactions with equal timestamps keep the order of the source files.
- Sources: J. McAuley, C. Targett, Q. Shi, and A. van den Hengel, SIGIR 2015; R. He and J. McAuley, WWW 2016;
  5-core files as distributed with the MMRec benchmark (X. Zhou, 2023). For research use only.

MovieLens-1M (F. M. Harper and J. A. Konstan, TiiS 2015) is not redistributed. Download `ml-1m.zip` from
https://grouplens.org/datasets/movielens/1m/ and run

```bash
python -I data/prep_ml1m.py ml-1m.zip data/splits/ml1m
```

Every rating becomes an implicit interaction. `python data/verify_splits.py` checks all splits and prints
their SHA-256 hashes, which must match `SHA256SUMS.txt`.
