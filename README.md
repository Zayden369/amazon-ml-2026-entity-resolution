# Scalable Business Entity Resolution with TF-IDF and LightGBM

A portfolio-safe reconstruction of my Amazon ML Challenge 2026 solution. The project links noisy business records across independent sources using a two-stage retrieval-and-ranking pipeline.

> **Competition result:** the selected LB_V01 submission scored **0.875** on the public leaderboard. This repository reports that result honestly and does not include challenge data, submitted predictions, or private trained artifacts.

## What this project demonstrates

- Unicode-aware normalization for noisy company names and addresses
- Generic same-country partitioning with no hard-coded country list
- Sparse character n-gram TF-IDF retrieval
- Deterministic top-K candidate generation
- Pairwise fuzzy, token, numeric, and retrieval features
- LightGBM classification optimized for precision-heavy F0.5
- Explicit singleton handling
- Batched, checkpointable inference for multi-million-row target pools

## Architecture

```text
Source1 query
    |
    v
generic country partition
    |
    v
name TF-IDF + address TF-IDF + combined retrieval
    |
    v
union and deterministic top-K candidates
    |
    v
pair features + LightGBM
    |
    v
pair threshold + singleton threshold
    |
    +--> matching_results.tsv
    +--> candidate_pairs.tsv
```

The competition system used K=20 per retrieval channel, strong corpus document-frequency pruning, a frozen 34-feature LightGBM model, pair threshold `0.64`, and singleton threshold `0.66`.

## Run the synthetic demo

The demo is intentionally trained on generated fictional records, so it is safe to run and redistribute.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .

python examples/generate_synthetic_data.py --output data/synthetic
python -m entity_resolution.train_demo --data data/synthetic --output artifacts/demo
```

The second command writes:

- `artifacts/demo/model.txt`
- `artifacts/demo/vectorizers.joblib`
- `artifacts/demo/metrics.json`

These generated files are ignored by Git.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
.
├── data/README.md
├── examples/generate_synthetic_data.py
├── src/entity_resolution/
│   ├── metric.py
│   ├── normalization.py
│   └── train_demo.py
├── tests/test_core.py
├── METHODOLOGY.md
├── requirements.txt
└── pyproject.toml
```

## Results and lessons

The strongest realistic 50,000-query validation baseline reached macro F0.5 `0.909313`; the selected leaderboard submission scored `0.875` publicly. A later peer-consensus experiment improved local validation but scored `0.870` publicly, so it was rejected. Wider retrieval offered a better oracle ceiling but was not computationally viable within the competition deadline.

The main lesson was that entity resolution is constrained as much by candidate recall and calibrated singleton decisions as by classifier quality. Runtime engineering—sparse matrices, pruning, caching, batching, deterministic top-K, and resumable shards—was essential at this scale.

## Data and fair-use notice

No challenge dataset, prediction file, candidate file, cache, credential, or external business data is included. To reproduce the original competition result, an authorized participant must obtain the official data through the competition portal and comply with its terms. The included demo uses only synthetic fictional records.

## License

The original code in this repository is released under the [MIT License](LICENSE). This license does not grant rights to any Amazon or Unstop datasets, names, trademarks, or competition materials.
