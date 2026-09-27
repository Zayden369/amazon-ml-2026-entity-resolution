"""Train and evaluate a compact two-stage entity-resolution demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sparse_dot_topn import sp_matmul_topn

from .metric import evaluate
from .normalization import normalize_text


def load_tsv(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / name, sep="\t", dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    queries = load_tsv(args.data, "train_source1.tsv")
    targets = pd.concat(
        [load_tsv(args.data, "train_source2.tsv"), load_tsv(args.data, "train_source3.tsv")],
        ignore_index=True,
    )
    ground_truth = load_tsv(args.data, "train_ground_truth.tsv")
    truth = {
        row.source1_entity_id: set(filter(None, row.matched_entity_ids.split(",")))
        for row in ground_truth.itertuples()
    }
    indices = np.arange(len(queries))
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=args.seed)
    query_ids = queries.entity_id.tolist()
    target_ids = targets.entity_id.tolist()

    pair_data: dict[tuple[int, int], list[float]] = {}
    normalized: dict[str, tuple[list[str], list[str]]] = {}
    vectorizers: dict[str, TfidfVectorizer] = {}
    for field in ("business_name", "business_address"):
        query_text = queries[field].map(normalize_text).tolist()
        target_text = targets[field].map(normalize_text).tolist()
        normalized[field] = (query_text, target_text)
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32, min_df=1)
        vectorizer.fit([query_text[i] for i in train_indices] + target_text)
        vectorizers[field] = vectorizer
        query_matrix = vectorizer.transform(query_text)
        target_matrix = vectorizer.transform(target_text)
        offset = 0 if field == "business_name" else 1
        for country in sorted(set(queries.country)):
            query_rows = np.flatnonzero(queries.country.eq(country).to_numpy())
            target_rows = np.flatnonzero(targets.country.eq(country).to_numpy())
            if not len(query_rows) or not len(target_rows):
                continue
            retrieved = sp_matmul_topn(
                query_matrix[query_rows], target_matrix[target_rows].T.tocsr(),
                top_n=args.top_k, threshold=0.0, sort=True, n_threads=1,
            )
            for local_query, query_index in enumerate(query_rows):
                low, high = retrieved.indptr[local_query:local_query + 2]
                for rank, (local_target, score) in enumerate(
                    zip(retrieved.indices[low:high], retrieved.data[low:high]), start=1
                ):
                    target_index = int(target_rows[local_target])
                    values = pair_data.setdefault(
                        (int(query_index), target_index), [0.0, 0.0, args.top_k + 1.0, args.top_k + 1.0]
                    )
                    values[offset] = float(score)
                    values[offset + 2] = float(rank)

    query_index = np.array([pair[0] for pair in pair_data], dtype=np.int64)
    target_index = np.array([pair[1] for pair in pair_data], dtype=np.int64)
    retrieval = np.asarray(list(pair_data.values()), dtype=np.float32)
    labels = np.array(
        [target_ids[j] in truth[query_ids[i]] for i, j in zip(query_index, target_index)], dtype=np.int8
    )
    features = [retrieval]
    for field in ("business_name", "business_address"):
        query_text, target_text = normalized[field]
        features.append(np.column_stack([
            np.array([fuzz.ratio(query_text[i], target_text[j]) / 100 for i, j in zip(query_index, target_index)]),
            np.array([fuzz.token_sort_ratio(query_text[i], target_text[j]) / 100 for i, j in zip(query_index, target_index)]),
            np.array([fuzz.token_set_ratio(query_text[i], target_text[j]) / 100 for i, j in zip(query_index, target_index)]),
            np.array([query_text[i] == target_text[j] for i, j in zip(query_index, target_index)]),
        ]))
    feature_matrix = np.column_stack(features).astype(np.float32)
    training_pairs = np.isin(query_index, train_indices)
    model = LGBMClassifier(
        n_estimators=180, num_leaves=31, learning_rate=0.06,
        random_state=args.seed, n_jobs=1, verbosity=-1,
    )
    model.fit(feature_matrix[training_pairs], labels[training_pairs])
    probabilities = model.predict_proba(feature_matrix)[:, 1]

    predictions = {query_ids[i]: set() for i in test_indices}
    test_set = set(map(int, test_indices))
    for row in np.flatnonzero(probabilities >= 0.5):
        query = int(query_index[row])
        if query in test_set:
            predictions[query_ids[query]].add(target_ids[int(target_index[row])])
    report = evaluate({key: truth[key] for key in predictions}, predictions)
    report.update({
        "queries": len(queries),
        "targets": len(targets),
        "candidate_pairs": len(pair_data),
        "features": feature_matrix.shape[1],
        "top_k": args.top_k,
        "note": "Synthetic demonstration only; not the competition score.",
    })
    model.booster_.save_model(str(args.output / "model.txt"))
    joblib.dump(vectorizers, args.output / "vectorizers.joblib")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
