"""Generate a tiny, entirely synthetic entity-resolution dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


NOUNS = ["Orchid", "Nimbus", "Pioneer", "Maple", "Cobalt", "Sunrise", "Vertex", "Harbor"]
KINDS = ["Foods", "Logistics", "Pharmacy", "Textiles", "Consulting", "Hardware"]
STREETS = ["Market Road", "Lake Street", "Station Avenue", "Garden Lane", "Hill Road"]
COUNTRIES = ["Northland", "Southland", "Westland"]


def noisy(text: str, rng: random.Random) -> str:
    replacements = [("Limited", "Ltd"), ("Road", "Rd"), ("Street", "St"), ("Avenue", "Ave")]
    for original, replacement in replacements:
        if rng.random() < 0.5:
            text = text.replace(original, replacement)
    if rng.random() < 0.35:
        text = text.lower()
    if rng.random() < 0.25:
        text = text.replace(" ", "  ", 1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entities", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    source1, source2, source3, truth = [], [], [], []
    s2_index = s3_index = 0
    for index in range(args.entities):
        country = COUNTRIES[index % len(COUNTRIES)]
        name = f"{NOUNS[index % len(NOUNS)]} {KINDS[(index // len(NOUNS)) % len(KINDS)]} Limited"
        address = f"{100 + index} {STREETS[index % len(STREETS)]}, District {index % 17}"
        query_id = f"S1-{index:05d}"
        source1.append((query_id, name, address, country))
        matches: list[str] = []
        if index % 7 != 0:
            target_id = f"S2-{s2_index:05d}"
            s2_index += 1
            source2.append((target_id, noisy(name, rng), noisy(address, rng), country))
            matches.append(target_id)
        if index % 5 == 0:
            target_id = f"S3-{s3_index:05d}"
            s3_index += 1
            source3.append((target_id, noisy(name, rng), noisy(address, rng), country))
            matches.append(target_id)
        truth.append((query_id, ",".join(matches)))

    # Add deterministic distractors in every country.
    for index in range(max(60, args.entities // 2)):
        country = COUNTRIES[index % len(COUNTRIES)]
        source2.append((f"S2-{s2_index:05d}", f"Independent Shop {index}", f"{9000 + index} Remote Way", country))
        source3.append((f"S3-{s3_index:05d}", f"Unrelated Services {index}", f"{7000 + index} Other Road", country))
        s2_index += 1
        s3_index += 1

    columns = ["entity_id", "business_name", "business_address", "country"]
    pd.DataFrame(source1, columns=columns).to_csv(args.output / "train_source1.tsv", sep="\t", index=False)
    pd.DataFrame(source2, columns=columns).to_csv(args.output / "train_source2.tsv", sep="\t", index=False)
    pd.DataFrame(source3, columns=columns).to_csv(args.output / "train_source3.tsv", sep="\t", index=False)
    pd.DataFrame(truth, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        args.output / "train_ground_truth.tsv", sep="\t", index=False
    )
    print(f"Wrote synthetic dataset to {args.output}")


if __name__ == "__main__":
    main()

