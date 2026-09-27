# Methodology

## Problem framing

Business entity resolution links incomplete and noisy records referring to the same real-world organization. A query may have no match, one match, or several variants. Because F0.5 weights precision more heavily than recall, false merges are particularly costly.

## Two-stage system

An exhaustive comparison against millions of targets is impractical. The solution therefore separates candidate generation from pair classification:

1. Normalize official name and address fields.
2. Partition records by generic country equality.
3. Retrieve candidates independently using name, address, and combined character TF-IDF evidence.
4. Union and deterministically order the top candidates.
5. Generate pairwise similarity and retrieval features.
6. Score pairs with LightGBM.
7. Apply pair and singleton thresholds.

## Normalization

Text is lowercased, Unicode-decomposed, stripped of combining marks, and normalized for punctuation and whitespace. Ampersands are mapped consistently. Numeric tokens are retained because street and unit numbers are highly discriminative.

## Retrieval

Character n-grams tolerate spelling errors, word fragments, abbreviations, and transliteration variation better than exact token matching. Separate name and address retrieval channels prevent a strong field from being hidden by a weak one. A combined channel captures agreement across both fields.

Country partitioning uses the observed string dynamically. It does not contain country-specific branches, so unseen labels receive the same treatment as known labels.

Very common n-grams were removed using corpus document-frequency statistics. This reduced sparse-matrix density and similarity cost without using validation labels. K=20 per channel provided the selected accuracy/runtime trade-off.

## Determinism

Similarity ties are resolved by score descending and target entity ID ascending. Boundary ties are handled explicitly before truncation. This makes results stable across worker order and repeated runs.

## Pair model

The competition model used 34 features covering:

- normalized name and address similarity;
- exact and partial agreements;
- token and character overlap;
- numeric evidence;
- string-length relationships;
- retrieval scores and ranks;
- explicit retrieval-channel presence;
- retrieval-channel consensus.

A compact LightGBM classifier was trained on retrieved positives and hard negatives. The final pair threshold was `0.64`; a separate singleton threshold of `0.66` suppressed weak best-candidate links.

## Validation

Development used a staged process: 5,000-query screening, larger confirmation samples, and realistic 50,000-query evaluation against the full target pool. Validation Source1 labels were excluded from fitting. Metrics included candidate recall, macro F0.5, pair precision/recall, singleton accuracy, ambiguity, runtime, and completeness.

## Scaling

Target vectors were stored as float32 CSR shards. Cached matrices avoided repeated vectorization. Query batches were distributed across isolated workers with one numerical thread each. Atomic manifests, checksums, and completion markers enabled safe checkpoint recovery and deterministic merging.

## Experiments rejected

- Wider global retrieval increased the oracle ceiling but exceeded the deployment budget.
- Multiple speculative blockers added complexity without sufficient measured recovery.
- Peer-consensus features improved local validation but reduced the public score from `0.875` to `0.870`.
- Transitive target expansion was abandoned due to time and precision risk.

## Limitations

Same-country blocking can miss legitimate cross-country links. TF-IDF retrieval remains vulnerable when both name and address are heavily corrupted. A classifier cannot recover a true target absent from its candidate set, and conservative thresholds may miss secondary links in multi-match groups.

## Reproducibility and disclosure

The public demo preserves the same architectural ideas but uses a smaller feature set and entirely synthetic data. Its metrics are demonstration results and must not be confused with the competition score. No official data, submission output, cache, trained competition artifact, or external entity data is distributed.

