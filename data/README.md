# Data

No Amazon ML Challenge data is distributed with this repository.

To run the public demo, generate a small synthetic dataset:

```bash
python examples/generate_synthetic_data.py --output data/synthetic
```

The training CLI expects four tab-separated files:

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

Source files contain `entity_id`, `business_name`, `business_address`, and
`country`. Ground truth contains `source1_entity_id` and `matched_entity_ids`.

