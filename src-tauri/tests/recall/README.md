# Recall Test Framework

Tests Recall@K metrics for patch-based image search.

## Dataset Structure

```
tests/recall/
├── ground_truth.yaml
├── data/
│   ├── queries/
│   │   ├── query_001.jpg
│   │   └── query_002.jpg
│   └── database/
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
```

## Ground Truth Format

```yaml
query_001.jpg:
  relevant:
    - img_001.jpg
    - img_003.jpg
  irrelevant:
    - unrelated.jpg

query_002.jpg:
  relevant:
    - img_002.jpg
```

## Usage

1. Create dataset structure:
```rust
use recall_test::{RecallEvaluator, RecallConfig, create_dataset_structure};

let base_path = Path::new("tests/recall/data");
create_dataset_structure(base_path).unwrap();
```

2. Add images:
   - Place query images in `tests/recall/data/queries/`
   - Place database images in `tests/recall/data/database/`

3. Update ground truth in `tests/recall/ground_truth.yaml`

4. Run recall test:
```rust
let config = RecallConfig {
    dataset_path: "tests/recall/data".to_string(),
    ground_truth_path: "tests/recall/ground_truth.yaml".to_string(),
    superpoint_path: "resources/models/superpoint.onnx".to_string(),
    lightglue_path: "resources/models/lightglue.mnn".to_string(),
    ..Default::default()
};

let mut evaluator = RecallEvaluator::new(config).unwrap();
let results = evaluator.run_coarse_recall_test(&patch_search, 10).unwrap();
evaluator.print_results(&results);
```

## Metrics

- **Recall@K**: % of queries where relevant images appear in top K results
- **Top-1 Accuracy**: % of queries where top result is relevant
- **MRR**: Mean Reciprocal Rank
- **Coverage**: % of queries with at least one relevant result found