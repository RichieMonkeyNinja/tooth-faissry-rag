# RAGAS Evaluation Input

Put your human-curated evaluation rows in `qa_eval.csv`.

Required columns:

- `question`
- `reference`
- `reference_contexts`

Optional columns:

- `source_filter` with values `all`, `pdf`, or `medmcqa`

`reference_contexts` must be a JSON array string, for example:

```json
["The key supporting passage.", "A second supporting passage."]
```

Evaluation outputs are written under `data/evals/results/` in timestamped subdirectories.
