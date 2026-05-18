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

## Rerank Threshold Tuning Input

Use `rerank_threshold_eval.csv` for tuning the cross-encoder acceptance threshold.
The tuning script always evaluates the full corpus with source filter `all`.

Required columns:

- `question`
- `expected_relevant`

Optional columns:

- `notes`

`expected_relevant` should be `true` when the corpus should contain an acceptable context for the question, and `false` for out-of-corpus or intentionally unanswerable questions.
