from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from preprocessing import SOURCE_FILTER_ALL, VECTORSTORE_DIR
from retrieval import RERANK_ACCEPT_THRESHOLD, load_vectorstore, retrieve_nearest_chunk


DEFAULT_INPUT_PATH = Path("data/evals/rerank_threshold_eval.csv")
DEFAULT_OUTPUT_DIR = Path("data/evals/results")
DEFAULT_THRESHOLD_START = Decimal("-3.0")
DEFAULT_THRESHOLD_END = Decimal("5.0")
DEFAULT_THRESHOLD_STEP = Decimal("0.25")
DEFAULT_MIN_PRECISION = 0.95


@dataclass(frozen=True)
class TuneRow:
    row_index: int
    question: str
    expected_relevant: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the cross-encoder rerank acceptance threshold using live retrieval."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="CSV with question, expected_relevant, and optional notes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for threshold tuning artifacts.",
    )
    parser.add_argument(
        "--threshold-start",
        type=Decimal,
        default=DEFAULT_THRESHOLD_START,
        help="Lowest threshold to test.",
    )
    parser.add_argument(
        "--threshold-end",
        type=Decimal,
        default=DEFAULT_THRESHOLD_END,
        help="Highest threshold to test.",
    )
    parser.add_argument(
        "--threshold-step",
        type=Decimal,
        default=DEFAULT_THRESHOLD_STEP,
        help="Threshold increment.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help="Preferred minimum precision when selecting a recommended threshold.",
    )
    return parser.parse_args()


def normalize_text(value: Any, *, field: str, row_index: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row_index}: required field '{field}' is missing")

    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_index}: required field '{field}' is empty")
    return text


def parse_bool(value: Any, *, field: str, row_index: int) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        raise ValueError(f"Row {row_index}: required field '{field}' is missing")

    normalized = str(value).strip().lower()
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    if normalized in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(
        f"Row {row_index}: field '{field}' must be true/false, yes/no, or 1/0"
    )


def load_tune_rows(path: Path) -> list[TuneRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"Threshold tuning CSV not found at {path}. Expected columns: "
            "question, expected_relevant, optional notes."
        )

    frame = pd.read_csv(path)
    required_columns = {"question", "expected_relevant"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Threshold tuning CSV is missing required columns: {sorted(missing_columns)}"
        )

    rows: list[TuneRow] = []
    for row_index, row in frame.iterrows():
        rows.append(
            TuneRow(
                row_index=int(row_index),
                question=normalize_text(
                    row["question"],
                    field="question",
                    row_index=int(row_index),
                ),
                expected_relevant=parse_bool(
                    row["expected_relevant"],
                    field="expected_relevant",
                    row_index=int(row_index),
                ),
                notes=""
                if "notes" not in frame.columns or pd.isna(row["notes"])
                else str(row["notes"]).strip(),
            )
        )
    return rows


def run_live_retrieval(rows: list[TuneRow], vectorstore: Any) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        result = retrieve_nearest_chunk(
            row.question,
            vectorstore,
            source_filter=SOURCE_FILTER_ALL,
        )
        samples.append(
            {
                "row_index": row.row_index,
                "question": row.question,
                "source_filter": SOURCE_FILTER_ALL,
                "expected_relevant": row.expected_relevant,
                "notes": row.notes,
                "current_accepted": bool(result["accepted"]),
                "rerank_score": result["rerank_score"],
                "rrf_score": result["rrf_score"],
                "vector_score": result["vector_score"],
                "bm25_score": result["bm25_score"],
                "retrieved_chunk": result["retrieved_chunk"],
                "retrieved_contexts": json.dumps(
                    result.get("retrieved_contexts", []),
                    ensure_ascii=False,
                ),
                "reason": result["reason"],
                "metadata": json.dumps(result.get("metadata", {}), ensure_ascii=False),
            }
        )
    return samples


def threshold_values(start: Decimal, end: Decimal, step: Decimal) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    if start > end:
        raise ValueError("--threshold-start must be less than or equal to --threshold-end")

    values: list[float] = []
    current = start
    while current <= end:
        values.append(float(current))
        current += step
    return values


def confusion_counts(samples: list[dict[str, Any]], threshold: float) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for sample in samples:
        score = sample["rerank_score"]
        predicted_accept = score is not None and float(score) >= threshold
        expected_accept = bool(sample["expected_relevant"])

        if predicted_accept and expected_accept:
            counts["tp"] += 1
        elif predicted_accept and not expected_accept:
            counts["fp"] += 1
        elif not predicted_accept and not expected_accept:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def score_thresholds(
    samples: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(samples)
    for threshold in thresholds:
        counts = confusion_counts(samples, threshold)
        precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_divide(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])
        accuracy = safe_divide(counts["tp"] + counts["tn"], total)

        rows.append(
            {
                "threshold": threshold,
                "accepted_count": counts["tp"] + counts["fp"],
                "rejected_count": counts["tn"] + counts["fn"],
                "true_accepts": counts["tp"],
                "false_accepts": counts["fp"],
                "true_rejects": counts["tn"],
                "false_rejects": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )
    return rows


def choose_threshold(
    sweep_rows: list[dict[str, Any]],
    *,
    min_precision: float,
) -> dict[str, Any]:
    viable_rows = [
        row for row in sweep_rows if row["precision"] >= min_precision and row["accepted_count"] > 0
    ]
    if viable_rows:
        return max(
            viable_rows,
            key=lambda row: (row["recall"], row["f1"], row["precision"], -row["threshold"]),
        )

    return max(
        sweep_rows,
        key=lambda row: (row["f1"], row["precision"], row["recall"], -row["threshold"]),
    )


def write_artifacts(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"rerank_threshold_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(samples).to_csv(run_dir / "samples.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(run_dir / "threshold_sweep.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    load_dotenv()
    args = parse_args()
    rows = load_tune_rows(args.input)
    if not rows:
        raise ValueError("No threshold tuning rows found.")

    vectorstore = load_vectorstore(VECTORSTORE_DIR)
    samples = run_live_retrieval(rows, vectorstore)
    thresholds = threshold_values(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    sweep_rows = score_thresholds(samples, thresholds)
    recommended = choose_threshold(sweep_rows, min_precision=args.min_precision)

    summary = {
        "input_path": str(args.input),
        "vectorstore_path": str(VECTORSTORE_DIR),
        "row_count": len(samples),
        "expected_relevant_count": int(sum(1 for sample in samples if sample["expected_relevant"])),
        "expected_irrelevant_count": int(
            sum(1 for sample in samples if not sample["expected_relevant"])
        ),
        "current_threshold": RERANK_ACCEPT_THRESHOLD,
        "min_precision": args.min_precision,
        "recommended_threshold": recommended["threshold"],
        "recommended_metrics": recommended,
    }

    run_dir = write_artifacts(
        output_dir=args.output_dir,
        samples=samples,
        sweep_rows=sweep_rows,
        summary=summary,
    )

    print(f"Saved threshold tuning artifacts to {run_dir}")
    print(f"Current threshold: {RERANK_ACCEPT_THRESHOLD:.4f}")
    print(f"Recommended threshold: {recommended['threshold']:.4f}")
    print(
        "Recommended metrics: "
        f"precision={recommended['precision']:.4f}, "
        f"recall={recommended['recall']:.4f}, "
        f"f1={recommended['f1']:.4f}, "
        f"false_accepts={recommended['false_accepts']}, "
        f"false_rejects={recommended['false_rejects']}"
    )


if __name__ == "__main__":
    main()
