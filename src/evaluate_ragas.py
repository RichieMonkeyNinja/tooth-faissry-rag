from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from preprocessing import (
    EMBEDDING_MODEL,
    SOURCE_FILTER_ALL,
    SOURCE_FILTER_OPTIONS,
    VECTORSTORE_DIR,
)
from retrieval import (
    NO_GROUNDED_ANSWER_MESSAGE,
    answer_question,
    load_vectorstore,
)


EVAL_INPUT_PATH = Path("data/evals/qa_eval.csv")
EVAL_OUTPUT_DIR = Path("data/evals/results")
EVAL_JUDGE_LLM_MODEL = "gpt-4o"

DEFAULT_METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")

METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}


@dataclass(frozen=True)
class EvalRow:
    row_index: int
    question: str
    reference: str
    reference_contexts: list[str]
    source_filter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of eval rows to run.",
    )
    parser.add_argument(
        "--source-filter",
        choices=SOURCE_FILTER_OPTIONS,
        default=None,
        help="Optional global source filter override for every eval row.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRIC_LABELS.keys()),
        default=list(DEFAULT_METRICS),
        help="Subset of metrics to compute. Defaults to all four RAGAS metrics.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=EVAL_INPUT_PATH,
        help="Curated evaluation CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_OUTPUT_DIR,
        help="Directory for evaluation artifacts.",
    )
    return parser.parse_args()


def normalize_text(value: Any, *, field: str, row_index: int) -> str:
    if pd.isna(value):
        raise ValueError(f"Row {row_index}: required field '{field}' is missing")

    text = str(value).strip()
    if not text:
        raise ValueError(f"Row {row_index}: required field '{field}' is empty")
    return text


def parse_reference_contexts(value: Any, *, row_index: int) -> list[str]:
    if pd.isna(value):
        raise ValueError(f"Row {row_index}: required field 'reference_contexts' is missing")

    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Row {row_index}: reference_contexts must be a JSON array string"
        ) from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Row {row_index}: reference_contexts must be a non-empty JSON array")

    contexts: list[str] = []
    for item_index, item in enumerate(parsed):
        if not isinstance(item, str):
            raise ValueError(
                f"Row {row_index}: reference_contexts[{item_index}] must be a string"
            )
        context = item.strip()
        if not context:
            raise ValueError(
                f"Row {row_index}: reference_contexts[{item_index}] must not be empty"
            )
        contexts.append(context)

    return contexts


def parse_source_filter(value: Any, *, row_index: int, override: str | None) -> str:
    if override is not None:
        return override

    if pd.isna(value) or str(value).strip() == "":
        return SOURCE_FILTER_ALL

    source_filter = str(value).strip().lower()
    if source_filter not in SOURCE_FILTER_OPTIONS:
        raise ValueError(
            f"Row {row_index}: source_filter must be one of {SOURCE_FILTER_OPTIONS}"
        )
    return source_filter


def load_eval_rows(path: Path, *, source_filter_override: str | None) -> list[EvalRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found at {path}. Create it with question, reference, "
            "reference_contexts, and optional source_filter columns."
        )

    frame = pd.read_csv(path)
    required_columns = {"question", "reference", "reference_contexts"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation CSV is missing required columns: {sorted(missing_columns)}"
        )

    eval_rows: list[EvalRow] = []
    for row_index, row in frame.iterrows():
        question = normalize_text(row["question"], field="question", row_index=row_index)
        reference = normalize_text(row["reference"], field="reference", row_index=row_index)
        reference_contexts = parse_reference_contexts(row["reference_contexts"], row_index=row_index)
        source_filter = parse_source_filter(
            row["source_filter"] if "source_filter" in frame.columns else None,
            row_index=row_index,
            override=source_filter_override,
        )

        eval_rows.append(
            EvalRow(
                row_index=int(row_index),
                question=question,
                reference=reference,
                reference_contexts=reference_contexts,
                source_filter=source_filter,
            )
        )

    return eval_rows


def build_metrics(
    metric_names: list[str],
    judge_llm: Any,
    judge_embeddings: Any,
) -> list[Any]:
    metrics: list[Any] = []
    for metric_name in metric_names:
        if metric_name == "context_precision":
            metrics.append(
                ContextPrecisionWithReference(llm=judge_llm, name="context_precision")
            )
        elif metric_name == "context_recall":
            metrics.append(ContextRecall(llm=judge_llm, name="context_recall"))
        elif metric_name == "faithfulness":
            metrics.append(Faithfulness(llm=judge_llm, name="faithfulness"))
        elif metric_name == "answer_relevancy":
            metrics.append(
                AnswerRelevancy(
                    llm=judge_llm,
                    embeddings=judge_embeddings,
                    name="answer_relevancy",
                )
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    return metrics


def build_sample(row: EvalRow, vectorstore: Any) -> dict[str, Any]:
    result = answer_question(
        row.question,
        vectorstore,
        source_filter=row.source_filter,
    )

    retrieved_contexts = result.get("retrieved_contexts") or []
    if not retrieved_contexts and result.get("retrieved_chunk"):
        retrieved_contexts = [str(result["retrieved_chunk"])]

    assistant_response = (
        result["answer"] if result["accepted"] else NO_GROUNDED_ANSWER_MESSAGE
    )

    return {
        "row_index": row.row_index,
        "question": row.question,
        "reference": row.reference,
        "reference_contexts": json.dumps(row.reference_contexts, ensure_ascii=False),
        "source_filter": row.source_filter,
        "accepted": bool(result["accepted"]),
        "assistant_response": assistant_response,
        "generated_answer": result["answer"],
        "retrieved_chunk": result["retrieved_chunk"],
        "retrieved_contexts": json.dumps(retrieved_contexts, ensure_ascii=False),
        "score": result["score"],
        "vector_score": result["vector_score"],
        "bm25_score": result["bm25_score"],
        "rrf_score": result["rrf_score"],
        "rerank_score": result["rerank_score"],
        "reason": result["reason"],
        "metadata": json.dumps(result.get("metadata", {}), ensure_ascii=False),
    }


def build_metric_inputs(metric_name: str, traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for trace in traces:
        retrieved_contexts = json.loads(trace["retrieved_contexts"])
        if metric_name in {"context_precision", "context_recall"}:
            inputs.append(
                {
                    "user_input": trace["question"],
                    "reference": trace["reference"],
                    "retrieved_contexts": retrieved_contexts,
                }
            )
        elif metric_name == "faithfulness":
            inputs.append(
                {
                    "user_input": trace["question"],
                    "response": trace["assistant_response"],
                    "retrieved_contexts": retrieved_contexts,
                }
            )
        elif metric_name == "answer_relevancy":
            inputs.append(
                {
                    "user_input": trace["question"],
                    "response": trace["assistant_response"],
                }
            )
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    return inputs


def score_metrics(
    *,
    traces: list[dict[str, Any]],
    metrics: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    metric_scores: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        metric_inputs = build_metric_inputs(metric.name, traces)
        results = metric.batch_score(metric_inputs)
        metric_scores[metric.name] = [
            {"value": float(result.value), "reason": result.reason}
            for result in results
        ]
    return metric_scores


def write_artifacts(
    *,
    output_dir: Path,
    traces: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"ragas_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    traces_frame = pd.DataFrame(traces)
    traces_frame.to_csv(run_dir / "samples.csv", index=False)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    load_dotenv()
    args = parse_args()

    eval_rows = load_eval_rows(args.input, source_filter_override=args.source_filter)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        eval_rows = eval_rows[: args.limit]

    if not eval_rows:
        raise ValueError("No evaluation rows found after filtering.")

    vectorstore = load_vectorstore(VECTORSTORE_DIR)

    openai_client = AsyncOpenAI()
    judge_llm = llm_factory(EVAL_JUDGE_LLM_MODEL, client=openai_client)
    judge_embeddings = RagasOpenAIEmbeddings(client=openai_client, model=EMBEDDING_MODEL)
    metrics = build_metrics(args.metrics, judge_llm, judge_embeddings)

    traces = [build_sample(row, vectorstore) for row in eval_rows]
    metric_scores = score_metrics(traces=traces, metrics=metrics)
    metric_columns = [metric.name for metric in metrics]
    for metric_name in metric_columns:
        for trace, score in zip(traces, metric_scores[metric_name], strict=True):
            trace[metric_name] = score["value"]
            trace[f"{metric_name}_reason"] = score["reason"]

    metric_frame = pd.DataFrame(
        {metric_name: [score["value"] for score in metric_scores[metric_name]] for metric_name in metric_columns}
    )
    summary_metrics = {}
    for column in metric_columns:
        summary_metrics[column] = float(pd.to_numeric(metric_frame[column], errors="coerce").mean())

    summary = {
        "input_path": str(args.input),
        "vectorstore_path": str(VECTORSTORE_DIR),
        "output_metrics": metric_columns,
        "metric_labels": {name: METRIC_LABELS[name] for name in metric_columns},
        "row_count": len(traces),
        "accepted_count": int(sum(1 for trace in traces if trace["accepted"])),
        "rejected_count": int(sum(1 for trace in traces if not trace["accepted"])),
        "source_filter_override": args.source_filter,
        "limit": args.limit,
        "judge_llm_model": EVAL_JUDGE_LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "metric_means": summary_metrics,
    }

    run_dir = write_artifacts(
        output_dir=args.output_dir,
        traces=traces,
        summary=summary,
    )

    print(f"Saved evaluation artifacts to {run_dir}")
    print("Metric means:")
    for metric_name in metric_columns:
        print(f"- {metric_name}: {summary_metrics[metric_name]:.4f}")


if __name__ == "__main__":
    main()
