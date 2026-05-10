import argparse
from functools import lru_cache
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from typing import Any

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import DATA_PATH


VECTORSTORE_DIR = Path("data/vectorstores/medmcqa")
RELEVANCE_THRESHOLD = 0.6
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_RETRIEVAL_K = 8
BM25_RETRIEVAL_K = 8
HYBRID_ALPHA = 0.5

def softmax_normalize(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}

    keys = list(scores.keys())
    values = np.array([scores[key] for key in keys], dtype=float) / temperature
    exp_values = np.exp(values - values.max())
    softmax_values = exp_values / exp_values.sum()
    return {key: float(value) for key, value in zip(keys, softmax_values, strict=True)}

def load_csv_documents(path: str | Path) -> list[Document]:
    df = pd.read_csv(path)
    documents: list[Document] = []

    for row_index, row in df.iterrows():
        documents.append(
            Document(
                page_content=(
                    f"Question: {row['question']}\n"
                    f"Correct Answer Text: {row['correct_ans']}\n"
                    # f"Explanation: {row['exp']}"
                )
                # metadata={
                #     "source": str(path),
                #     "row_index": int(row_index),
                #     "question": row["question"],
                #     "correct_answer": row["correct_ans"],
                #     "doc_type": "medmcqa",
                # },
            )
        )

    return documents

def load_vectorstore(vectorstore_dir: str | Path) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def load_keyword_retriever(data_path: str | Path = DATA_PATH) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(load_csv_documents(data_path))
    retriever.k = BM25_RETRIEVAL_K
    return retriever


@lru_cache(maxsize=1)
def load_reranker(
    model_name: str = RERANKER_MODEL,
) -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def rerank_documents(question: str, documents: list[Document]) -> list[tuple[Document, float]]:
    if not documents:
        return []

    tokenizer, model = load_reranker()
    pairs = [(question, document.page_content) for document in documents]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(**inputs).logits.squeeze(-1)

    scores = logits.tolist()
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [(document, float(score)) for document, score in ranked]

def retrieve_nearest_chunk(
    question: str,
    vectorstore: FAISS,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, Any]:
    vector_results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=INITIAL_RETRIEVAL_K,
    )
    bm25_retriever = load_keyword_retriever()
    bm25_results = bm25_retriever.invoke(question)

    if not vector_results and not bm25_results:
        return {
            "answer": None,
            "retrieved_chunk": None,
            "score": 0.0,
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "bm25_score_softmax": 0.0,
            "hybrid_score": 0.0,
            "rerank_score": None,
            "accepted": False,
            "reason": "Hybrid retrieval returned no results",
        }

    vector_scores_by_chunk = {
        document.page_content: float(score) for document, score in vector_results
    }
    bm25_scores_by_chunk: dict[str, float] = {}
    merged_documents: dict[str, Document] = {}

    for document, _ in vector_results:
        chunk = document.page_content
        merged_documents[chunk] = document

    for rank, document in enumerate(bm25_results, start=1):
        chunk = document.page_content
        merged_documents.setdefault(chunk, document)
        bm25_scores_by_chunk[chunk] = float(BM25_RETRIEVAL_K - rank + 1)

    vector_scores_for_candidates = {
        chunk: vector_scores_by_chunk.get(chunk, 0.0) for chunk in merged_documents
    }
    bm25_scores_for_candidates = {
        chunk: bm25_scores_by_chunk.get(chunk, 0.0) for chunk in merged_documents
    }
    softmax_bm25_scores = softmax_normalize(bm25_scores_for_candidates)
    hybrid_scores_by_chunk = {
        chunk: HYBRID_ALPHA * softmax_bm25_scores.get(chunk, 0.0)
        + (1.0 - HYBRID_ALPHA) * vector_scores_for_candidates.get(chunk, 0.0)
        for chunk in merged_documents
    }

    candidate_documents = sorted(
        merged_documents.values(),
        key=lambda document: hybrid_scores_by_chunk.get(document.page_content, 0.0),
        reverse=True,
    )
    reranked_results = rerank_documents(question, candidate_documents)
    document, rerank_score = reranked_results[0]
    vector_score = vector_scores_by_chunk.get(document.page_content, 0.0)
    bm25_score = bm25_scores_by_chunk.get(document.page_content, 0.0)
    bm25_score_softmax = softmax_bm25_scores.get(document.page_content, 0.0)
    hybrid_score = hybrid_scores_by_chunk.get(document.page_content, 0.0)
    accepted = (
    rerank_score >= 0.0
    and (vector_score >= 0.5 or bm25_score >= 5.0)
    )

    return {
        "answer": None,
        "retrieved_chunk": document.page_content,
        "score": float(hybrid_score),
        "vector_score": float(vector_score),
        "bm25_score": float(bm25_score),
        "bm25_score_softmax": float(bm25_score_softmax),
        "hybrid_score": float(hybrid_score),
        "rerank_score": float(rerank_score),
        "accepted": accepted,
        "reason": None if accepted else "No sufficiently similar context found in vector or keyword retrieval",
        "metadata": document.metadata,
    }


def generate_grounded_answer(question: str, retrieved_chunk: str) -> str:
    llm = ChatOpenAI(model=LLM_MODEL)
    messages = [
        SystemMessage(
            content=(
                "Rewrite the retrieved explanation into exactly one coherent sentence "
                "that answers the user's question using only the provided context. "
                "Do not add facts, assumptions, outside medical knowledge, caveats, "
                "or bullet points."
            )
        ),
        HumanMessage(
            content=(
                f"User question:\n{question}\n\n"
                f"Retrieved context:\n{retrieved_chunk}\n\n"
                "Return one sentence only."
            )
        ),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def answer_question(
    question: str,
    vectorstore: FAISS,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, Any]:
    result = retrieve_nearest_chunk(question, vectorstore, threshold=threshold)

    if not result["accepted"]:
        return result

    result["answer"] = generate_grounded_answer(question, result["retrieved_chunk"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to answer. If omitted, retrieval.py reads from interactive input.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.question:
        question = " ".join(args.question).strip()
    elif sys.stdin.isatty():
        try:
            question = input("Enter your question: ").strip()
        except EOFError:
            print("No question provided. Pass a question as a command-line argument or run interactively.")
            return
    else:
        print("No interactive input available. Pass a question as a command-line argument.")
        return

    if not question:
        print("No question provided.")
        return

    vectorstore = load_vectorstore(VECTORSTORE_DIR)
    result = answer_question(question, vectorstore)

    print(f"Accepted: {result['accepted']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Vector score: {result['vector_score']:.4f}")
    print(f"BM25 score: {result['bm25_score']:.4f}")
    print(f"BM25 score softmax: {result['bm25_score_softmax']:.4f}")
    print(f"Hybrid score: {result['hybrid_score']:.4f}")
    print(f"Retrieved chunk: {result['retrieved_chunk']}")
    print(f"Metadata: {result.get('metadata')}")

    if result["accepted"]:
        print(f"Answer: {result['answer']}")
    else:
        print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
