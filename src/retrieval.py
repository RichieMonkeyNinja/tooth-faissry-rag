import argparse
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocessing import (
    EMBEDDING_MODEL,
    SOURCE_FILTER_ALL,
    SOURCE_FILTER_OPTIONS,
    SOURCE_TYPE_CSV,
    VECTORSTORE_DIR,
)


RELEVANCE_THRESHOLD = 0.6
LLM_MODEL = "gpt-4o-mini"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_RETRIEVAL_K = 8
BM25_RETRIEVAL_K = 8
HYBRID_ALPHA = 0.5
NO_GROUNDED_ANSWER_MESSAGE = (
    "No sufficiently similar context was found in the database, so no grounded answer was generated."
)


def preprocess_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def softmax_normalize(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    if not scores:
        return {}

    keys = list(scores.keys())
    values = np.array([scores[key] for key in keys], dtype=float) / temperature
    exp_values = np.exp(values - values.max())
    softmax_values = exp_values / exp_values.sum()
    return {key: float(value) for key, value in zip(keys, softmax_values, strict=True)}


def load_vectorstore(vectorstore_dir: str | Path) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def document_source_matches(document: Document, source_filter: str) -> bool:
    if source_filter == SOURCE_FILTER_ALL:
        return True
    return document.metadata.get("source_type") == source_filter


def build_source_filter(source_filter: str) -> dict[str, str] | None:
    if source_filter == SOURCE_FILTER_ALL:
        return None
    return {"source_type": source_filter}


@lru_cache(maxsize=1)
def load_index_documents() -> tuple[Document, ...]:
    vectorstore = load_vectorstore(VECTORSTORE_DIR)
    docstore_values = tuple(vectorstore.docstore._dict.values())
    return tuple(document for document in docstore_values if isinstance(document, Document))


@lru_cache(maxsize=3)
def load_keyword_retriever(source_filter: str = SOURCE_FILTER_ALL) -> BM25Retriever:
    filtered_documents = [
        document
        for document in load_index_documents()
        if document_source_matches(document, source_filter)
    ]
    retriever = BM25Retriever.from_documents(
        filtered_documents,
        preprocess_func=preprocess_bm25,
    )
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
    source_filter: str = SOURCE_FILTER_ALL,
) -> dict[str, Any]:
    metadata_filter = build_source_filter(source_filter)
    vector_results = vectorstore.similarity_search_with_relevance_scores(
        question,
        k=INITIAL_RETRIEVAL_K,
        fetch_k=INITIAL_RETRIEVAL_K * 5,
        filter=metadata_filter,
    )

    bm25_retriever = load_keyword_retriever(source_filter)
    bm25_results = bm25_retriever.invoke(question)
    processed_query = bm25_retriever.preprocess_func(question)
    raw_bm25_scores = bm25_retriever.vectorizer.get_scores(processed_query)

    if not vector_results and not bm25_results:
        return {
            "answer": None,
            "retrieved_chunk": None,
            "retrieved_contexts": [],
            "score": 0.0,
            "vector_score": 0.0,
            "bm25_score": 0.0,
            "bm25_score_softmax": 0.0,
            "hybrid_score": 0.0,
            "rerank_score": None,
            "accepted": False,
            "reason": f"No results found for source filter '{source_filter}'",
            "metadata": {},
            "source_filter": source_filter,
        }

    vector_scores_by_chunk = {
        document.page_content: float(score) for document, score in vector_results
    }
    bm25_scores_by_chunk = {
        document.page_content: float(score)
        for document, score in zip(bm25_retriever.docs, raw_bm25_scores, strict=True)
    }
    merged_documents: dict[str, Document] = {}

    for document, _ in vector_results:
        chunk = document.page_content
        merged_documents[chunk] = document

    for rank, document in enumerate(bm25_results, start=1):
        chunk = document.page_content
        merged_documents.setdefault(chunk, document)

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
    retrieved_contexts = [document.page_content for document, _ in reranked_results]
    vector_score = vector_scores_by_chunk.get(document.page_content, 0.0)
    bm25_score = bm25_scores_by_chunk.get(document.page_content, 0.0)
    bm25_score_softmax = softmax_bm25_scores.get(document.page_content, 0.0)
    hybrid_score = hybrid_scores_by_chunk.get(document.page_content, 0.0)
    accepted = rerank_score >= 0.0 and (vector_score >= threshold or bm25_score >= 5.0)

    return {
        "answer": None,
        "retrieved_chunk": document.page_content,
        "retrieved_contexts": retrieved_contexts,
        "score": float(hybrid_score),
        "vector_score": float(vector_score),
        "bm25_score": float(bm25_score),
        "bm25_score_softmax": float(bm25_score_softmax),
        "hybrid_score": float(hybrid_score),
        "rerank_score": float(rerank_score),
        "accepted": accepted,
        "reason": None
        if accepted
        else "No sufficiently similar context found in vector or keyword retrieval",
        "metadata": document.metadata,
        "source_filter": source_filter,
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
    source_filter: str = SOURCE_FILTER_ALL,
) -> dict[str, Any]:
    result = retrieve_nearest_chunk(
        question,
        vectorstore,
        threshold=threshold,
        source_filter=source_filter,
    )

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
    parser.add_argument(
        "--source",
        choices=SOURCE_FILTER_OPTIONS,
        default=SOURCE_FILTER_ALL,
        help="Restrict retrieval to a source type.",
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
    result = answer_question(question, vectorstore, source_filter=args.source)

    print(f"Source filter: {result['source_filter']}")
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
