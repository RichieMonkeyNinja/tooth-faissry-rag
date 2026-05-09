from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


VECTORSTORE_DIR = Path("data/vectorstores/medmcqa")
RELEVANCE_THRESHOLD = 0.6
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def load_vectorstore(vectorstore_dir: str | Path) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_nearest_chunk(
    question: str,
    vectorstore: FAISS,
    threshold: float = RELEVANCE_THRESHOLD,
) -> dict[str, Any]:
    results = vectorstore.similarity_search_with_relevance_scores(question, k=1)

    if not results:
        return {
            "answer": None,
            "retrieved_chunk": None,
            "score": 0.0,
            "accepted": False,
            "reason": "Vector store returned no results",
        }

    document, score = results[0]
    accepted = score >= threshold

    return {
        "answer": None,
        "retrieved_chunk": document.page_content,
        "score": float(score),
        "accepted": accepted,
        "reason": None if accepted else "No sufficiently similar context found in vector store",
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


def main() -> None:
    load_dotenv()

    question = input("Enter your question: ").strip()
    if not question:
        print("No question provided.")
        return

    vectorstore = load_vectorstore(VECTORSTORE_DIR)
    result = answer_question(question, vectorstore)

    print(f"Accepted: {result['accepted']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Retrieved chunk: {result['retrieved_chunk']}")
    print(f"Metadata: {result.get('metadata')}")

    if result["accepted"]:
        print(f"Answer: {result['answer']}")
    else:
        print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
