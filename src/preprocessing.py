from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


DATA_PATH = Path("data/raw/medmcqa_data.csv")
VECTORSTORE_DIR = Path("data/vectorstores/medmcqa")


def load_csv_documents(path: str | Path) -> list[Document]:
    df = pd.read_csv(path)
    documents: list[Document] = []

    for row_index, row in df.iterrows():
        documents.append(
            Document(
                page_content=f"Question: {row['question']}\nExplanation: {row['exp']}",
                metadata={
                    "source": str(path),
                    "row_index": int(row_index),
                    "question": row["question"],
                    "doc_type": "medmcqa",
                },
            )
        )

    return documents


def build_vectorstore(documents: list[Document], save_dir: str | Path) -> FAISS:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    return vectorstore


def main() -> None:
    load_dotenv()

    documents = load_csv_documents(DATA_PATH)
    vectorstore = build_vectorstore(documents, VECTORSTORE_DIR)

    print(f"Loaded {len(documents)} documents from {DATA_PATH}")
    print(documents[0])
    print(f"Saved FAISS index to {VECTORSTORE_DIR}")
    print(f"Indexed {vectorstore.index.ntotal} vectors")


if __name__ == "__main__":
    main()
