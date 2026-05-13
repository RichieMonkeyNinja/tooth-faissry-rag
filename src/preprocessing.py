from pathlib import Path
import shutil

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mineru import MinerULoader


DATA_PATH = Path("data/raw/medmcqa_data.csv")
PDF_DIR = Path("data/raw/pdf")
MARKDOWN_DIR = Path("data/processed/pdf_markdown")
VECTORSTORE_DIR = Path("data/vectorstores/mixed")

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MINERU_MODE = "flash"

SOURCE_TYPE_CSV = "medmcqa"
SOURCE_TYPE_PDF = "pdf"
SOURCE_FILTER_ALL = "all"
SOURCE_FILTER_OPTIONS = (SOURCE_FILTER_ALL, SOURCE_TYPE_PDF, SOURCE_TYPE_CSV)


def load_csv_documents(path: str | Path) -> list[Document]:
    df = pd.read_csv(path)
    documents: list[Document] = []

    for row_index, row in df.iterrows():
        documents.append(
            Document(
                page_content=(
                    f"Question: {row['question']}\n"
                    f"Correct Answer Text: {row['correct_ans']}\n"
                ),
                metadata={
                    "source_type": SOURCE_TYPE_CSV,
                    "source_file": Path(path).name,
                    "row_index": int(row_index),
                },
            )
        )

    return documents


def extract_pdf_markdown(pdf_path: Path) -> str:
    loader = MinerULoader(source=str(pdf_path), mode=MINERU_MODE)
    extracted_documents = loader.load()
    markdown = "\n\n".join(
        document.page_content.strip()
        for document in extracted_documents
        if document.page_content.strip()
    ).strip()

    if not markdown:
        raise ValueError(f"MinerU returned no text for {pdf_path}")

    return markdown


def save_markdown(markdown_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")


def load_pdf_documents(pdf_dir: str | Path, markdown_dir: str | Path) -> list[Document]:
    pdf_dir_path = Path(pdf_dir)
    markdown_dir_path = Path(markdown_dir)

    if not pdf_dir_path.exists():
        return []

    if markdown_dir_path.exists():
        shutil.rmtree(markdown_dir_path)
    markdown_dir_path.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []
    for pdf_path in sorted(pdf_dir_path.glob("*.pdf")):
        markdown = extract_pdf_markdown(pdf_path)
        markdown_path = markdown_dir_path / f"{pdf_path.stem}.md"
        save_markdown(markdown, markdown_path)
        documents.append(
            Document(
                page_content=markdown,
                metadata={
                    "source_type": SOURCE_TYPE_PDF,
                    "source_file": pdf_path.name,
                },
            )
        )

    return documents


def chunk_pdf_documents(pdf_documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunked_documents: list[Document] = []

    for document in pdf_documents:
        chunks = splitter.split_text(document.page_content)
        for chunk_id, chunk_text in enumerate(chunks):
            chunked_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_id": chunk_id,
                    },
                )
            )

    return chunked_documents


def load_mixed_documents() -> list[Document]:
    csv_documents = load_csv_documents(DATA_PATH)
    pdf_documents = load_pdf_documents(PDF_DIR, MARKDOWN_DIR)
    pdf_chunks = chunk_pdf_documents(pdf_documents)
    return [*csv_documents, *pdf_chunks]


def build_vectorstore(documents: list[Document], save_dir: str | Path) -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)

    save_path = Path(save_dir)
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    return vectorstore


def summarize_documents(documents: list[Document]) -> tuple[int, int]:
    csv_count = sum(
        1 for document in documents if document.metadata.get("source_type") == SOURCE_TYPE_CSV
    )
    pdf_chunk_count = sum(
        1 for document in documents if document.metadata.get("source_type") == SOURCE_TYPE_PDF
    )
    return csv_count, pdf_chunk_count


def main() -> None:
    load_dotenv()

    documents = load_mixed_documents()
    vectorstore = build_vectorstore(documents, VECTORSTORE_DIR)
    csv_count, pdf_chunk_count = summarize_documents(documents)

    print(f"Loaded {csv_count} CSV documents from {DATA_PATH}")
    print(f"Extracted and chunked {pdf_chunk_count} PDF chunks from {PDF_DIR}")
    print(f"Saved extracted Markdown files to {MARKDOWN_DIR}")
    print(f"Saved FAISS index to {VECTORSTORE_DIR}")
    print(f"Indexed {vectorstore.index.ntotal} vectors")


if __name__ == "__main__":
    main()
