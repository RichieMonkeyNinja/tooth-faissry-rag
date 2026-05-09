# MedMCQA RAG Chatbot

This project builds a small retrieval-augmented generation (RAG) pipeline over the MedMCQA dataset. It exports the dataset to CSV, embeds each question-explanation pair with OpenAI embeddings, stores the vectors in FAISS, and exposes retrieval through both a terminal script and a Streamlit chatbot.

## What This Project Does

- Exports `train` and `validation` from `openlifescienceai/medmcqa` into a local CSV.
- Converts each row into a retrievable document using `question` and `exp`.
- Builds a FAISS vector store with `text-embedding-3-small`.
- Retrieves the nearest document for a user question.
- Rejects retrieval if the relevance score is below a hardcoded threshold of `0.6`.
- Uses `gpt-4o-mini` to rewrite accepted retrieved context into exactly one grounded sentence.

## Tech Stack

- Python `3.12`
- `uv` for environment and dependency management
- Hugging Face `datasets` for dataset loading
- `pandas` for CSV handling
- LangChain for orchestration
- OpenAI embeddings: `text-embedding-3-small`
- OpenAI chat model: `gpt-4o-mini`
- FAISS for vector search
- Streamlit for the chatbot interface

## Project Structure

```text
grp-asgmt/
|-- data/
|   |-- raw/
|   |   `-- medmcqa_data.csv
|   `-- vectorstores/
|       `-- medmcqa/
|           |-- index.faiss
|           `-- index.pkl
|-- src/
|   |-- datasets.py
|   |-- preprocessing.py
|   |-- retrieval.py
|   `-- interface.py
|-- AGENTS.md
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Replication Guide

### 1. Prerequisites

- Python `3.12.x`
- `uv` installed
- An OpenAI API key

### 2. Clone the repository

```powershell
git clone <your-repo-url>
cd <your-repo-url>
```

### 3. Sync dependencies

```powershell
uv sync
```

### 4. Create a `.env` file

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Export the MedMCQA dataset to CSV

```powershell
python src/datasets.py
```

This writes:

```text
data/raw/medmcqa_data.csv
```

The exported CSV keeps only:

- `question`
- `exp`

### 6. Build the FAISS vector store

```powershell
python src/preprocessing.py
```

This writes:

```text
data/vectorstores/medmcqa/index.faiss
data/vectorstores/medmcqa/index.pkl
```

### 7. Test retrieval from the terminal

```powershell
python src/retrieval.py
```

You will be prompted for a question. The script then:

- loads the saved FAISS index
- retrieves the nearest document
- checks whether the relevance score is at least `0.6`
- either rejects retrieval or generates a one-sentence grounded answer

### 8. Launch the Streamlit chatbot

```powershell
streamlit run src/interface.py
```

The UI:

- keeps visible chat history
- treats each user question independently
- caches the vector store for faster reuse
- shows the latest retrieval diagnostics in a separate panel

## Example Behavior

### Accepted retrieval

If the top match is sufficiently similar:

- the system accepts the retrieved chunk
- `gpt-4o-mini` rewrites the retrieved explanation into one coherent sentence

### Rejected retrieval

If the top match is below the `0.6` threshold:

- no answer is generated from the LLM
- the system explicitly reports that no sufficiently similar context was found

## Codebase Breakdown

### 1. Data export

File: [src/datasets.py](C:/Users/teohr/Downloads/UM/NLP/grp-asgmt/src/datasets.py)

Responsibilities:

- loads `openlifescienceai/medmcqa`
- concatenates `train` and `validation`
- keeps only `question` and `exp`
- saves the result as `data/raw/medmcqa_data.csv`

### 2. Indexing

File: [src/preprocessing.py](C:/Users/teohr/Downloads/UM/NLP/grp-asgmt/src/preprocessing.py)

Responsibilities:

- reads the exported CSV
- converts each row into a LangChain `Document`
- stores `question + exp` together as a single retrieval unit
- builds a FAISS index using `text-embedding-3-small`
- saves the vector store to disk

### 3. Retrieval and generation

File: [src/retrieval.py](C:/Users/teohr/Downloads/UM/NLP/grp-asgmt/src/retrieval.py)

Responsibilities:

- loads the local FAISS vector store
- embeds the user query
- retrieves the nearest document with a relevance score
- rejects matches below the hardcoded threshold of `0.6`
- uses `gpt-4o-mini` only when retrieval is accepted
- returns the answer, score, retrieved chunk, and metadata

### 4. User interface

File: [src/interface.py](C:/Users/teohr/Downloads/UM/NLP/grp-asgmt/src/interface.py)

Responsibilities:

- provides a Streamlit chatbot
- depends on the retrieval logic in `retrieval.py`
- shows the conversation in the main panel
- shows the latest retrieval diagnostics in a side panel

## Architecture Overview

The implemented pipeline is:

```text
MedMCQA dataset
    -> CSV export
    -> row-level documents
    -> OpenAI embeddings
    -> FAISS vector store
    -> nearest-neighbor retrieval
    -> threshold check
    -> grounded one-sentence answer
    -> terminal or Streamlit interface
```

### Retrieval design decisions

- Retrieval unit is one full MedMCQA row, not a chunked fragment.
- Retrieval uses top `k=1`.
- Generation is grounded only in the retrieved context.
- If retrieval confidence is too low, the system rejects instead of hallucinating.

## Known Limitations

- Retrieval is limited to a single nearest document (`k=1`).
- The relevance threshold is hardcoded to `0.6`.
- Answers are restricted to one sentence only.
- The system currently works only on the exported MedMCQA CSV.
- PDF ingestion is planned but not implemented.
- The current project structure still relies on script-level imports rather than a fully packaged module layout.

## Future Work

- Add PDF ingestion for medical reference documents.
- Add chunking for long-form sources such as PDFs.
- Compare retrieval with `k > 1`.
- Tune the rejection threshold on a validation set instead of using a fixed provisional value.
- Package the `src` directory more cleanly to avoid script-relative import fragility.
- Change to locally hosted models, including `gemma-4` and `multilingual-embeddings`

## Notes

- The vector store loader uses `allow_dangerous_deserialization=True` when restoring `index.pkl`.
- This is acceptable for locally generated FAISS artifacts you trust.
- Do not use that setting to load arbitrary vector store files from untrusted sources.
