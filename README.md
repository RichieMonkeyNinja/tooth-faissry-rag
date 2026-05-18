# MedMCQA RAG Chatbot

This project builds a small retrieval-augmented generation (RAG) pipeline over the MedMCQA dataset and a set of dental PDF reference documents. It exports the dataset to CSV, extracts PDFs to Markdown, stores the mixed corpus in FAISS, and exposes retrieval through both a terminal script and a Streamlit chatbot.

## What This Project Does

- Exports `train` and `validation` from `openlifescienceai/medmcqa` into a local CSV.
- Converts each row into a retrievable document using `question` and `exp`.
- Extracts the PDFs in `data/raw/pdf` with MinerU, saves Markdown intermediates, and chunks the extracted text.
- Builds a mixed FAISS vector store with `text-embedding-3-small`.
- Retrieves candidate documents with FAISS and BM25, fuses them with RRF, and reranks the top candidates with a cross-encoder.
- Rejects retrieval if the cross-encoder rerank score is below the configured acceptance threshold.
- Uses `gpt-4o-mini` to rewrite accepted retrieved context into exactly one grounded sentence.
- Provides a RAGAS evaluation script for `Context Precision`, `Context Recall`, `Faithfulness`, and `Answer Relevancy`.

## Tech Stack

- Python `3.12`
- `uv` for environment and dependency management
- Hugging Face `datasets` for dataset loading
- `pandas` for CSV handling
- LangChain for orchestration
- OpenAI embeddings: `text-embedding-3-small`
- OpenAI chat model: `gpt-4o-mini`
- FAISS for vector search
- RAGAS for evaluation
- Streamlit for the chatbot interface

## Project Structure

```text
grp-asgmt/
|-- data/
|   |-- evals/
|   |   |-- qa_eval.csv
|   |   `-- results/
|   |-- processed/
|   |   `-- pdf_markdown/
|   |-- raw/
|   |   `-- medmcqa_data.csv
|   `-- vectorstores/
|       `-- mixed/
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
data/vectorstores/mixed/index.faiss
data/vectorstores/mixed/index.pkl
```

### 7. Test retrieval from the terminal

```powershell
python src/retrieval.py --source all
```

You will be prompted for a question. The script then:

- loads the saved FAISS index
- retrieves the top 15 FAISS and top 15 BM25 candidates
- supports `--source all`, `--source pdf`, or `--source medmcqa`
- fuses candidates with RRF, keeps the top 20, and reranks them with a cross-encoder
- checks whether the rerank score is at least the configured acceptance threshold
- either rejects retrieval or generates a one-sentence grounded answer

### 8. Run RAGAS evaluation

```powershell
python src/evaluate_ragas.py
```

The evaluator expects `data/evals/qa_eval.csv` with:

- `question`
- `reference`
- `reference_contexts` as a JSON array string
- optional `source_filter`

It uses `gpt-4o` as the RAGAS judge model and writes per-sample traces and metric summaries to `data/evals/results/` in a timestamped folder.

### 9. Tune the rerank acceptance threshold

```powershell
python src/tune_rerank_threshold.py
```

The tuner expects `data/evals/rerank_threshold_eval.csv` with:

- `question`
- `expected_relevant`
- optional `notes`

It runs live retrieval against all sources, saves per-question rerank scores, sweeps candidate thresholds, and writes artifacts to `data/evals/results/`.

### 10. Launch the Streamlit chatbot

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

If the top reranked match is below the rerank acceptance threshold:

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
- retrieves the top 15 vector and top 15 BM25 candidates
- uses RRF to keep the top 20 fused candidates for cross-encoder reranking
- rejects matches below the rerank acceptance threshold
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
    -> FAISS top 15 + BM25 top 15 retrieval
    -> RRF top 20 candidate fusion
    -> cross-encoder reranking
    -> rerank acceptance check
    -> grounded one-sentence answer
    -> terminal or Streamlit interface
```

### Retrieval design decisions

- Retrieval unit is one full MedMCQA row, not a chunked fragment.
- Retrieval combines vector and keyword candidates before reranking.
- Generation is grounded only in the retrieved context.
- If retrieval confidence is too low, the system rejects instead of hallucinating.

## Known Limitations

- Reranker acceptance still uses a fixed provisional threshold.
- Answers are restricted to one sentence only.
- The system now works over both the exported MedMCQA CSV and the PDFs in `data/raw/pdf`.
- The evaluation framework still needs curated rows in `data/evals/qa_eval.csv` before it can produce meaningful RAGAS scores.
- The current project structure still relies on script-level imports rather than a fully packaged module layout.

## Future Work

- Add PDF ingestion for medical reference documents.
- Add chunking for long-form sources such as PDFs.
- Tune the rerank acceptance threshold on a validation set instead of using a fixed provisional value.
- Package the `src` directory more cleanly to avoid script-relative import fragility.
- Change to locally hosted models, including `gemma-4` and `multilingual-embeddings`

## Notes

- The vector store loader uses `allow_dangerous_deserialization=True` when restoring `index.pkl`.
- This is acceptable for locally generated FAISS artifacts you trust.
- Do not use that setting to load arbitrary vector store files from untrusted sources.
