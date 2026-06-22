# Local Deep Research RAG

Private local legal research system. Ingests PDFs from hierarchical folders, chunks and embeds content, mirrors to Chroma for retrieval, and answers questions using Ollama models.

## What it does

- **Ask mode**: Natural-language questions → task decomposition → scoped retrieval → answer synthesis with citations. Supports orchestrator, retrieval/answer, and embedding model selection. Thinking logs and cited page previews when available.
- **Knowledge Base**: States root sync, per-jurisdiction ingestion status, indexed coverage. "Sync States Root" runs the full pipeline for all discovered state folders.
- **Advanced**: Manual ingestion (workspaces, sources, discovery, parsing, chunking, Chroma mirror), single-scope retrieval test, run compiler/orchestrator, execution, cumulative output, reports.

## Demo 1
https://github.com/user-attachments/assets/18dcb451-04a3-4fb0-bf36-492b307f832b

## Demo 2
https://github.com/user-attachments/assets/3194ba36-7500-4b75-8b53-a3c47390ccef

## Architecture

- **SQL is canonical**: Chunks, embeddings, documents, legal titles, run state live in SQLite. Source of truth for what was ingested and indexed.
- **Chroma is a mirror**: Catalog and title chunk collections are mirrored from SQL for fast semantic retrieval. Chroma is not the source of truth.
- **Ollama**: Embeddings and chat via local Ollama. No external API keys.

## UI structure

| Tab | Purpose |
|-----|---------|
| Ask | Primary query flow. Orchestrator parses questions into tasks; answer model synthesizes over evidence. |
| Knowledge Base | States root path, sync button, per-state readiness, indexed coverage. |
| Advanced | Operator tools: workspaces, sources, parsing, chunking, Chroma mirror, retrieval test, run compiler, execution, reports. |

## Setup (new machine)

1. Clone the repo.
2. Create environment: `conda create -n <env> python=3.10` or `python -m venv .venv`.
3. Install: pip install -r requirements.txt
4. Copy `.env.example` to `.env` and set `OLLAMA_BASE_URL`, `APP_DATA_DIR`, `STATES_ROOT`, etc.
5. Run Ollama locally: `ollama serve`. Pull models: `ollama pull nomic-embed-text`, `ollama pull qwen2.5` (or similar).
6. Place corpus PDFs under `Laws_Regulations/States/<StateName>/` (or set `STATES_ROOT` to your path).

## Run

```bash
streamlit run main.py
# or
python -m app
```

## Environment

- Python 3.10+
- Ollama running locally
- `APP_DATA_DIR`: where SQLite, Chroma, logs, traces live (default: `~/.local_deep_research`)
- `STATES_ROOT`: parent folder for state subfolders (default: `<project>/Laws_Regulations/States`)

## Corpus layout

```
Laws_Regulations/
  States/
    Alabama/
      Alabama_Title_1_General_Provisions.pdf
      Alabama_Title_2_Agriculture.pdf
      ...
    Alaska/
      ...
```

Corpora and generated state (SQLite, Chroma, logs, traces, reports) are **local and not committed**. See `.gitignore`.

## Current status

- Ask flow with task decomposition, retrieval, answer synthesis, thinking logs, page previews.
- States root auto-discovery and sync pipeline.
- Per-jurisdiction ingestion status; Ask mode gates on `ready` status only.
- Advanced manual tools intact.

## Next priorities

- Retrieval evaluation harness
- Further ingestion/readiness refinements
