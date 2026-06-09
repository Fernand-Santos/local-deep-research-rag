# Internal Setup (Private Repo)

## Local setup

1. Clone the repo.
2. Create a virtual environment (Conda or venv).
3. Install dependencies: `pip install -r requirements-deep-research.txt`
4. Copy `.env.example` to `.env` and set values for your machine.
5. Ensure Ollama is running locally (`ollama serve`).

## Do not commit

- Corpora folders (`Laws_Regulations/`, `Agencies/`, etc.)
- SQLite / database files
- Chroma persistence / vectorstore artifacts
- Logs, traces, reports, previews
- `.local_deep_research/` and other generated state
- `.env` (contains local paths)

These are machine-specific and excluded via `.gitignore`.
