from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_db_path(config) -> str:
    return str(Path(getattr(config, "APP_DATA_DIR")) / "state.sqlite")


def connect_db(config) -> sqlite3.Connection:
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(config) -> None:
    db_path = Path(get_db_path(config))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with connect_db(config) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                path TEXT NOT NULL,
                source_type TEXT NOT NULL,
                is_temp INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
            );
            """
        )
        # Basic helpful uniqueness: avoid duplicate paths per workspace.
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_workspace_path
            ON sources(workspace_id, path);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_files (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                path TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES sources(id)
            );
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_source_files_source_path
            ON source_files(source_id, path);
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexing_runs (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_id TEXT,
                status TEXT NOT NULL,
                files_scanned INTEGER NOT NULL DEFAULT 0,
                files_added INTEGER NOT NULL DEFAULT 0,
                files_updated INTEGER NOT NULL DEFAULT 0,
                files_unchanged INTEGER NOT NULL DEFAULT 0,
                files_skipped INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_file_id TEXT NOT NULL,
                path TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                corpus TEXT,
                jurisdiction_type TEXT,
                jurisdiction TEXT,
                document_family TEXT,
                title_number TEXT,
                title_name TEXT,
                file_type TEXT NOT NULL,
                extraction_mode TEXT NOT NULL,
                page_count INTEGER,
                parse_status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_source_file_id ON documents(source_file_id);"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_sections (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                section_number TEXT,
                catchline TEXT,
                history TEXT,
                body_text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                section_order INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_sections_document_id ON document_sections(document_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_sections_section_number ON document_sections(section_number);"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_section_id TEXT,
                chunk_index INTEGER NOT NULL,
                chunk_strategy TEXT NOT NULL,
                text TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                jurisdiction TEXT,
                title_number TEXT,
                section_number TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_section_id ON chunks(document_section_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_jurisdiction_title ON chunks(jurisdiction, title_number);"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_map (
                id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                index_type TEXT NOT NULL,
                external_id TEXT,
                collection_key TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_index_map_chunk_id ON index_map(chunk_id);"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS legal_titles (
                id TEXT PRIMARY KEY,
                jurisdiction TEXT NOT NULL,
                title_number TEXT NOT NULL,
                title_name TEXT,
                collection_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_legal_titles_jur_num ON legal_titles(jurisdiction, title_number);"
        )

        # -- Phase 7: orchestrator run tables --

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                user_prompt TEXT NOT NULL,
                run_mode TEXT NOT NULL,
                corpus_family TEXT,
                status TEXT NOT NULL,
                clarification_needed INTEGER NOT NULL DEFAULT 0,
                stop_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_specs (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_specs_run_id ON run_specs(run_id);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_queue (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                queue_order INTEGER NOT NULL,
                corpus_family TEXT NOT NULL,
                jurisdiction_or_issuer TEXT NOT NULL,
                scope_key TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_queue_run_order ON run_queue(run_id, queue_order);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_queue_run_status ON run_queue(run_id, status);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_results (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                queue_item_id TEXT NOT NULL,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_run_results_run_id ON run_results(run_id);")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jurisdiction_ingestion_status (
                jurisdiction TEXT PRIMARY KEY,
                source_root TEXT NOT NULL,
                last_scan_at TEXT,
                documents_count INTEGER NOT NULL DEFAULT 0,
                parsed_count INTEGER NOT NULL DEFAULT 0,
                chunked_count INTEGER NOT NULL DEFAULT 0,
                mirrored_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'discovered',
                last_error TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dev_analytics_events (
                id TEXT PRIMARY KEY,
                session_run_id TEXT,
                event_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_prompt TEXT,
                orchestrator_model TEXT,
                answer_model TEXT,
                embedding_model TEXT,
                corpus_family TEXT,
                jurisdiction TEXT,
                task_count INTEGER,
                evidence_chunks_total INTEGER,
                retrieval_iterations_max INTEGER,
                unavailable_scopes_count INTEGER,
                duration_ms INTEGER,
                status TEXT NOT NULL,
                error_message TEXT,
                extra_json TEXT
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dev_analytics_created ON dev_analytics_events(created_at DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dev_analytics_type ON dev_analytics_events(event_type);"
        )

        conn.commit()
