from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


class UIStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS configs (
                user_id TEXT NOT NULL,
                config_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, config_type)
            );

            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                trace TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comments TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                message TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                assistant_capabilities TEXT NOT NULL,
                sample_queries TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            );
            """
        )
        columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(user_profiles)").fetchall()
        }
        if "sample_queries" not in columns:
            with self._conn:
                self._conn.execute(
                    "ALTER TABLE user_profiles ADD COLUMN sample_queries TEXT NOT NULL DEFAULT '[]'"
                )

        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    def create_user(self, username: str) -> str:
        username = username.strip()
        if not username:
            user_id = str(uuid4())
            logger.info("Generated anonymous user %s", user_id)
            return user_id
        existing = self._conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            existing_id = existing[0]
            if existing_id != username:
                with self._conn:
                    self._conn.execute(
                        "UPDATE users SET id = ? WHERE username = ?",
                        (username, username),
                    )
                    self._conn.execute(
                        "UPDATE configs SET user_id = ? WHERE user_id = ?",
                        (username, existing_id),
                    )
                    self._conn.execute(
                        "UPDATE chats SET user_id = ? WHERE user_id = ?",
                        (username, existing_id),
                    )
                    self._conn.execute(
                        "UPDATE feedback SET user_id = ? WHERE user_id = ?",
                        (username, existing_id),
                    )
                    self._conn.execute(
                        "UPDATE jobs SET user_id = ? WHERE user_id = ?",
                        (username, existing_id),
                    )
                logger.info("Migrated user %s to username-based id=%s", existing_id, username)
            return username
        user_id = username
        with self._conn:
            self._conn.execute(
                "INSERT INTO users (id, username, created_at) VALUES (?, ?, ?)",
                (user_id, username, self._now()),
            )
        logger.info("Created user %s", user_id)
        return user_id

    def upsert_config(self, user_id: str, config_type: str, payload: Dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO configs (user_id, config_type, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, config_type) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (user_id, config_type, json.dumps(payload), self._now()),
            )
        logger.debug("Saved %s config for user %s", config_type, user_id)

    def get_config(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT config_type, payload FROM configs WHERE user_id = ?",
            (user_id,),
        )
        configs: Dict[str, Dict[str, Any]] = {}
        for config_type, payload_json in cursor.fetchall():
            configs[config_type] = json.loads(payload_json)
        return configs

    def create_job(self, user_id: str, query: str) -> str:
        job_id = str(uuid4())
        now = self._now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO jobs (id, user_id, query, status, progress, message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, user_id, query, "queued", 5, "Queued", now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        message: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = self._now()
        result_json = json.dumps(result) if result is not None else None
        with self._conn:
            self._conn.execute(
                """
                UPDATE jobs
                SET status = ?, progress = ?, message = ?, result = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, progress, message, result_json, now, job_id),
            )

    def update_job_todo_snapshot(
        self,
        job_id: str,
        *,
        todos: List[Dict[str, Any]],
        todo_progress: Dict[str, Any],
    ) -> None:
        row = self._conn.execute(
            "SELECT result FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return

        existing_result: Dict[str, Any] = {}
        if row[0]:
            try:
                loaded = json.loads(row[0])
                if isinstance(loaded, dict):
                    existing_result = loaded
            except (TypeError, json.JSONDecodeError):
                existing_result = {}

        existing_result["todos"] = todos
        existing_result["todo_progress"] = todo_progress

        with self._conn:
            self._conn.execute(
                """
                UPDATE jobs
                SET result = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(existing_result), self._now(), job_id),
            )

    def get_job(self, job_id: str) -> Dict[str, Any]:
        cursor = self._conn.execute(
            """
            SELECT id, user_id, query, status, progress, message, result, created_at, updated_at
            FROM jobs WHERE id = ?
            """,
            (job_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise KeyError("Job not found")
        result = json.loads(row[6]) if row[6] else None
        return {
            "id": row[0],
            "user_id": row[1],
            "query": row[2],
            "status": row[3],
            "progress": row[4],
            "message": row[5],
            "result": result,
            "created_at": row[7],
            "updated_at": row[8],
        }

    def has_active_job(self, user_id: str) -> bool:
        cursor = self._conn.execute(
            """
            SELECT 1
            FROM jobs
            WHERE user_id = ? AND status IN ('queued', 'running')
            LIMIT 1
            """,
            (user_id,),
        )
        return cursor.fetchone() is not None

    def save_chat(self, user_id: str, query: str, response: str, trace: Any) -> str:
        chat_id = str(uuid4())
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO chats (id, user_id, query, response, trace, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, query, response, json.dumps(trace), self._now()),
            )
        return chat_id

    def list_chats(self, user_id: str) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            """
            SELECT id, query, response, trace, created_at
            FROM chats WHERE user_id = ? ORDER BY created_at DESC
            """,
            (user_id,),
        )
        chats = []
        for chat_id, query, response, trace_json, created_at in cursor.fetchall():
            chats.append(
                {
                    "id": chat_id,
                    "query": query,
                    "response": response,
                    "trace": json.loads(trace_json),
                    "created_at": created_at,
                }
            )
        return chats

    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        cursor = self._conn.execute(
            """
            SELECT id, user_id, query, response, trace, created_at
            FROM chats WHERE id = ?
            """,
            (chat_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise KeyError("Chat not found")
        return {
            "id": row[0],
            "user_id": row[1],
            "query": row[2],
            "response": row[3],
            "trace": json.loads(row[4]),
            "created_at": row[5],
        }

    def save_feedback(self, chat_id: str, user_id: str, rating: int, comments: str | None) -> str:
        feedback_id = str(uuid4())
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO feedback (id, chat_id, user_id, rating, comments, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (feedback_id, chat_id, user_id, rating, comments or "", self._now()),
            )
        return feedback_id

    def get_assistant_capabilities(self, user_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT assistant_capabilities FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return str(row[0]).strip() or None

    def get_assistant_sample_queries(self, user_id: str) -> List[str]:
        row = self._conn.execute(
            "SELECT sample_queries FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row or row[0] is None:
            return []
        try:
            parsed = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    def upsert_assistant_capabilities(
        self,
        user_id: str,
        capabilities: str,
        sample_queries: Optional[List[str]] = None,
    ) -> None:
        payload = capabilities.strip()
        if not payload:
            return
        normalized_queries = [q.strip() for q in (sample_queries or []) if isinstance(q, str) and q.strip()]
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO user_profiles (user_id, assistant_capabilities, sample_queries, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    assistant_capabilities = excluded.assistant_capabilities,
                    sample_queries = excluded.sample_queries,
                    updated_at = excluded.updated_at
                """,
                (user_id, payload, json.dumps(normalized_queries), self._now()),
            )
