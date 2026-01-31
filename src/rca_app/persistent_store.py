from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List

from langgraph.store.base import BaseStore, IndexConfig, Op, PutOp, Result
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


class SQLiteBackedStore(BaseStore):
    def __init__(self, path: Path, *, index: IndexConfig | None = None) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._store = InMemoryStore(index=index)
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        with self._lock:
            cursor = self._conn.execute("SELECT namespace, key, value FROM memory_store")
            rows = cursor.fetchall()
        if not rows:
            logger.info("No persisted memory found at %s", self._path)
            return
        ops: List[PutOp] = []
        for namespace_json, key, value_json in rows:
            namespace = tuple(json.loads(namespace_json))
            value = json.loads(value_json)
            ops.append(PutOp(namespace, key, value))
        self._store.batch(ops)
        logger.info("Loaded %s persisted memory records from %s", len(rows), self._path)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops_list = list(ops)
        results = self._store.batch(ops_list)
        put_ops = [op for op in ops_list if isinstance(op, PutOp)]
        if put_ops:
            with self._lock:
                with self._conn:
                    for op in put_ops:
                        namespace_json = json.dumps(op.namespace)
                        if op.value is None:
                            self._conn.execute(
                                "DELETE FROM memory_store WHERE namespace = ? AND key = ?",
                                (namespace_json, op.key),
                            )
                        else:
                            self._conn.execute(
                                "INSERT OR REPLACE INTO memory_store (namespace, key, value) VALUES (?, ?, ?)",
                                (namespace_json, op.key, json.dumps(op.value)),
                            )
            logger.debug("Persisted %s memory operations", len(put_ops))
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch, list(ops))
