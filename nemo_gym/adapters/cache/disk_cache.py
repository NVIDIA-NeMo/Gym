# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_RELEVANT_KEYS = ("model", "messages", "tools", "temperature", "max_tokens", "top_p", "seed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_v1 (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

_UPSERT = """
INSERT INTO cache_v1 (key, value, created_at) VALUES (?, ?, ?)
ON CONFLICT(key) DO UPDATE SET value=excluded.value, created_at=excluded.created_at;
"""


class DiskCache:
    def __init__(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        self._db_path = os.path.join(cache_dir, "adapter_cache.db")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        try:
            with self._lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA busy_timeout=5000")
                    conn.execute(_SCHEMA)
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.Error:
            logger.warning("Failed to initialize cache DB at %s", self._db_path, exc_info=True)

    @staticmethod
    def cache_key(body: dict[str, Any], *, session_prefix: str = "") -> str:
        canonical: dict[str, Any] = {}
        for k in _RELEVANT_KEYS:
            if k in body:
                canonical[k] = body[k]
        if "extra_body" in body and isinstance(body["extra_body"], dict):
            canonical["extra_body"] = body["extra_body"]
        raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
        if session_prefix:
            raw = session_prefix + "|" + raw
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_sync(self, key: str) -> dict | None:
        try:
            with self._lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    row = conn.execute("SELECT value FROM cache_v1 WHERE key = ?", (key,)).fetchone()
                finally:
                    conn.close()
            if row is None:
                return None
            return json.loads(row[0])
        except (sqlite3.Error, json.JSONDecodeError):
            logger.warning("Cache get failed for key=%s", key[:16], exc_info=True)
            return None

    def _set_sync(self, key: str, value: dict) -> None:
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            with self._lock:
                conn = sqlite3.connect(self._db_path)
                try:
                    conn.execute(_UPSERT, (key, serialized, time.time()))
                    conn.commit()
                finally:
                    conn.close()
        except sqlite3.Error:
            logger.warning("Cache set failed for key=%s", key[:16], exc_info=True)

    async def get(self, key: str) -> dict | None:
        return await asyncio.to_thread(self._get_sync, key)

    async def set(self, key: str, value: dict) -> None:
        await asyncio.to_thread(self._set_sync, key, value)
