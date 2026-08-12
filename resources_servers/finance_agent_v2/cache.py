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
"""Disk-backed cache for the Finance Agent v2 tools.

``ToolCache`` is a dependency-free key/value store on disk. It is deliberately dumb:
atomic reads and writes, no knowledge of pricing/SEC semantics — key derivation and
merge logic live in ``cached_tools.py``.

It is either on (read and write) or off (tools run live). It stores the raw upstream
response and lets the untouched upstream serializer render it, so a hit is
byte-identical to a live call.

Cache namespaces live as subdirectories under the root:
  - ``pricing/``        per-(endpoint, ticker) master records (Tiingo)
  - ``edgar_search/``   raw sec-api.io search result lists
  - ``sec_filings/``    parsed sec.gov filing documents
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class ToolCache:
    """Namespaced, atomic disk cache with an on/off switch.

    ``cache_dir`` roots the cache (relative paths resolve from the CWD); when unset
    and enabled, it defaults under ``~/.cache/nemo_gym/finance_agent_v2``.
    """

    #: A single path component: no separator, and a leading alphanumeric so the
    #: name can never be ``..`` or a dotfile.
    _SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

    def __init__(
        self,
        cache_dir: Optional[str | os.PathLike[str]],
        use_cache: bool = True,
    ) -> None:
        self.root: Optional[Path] = None
        if not use_cache:
            return

        if cache_dir:
            root = Path(cache_dir)
            if not root.is_absolute():
                root = Path.cwd() / root
        else:
            root = Path.home() / ".cache" / "nemo_gym" / "finance_agent_v2"
            logger.warning(
                "use_cache is on but cache_dir is not set; defaulting to %s. "
                "This path is ephemeral in containers and not shared across jobs. "
                "Set cache_dir to a shared absolute path for production/multi-seed use.",
                root,
            )
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._root_resolved = root.resolve()

    @property
    def enabled(self) -> bool:
        return self.root is not None

    # -- paths ----------------------------------------------------------------
    def path(self, *parts: str) -> Path:
        """Resolve a path under the cache root. Raises if the cache is disabled.

        Rejects any result that lands outside the root. Parts are derived from
        tool arguments, so this is the one place that guarantee can be enforced
        for every namespace.
        """
        if self.root is None:
            raise RuntimeError("ToolCache is disabled; no paths are available.")
        candidate = self.root.joinpath(*parts)
        resolved = candidate.resolve()
        if resolved != self._root_resolved and self._root_resolved not in resolved.parents:
            raise ValueError(f"cache path escapes the cache root: {os.path.join(*parts)}")
        return candidate

    @staticmethod
    def hash_key(payload: Any) -> str:
        """Stable sha256 of a JSON-serializable payload (sorted keys)."""
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @classmethod
    def safe_name(cls, value: str) -> str:
        """Return *value* if it is usable as one path component, else its digest.

        A value that names a cache file can come straight from a model tool call,
        so it must not be able to steer the path. Unusable values are replaced by
        a digest rather than stripped of offending characters, so two different
        inputs cannot be folded onto the same file.
        """
        return value if cls._SAFE_NAME_RE.match(value) else cls.hash_key(value)

    # -- atomic IO ------------------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=encoding) as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def read_text(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    def write_text(self, path: Path, text: str) -> None:
        self._atomic_write(path, text)

    def read_json(self, path: Path) -> Optional[Any]:
        text = self.read_text(path)
        if text is None:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def write_json(self, path: Path, obj: Any) -> None:
        # indent=2 keeps cache files human-readable for debugging.
        self._atomic_write(path, json.dumps(obj, indent=2, default=str))

    def read_jsonl(self, path: Path) -> Optional[list[Any]]:
        text = self.read_text(path)
        if text is None:
            return None
        records: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return None
        return records

    def write_jsonl(self, path: Path, records: list[Any]) -> None:
        body = "\n".join(json.dumps(r, default=str) for r in records)
        self._atomic_write(path, body + ("\n" if body else ""))
