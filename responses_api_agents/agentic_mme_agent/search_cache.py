# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared, persistent cache of retrieval results: flat files + a read-only FTS5 fuzzy index.

Layout (no external tooling; `python -m responses_api_agents.agentic_mme_agent.search_cache <root>` rebuilds the index):

    <root>/search/<ab>/<cd>/<sha256(provider|gl|hl|norm query)>.json  {"query","norm","provider","gl","hl","results", ...}
    <root>/lens/<ab>/<cd>/<sha256(provider|pixel key)>.json            {"image_key","provider","results", ...}
    <root>/browse/<ab>/<cd>/<sha256(provider|norm url)>.json           {"url","norm","provider","content", ...}
    <root>/index/master.db                              FTS5 over every cached query norm (read-only here)
    <root>/index/search_norms_<host>_<pid>.txt          this process's new norms (merged by rebuild_index)

Lookups: search = exact norm, then fuzzy (token_sort_ratio >= threshold over FTS5 candidates, see
sqlite_fuzzy_index.py) unless the candidate's numbers differ from the query's (2010 vs 2020);
lens = sha256 of the decoded RGB pixels + size; browse = normalised URL. Every key includes the
provider (and, for search, gl/hl), so a run never gets another provider's or another locale's
results under its own provider's name. Entries written before keys carried the provider (keyed
by the bare norm / pixels / URL) are still served when their stored provider (and gl/hl; missing
counts as us/en) matches. Files are content-addressed
and written atomically (temp + os.replace), so many processes on many nodes can share one root.
Only provider responses are stored, never model-written text.
"""

import contextlib
import glob
import hashlib
import json
import os
import re
import socket
import threading
from functools import lru_cache
from typing import Any

from PIL import Image

from responses_api_agents.agentic_mme_agent.sqlite_fuzzy_index import SqliteFuzzyIndex, build_master_db, master_db_path


_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_query(query: str) -> str:
    """Lowercase, strip one layer of matching surrounding quotes, collapse whitespace."""
    q = (query or "").strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
        q = q[1:-1]
    return _WS_RE.sub(" ", q).strip().lower()


def normalize_url(url: str) -> str:
    u = (url or "").strip().split("#", 1)[0]
    return u[:-1] if u.endswith("/") else u


def pixel_key(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    return hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()


# Shared caches are written by processes with different uids and umasks (e.g. root in a container,
# squashed to nobody on Lustre); entries must stay readable and directories writable for the others.
_DIR_MODE = 0o2775
_FILE_MODE = 0o664


def _makedirs_shared(path: str) -> None:
    missing = []
    while path and not os.path.isdir(path):
        missing.append(path)
        path = os.path.dirname(path)
    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            continue
        os.chmod(directory, _DIR_MODE)


# Locale a legacy search entry is taken to have when it did not record one (the tool defaults).
_SCOPE_DEFAULTS = {"gl": "us", "hl": "en"}


def _scoped_key(key: str, scope: dict[str, str]) -> str:
    return "|".join([*(scope[k] for k in sorted(scope)), key])


def _numbers(norm: str) -> set[str]:
    return set(_NUM_RE.findall(norm))


class SearchCache:
    def __init__(self, root: str, fuzzy_threshold: float = 0.0, write: bool = True) -> None:
        self.root = root
        self.write = write
        # With no master.db yet the index still matches this process's own new entries (local tier).
        self.fuzzy = SqliteFuzzyIndex(root, threshold=fuzzy_threshold) if fuzzy_threshold else None
        self._shard = os.path.join(root, "index", f"search_norms_{socket.gethostname()}_{os.getpid()}.txt")
        self._lock = threading.Lock()

    # ---- flat files ---------------------------------------------------------------------------
    def _path(self, kind: str, key: str) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, kind, h[:2], h[2:4], f"{h}.json")

    def _read(self, kind: str, key: str) -> dict[str, Any] | None:
        try:
            with open(self._path(kind, key), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _read_scoped(self, kind: str, key: str, scope: dict[str, str]) -> dict[str, Any] | None:
        """The entry for key under this provider (and locale); falls back to an entry written before keys
        carried the scope, only when its stored scope matches."""
        entry = self._read(kind, _scoped_key(key, scope))
        if entry is not None:
            return entry
        legacy = self._read(kind, key)
        if legacy and all(legacy.get(k, _SCOPE_DEFAULTS.get(k)) == v for k, v in scope.items()):
            return legacy
        return None

    def _write(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        if not self.write:
            return
        path = self._path(kind, key)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            _makedirs_shared(os.path.dirname(path))
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.chmod(tmp, _FILE_MODE)
            os.replace(tmp, path)
        except OSError:
            # Best effort: an unwritable shard of a shared cache must not turn a paid provider
            # answer into a tool error.
            with contextlib.suppress(OSError):
                os.remove(tmp)

    # ---- search -------------------------------------------------------------------------------
    def get_search(
        self, query: str, provider: str, gl: str = "us", hl: str = "en"
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        """(Serper-shaped results, "exact" | "fuzzy" | "miss", the cached query a fuzzy hit answers).

        Only entries from this provider and locale count. A fuzzy hit answers a different, similar
        query; callers must show that query to the model.
        """
        scope = {"provider": provider, "gl": gl, "hl": hl}
        norm = normalize_query(query)
        entry = self._read_scoped("search", norm, scope)
        if entry and isinstance(entry.get("results"), dict):
            return entry["results"], "exact", None
        if self.fuzzy is not None and norm:
            match = self.fuzzy.lookup(norm)
            # A near-identical query about another year / quantity is a different question.
            if match and _numbers(match) == _numbers(norm):
                entry = self._read_scoped("search", match, scope)
                if entry and isinstance(entry.get("results"), dict):
                    return entry["results"], "fuzzy", entry.get("query") or match
        return None, "miss", None

    def put_search(self, query: str, results: dict[str, Any], provider: str, gl: str, hl: str) -> None:
        norm = normalize_query(query)
        if not norm or not self.write:
            return
        self._write(
            "search",
            _scoped_key(norm, {"provider": provider, "gl": gl, "hl": hl}),
            {
                "query": query,
                "norm": norm,
                "provider": provider,
                "gl": gl,
                "hl": hl,
                "results": results,
                "origin": "raw",
                "source": "agentic_mme_agent",
            },
        )
        if self.fuzzy is not None and self.fuzzy.add(norm):
            with self._lock, contextlib.suppress(OSError):  # durable record for the next index build
                _makedirs_shared(os.path.dirname(self._shard))
                with open(self._shard, "a", encoding="utf-8") as f:
                    f.write(norm + "\n")
                os.chmod(self._shard, _FILE_MODE)

    # ---- lens / browse ------------------------------------------------------------------------
    def get_lens(self, image: Image.Image, provider: str) -> dict[str, Any] | None:
        entry = self._read_scoped("lens", pixel_key(image), {"provider": provider})
        return entry["results"] if entry and isinstance(entry.get("results"), dict) else None

    def put_lens(self, image: Image.Image, results: dict[str, Any], provider: str) -> None:
        key = pixel_key(image)
        self._write(
            "lens",
            _scoped_key(key, {"provider": provider}),
            {
                "image_key": key,
                "provider": provider,
                "results": results,
                "origin": "raw",
                "source": "agentic_mme_agent",
            },
        )

    def get_browse(self, url: str, provider: str) -> str | None:
        entry = self._read_scoped("browse", normalize_url(url), {"provider": provider})
        return entry.get("content") if entry and isinstance(entry.get("content"), str) else None

    def put_browse(self, url: str, content: str, provider: str = "jina") -> None:
        norm = normalize_url(url)
        self._write(
            "browse",
            _scoped_key(norm, {"provider": provider}),
            {
                "url": url,
                "norm": norm,
                "provider": provider,
                "content": content,
                "origin": "raw",
                "source": "agentic_mme_agent",
            },
        )


@lru_cache(maxsize=None)
def shared_cache(root: str, fuzzy_threshold: float, write: bool) -> SearchCache:
    """One cache per (root, settings) per process: the fuzzy index's local tier is shared by all rollouts."""
    return SearchCache(root, fuzzy_threshold, write)


def rebuild_index(root: str) -> int:
    """Rebuild <root>/index/master.db from every cached search entry plus the per-process shards.

    Run between evaluations (readers open master.db read-only and immutable); returns the norm count.
    """
    norms = set()
    for path in glob.glob(os.path.join(root, "search", "*", "*", "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                norm = json.load(f).get("norm")
        except (OSError, ValueError):
            continue
        if norm:
            norms.add(norm)
    for path in glob.glob(os.path.join(root, "index", "search_norms_*.txt")):
        with open(path, encoding="utf-8") as f:
            norms.update(line.strip() for line in f if line.strip())
    target = master_db_path(root)
    _makedirs_shared(os.path.dirname(target))
    tmp = f"{target}.{os.getpid()}.new"
    build_master_db(tmp, sorted(norms))
    os.chmod(tmp, _FILE_MODE)
    os.replace(tmp, target)
    return len(norms)


if __name__ == "__main__":
    import sys

    print(f"master.db: {rebuild_index(sys.argv[1])} query norms")
