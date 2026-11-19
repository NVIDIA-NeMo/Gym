# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flat-file cache for Exa MCP responses, shared by every shard and every sandbox.

LAYOUT -- deliberately the legacy one (the previous stack's
responses_api_agents/swe_agents/search_cache.py), not a new design:

    <root>/search/<ab>/<cd>/<sha256(norm_query)>.json  -> {query, norm, provider, payload}
    <root>/browse/<ab>/<cd>/<sha256(norm_url)>.json    -> {url,   norm, provider, payload}

One store with two sections, content-addressed, two-level hex sharding. Keys are GLOBAL rather
than partitioned per question: measured over a completed run, cross-question reuse is
essentially nil at both an exact match and a high Jaccard threshold, so partitioning would buy
nothing while fencing off reuse if the question set ever changes.

WHY FUZZY MATCHING IS THE DEFAULT. Exact-key reuse across runs
of an identical question set is negligible, while near-duplicate reuse is an order of magnitude
larger, and most questions issue several near-duplicate queries. An exact-only cache would look
broken as a way to save cost. Evals weigh it the other way: a fuzzy hit serves results stored
for a different query, so a run that takes them is not the same experiment as an uncached one.
exact_only=True serves exact and legacy_root hits only and builds no index. An eval that sets a
legacy_root should also set use_tavily=False: a Tavily entry is served translated into Exa's text
(render_tavily), which is not what Exa returned.

WHAT IS STORED: the text the transport returned, BEFORE the contamination guard and the TDM
domain filter run. Those are applied by the plugin on the way out, on hits and misses alike. Cache
post-filter text and a later run with a different exclude-list silently inherits the old
filtering; legacy made the same choice for the same reason.

PROVIDER TAGGING: entries carry the provider that produced them, so an Exa entry is never served
to a Tavily caller or vice versa. We only run Exa today, but a mixed store that forgets which is
which is silently wrong rather than loudly broken.

INDEX: fuzzy matching scans an in-memory index of every stored norm, and building it from the
store means reading every entry, which is slow for a large store. So nothing waits for
it. load_in_background() builds it on a daemon thread, which the server starts at startup; exact
and legacy_root lookups and put() never wait for the build, and until the index is ready a lookup
that would need fuzzy matching is a miss (counted as fuzzy_skipped). With index_path set the index
is also kept in one JSON file, the index file, which records the root it indexes; a file saved for
another root is ignored. A usable index file makes the index ready as soon as it is read. The walk
that follows still lists the whole store: it reads only the entries the file does not list, drops
the keys whose entries are gone, and rewrites the file if it did either. Until the walk has dropped
them, a fuzzy match on an entry deleted since the file was saved is a miss, as for a torn entry.

CONCURRENCY: writes are tmp-file + os.replace, which is atomic within a directory on Lustre, so a
reader never sees a half-written entry and concurrent writers of the same key are last-wins on
identical content. No locks, no SQLite, nothing that takes a cross-node lease -- the failure mode
of a lock on this filesystem is worse than the duplicate work it would prevent. Within the process
there are two locks: _lock guards each change to the index, and _load_lock lets only one load()
run. Lookups and put() never take _load_lock, so they never wait for the build.
"""

import contextlib
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Every entry is named sha256(norm), and a key read from the index file is joined into a path, so
# an index key must look like one: anything else could point a fuzzy hit outside the store.
_KEY_RE = re.compile(r"[0-9a-f]{64}")
# Bump on any change to the index file's layout: a file with another version is ignored and the
# index is rebuilt from the store.
_INDEX_FILE_VERSION = 1


def normalize_query(text: str) -> str:
    """Lowercase, strip one layer of surrounding quotes, collapse whitespace.

    Byte-compatible with the legacy normaliser (the previous stack's
    swe_agents/search_cache.py) ON PURPOSE. The key is sha256(norm), so a normaliser that differs
    by even a trailing character addresses a different file and every entry already on disk
    becomes invisible. Do not "improve" this without a new root.
    """
    q = (text or "").strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
        q = q[1:-1]
    return " ".join(q.split()).strip().lower()


def normalize_url(url: str) -> str:
    """Drop the #fragment and one trailing slash (legacy search_cache.py:122-128)."""
    u = (url or "").strip().split("#", 1)[0]
    return u[:-1] if u.endswith("/") else u


def normalize(text: str, kind: str = "search") -> str:
    return normalize_query(text) if kind == "search" else normalize_url(text)


def render_tavily(results: list) -> str:
    """Render Tavily's structured results into the Title:/URL:/... text Exa returns.

    Needed because the legacy store is ~90% Tavily by volume and those entries hold a list of
    dicts, while our plugin parses the rendered Exa SERP. Field order matches the fork's renderer
    so the model sees the shape it expects.

    This is a TRANSLATION, not a recording: the content came from Tavily, not Exa. Fine for cost
    (the model still gets real search results) but it breaks strict comparability with an
    Exa-only baseline, which is why it sits behind its own flag.
    """
    blocks = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        entry = "Title: " + str(r.get("title") or "") + "\nURL: " + str(r.get("url") or "")
        entry += "\nPublished: " + str(r.get("published_date") or "N/A")
        body = r.get("raw_content") or r.get("content") or ""
        if body:
            entry += "\nText: " + str(body)
        blocks.append(entry)
    return "\n\n".join(blocks)


def to_sse(text: str) -> str:
    """Wrap rendered SERP text in the one-frame SSE envelope the plugin parses.

    parseMcpBody reads `data: ` frames and takes result.content[0].text, so a legacy entry -- which
    stores only the rendered text -- has to be re-enveloped before it can be served as a cache hit.
    """
    import json as _json

    return "event: message\ndata: " + _json.dumps({"result": {"content": [{"type": "text", "text": text}]}}) + "\n\n"


def _tokens(norm: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(norm))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ExaCache:
    """Read/write the flat-file store, with an in-process index for near-duplicate lookup.

    The index is filled on a background thread, from the index file when there is one and from a
    walk of the store, and every put() adds to it. It holds (norm, tokens, key) per entry, so its
    memory grows with the store. Payloads are never held in memory.

    exact_only: serve exact and legacy_root hits only; no index is built, read or saved.
    read_only: write nothing; put() stores nothing and the index file is never saved.
    index_path: the index file, one per root; empty keeps the index in memory only.
    """

    def __init__(
        self,
        root: str,
        fuzzy_threshold: float = 0.8,
        provider: str = "exa",
        legacy_root: str = "",
        use_tavily: bool = True,
        *,
        exact_only: bool = False,
        read_only: bool = False,
        index_path: str = "",
    ) -> None:
        self.root = Path(root)
        # A READ-ONLY second store, searched by exact key only. The legacy cache holds ~2M
        # entries; walking it to build a fuzzy index would hang server startup while rollouts
        # wait on seed_session, and an exact lookup is a direct path from the hash, so it costs
        # one stat() regardless of store size. We never write here.
        self.legacy_root = Path(legacy_root) if legacy_root else None
        # Legacy entries are ~90% Tavily by volume. Those hold structured results rather than
        # rendered Exa text, so serving them means TRANSLATING (render_tavily). The model still
        # gets real search results, but they did not come from Exa -- acceptable for cost,
        # not for a strict comparison against an Exa-only baseline.
        self.use_tavily = use_tavily
        self.threshold = fuzzy_threshold
        self.provider = provider
        self.exact_only = exact_only
        self.read_only = read_only
        self.index_path = Path(index_path) if index_path else None
        # Held only to append one entry or to start the loader thread, never for the build.
        self._lock = threading.Lock()
        # Lets one load() run at a time; lookups and put() never take it.
        self._load_lock = threading.Lock()
        # kind -> list[(norm, tokens, key)]. Rows are only appended, so a lookup can scan a list
        # without a lock while the build adds to it. Dropping rows replaces the list instead.
        self._index: Dict[str, list] = {"search": [], "browse": []}
        # kind -> keys already in _index. The index file, the walk and put() can all reach the
        # same entry; this keeps it to one index row.
        self._indexed: Dict[str, set] = {"search": set(), "browse": set()}
        # Set once fuzzy matching may run: after the index file is read, or else after the walk.
        self._ready = threading.Event()
        self._loader: Optional[threading.Thread] = None
        self.stats = collections_counter()
        self._loaded = False

    # ---- paths -------------------------------------------------------------------------------
    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2] / key[2:4] / f"{key}.json"

    @staticmethod
    def _key(norm: str) -> str:
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    # ---- index -------------------------------------------------------------------------------
    def load_in_background(self) -> None:
        """Start load() on a daemon thread and return at once. A no-op once started, or if exact_only."""
        if self.exact_only or self._loaded or self._loader is not None:
            return
        with self._lock:
            if self._loader is None:
                # Daemon, so a half-done walk never holds up interpreter exit.
                self._loader = threading.Thread(target=self._load_logging_errors, name="exa-cache-index", daemon=True)
                self._loader.start()

    def _load_logging_errors(self) -> None:
        try:
            self.load()
        except Exception as e:  # noqa: BLE001 - the loader thread's boundary: log it and keep serving
            fallback = "fuzzy matching covers what was indexed" if self._ready.is_set() else "serving exact hits only"
            print(f"[exa-cache] fuzzy index build failed ({type(e).__name__}: {e}); {fallback}", flush=True)

    def load(self) -> None:
        """Build the fuzzy index. Synchronous and idempotent; the server calls load_in_background().

        Reads the index file if there is a usable one, which makes the index ready at once, then
        walks the store: indexes the entries the index does not hold yet, drops the keys whose
        entries are gone and, if it did either, rewrites the index file.
        """
        if self.exact_only:
            return
        with self._load_lock:
            if self._loaded:
                return
            started = time.time()
            # The index file records this, so that a file saved for another store is not used.
            # Resolved, so that a symlink to the store still finds the store's file.
            root = str(self.root.resolve())
            from_file = self._read_index_file(root)
            if from_file is not None:
                self._set_ready(started)
            from_store, unlisted = self._walk()
            dropped = self._drop_missing(unlisted)
            self._set_ready(started)
            self.stats["index_from_file"] = from_file or 0
            self.stats["index_from_store"] = from_store
            self.stats["index_dropped"] = dropped
            self.stats["index_entries"] = self._index_size()
            self.stats["index_load_ms"] = int((time.time() - started) * 1000)
            if (from_store or dropped) and self.index_path is not None and not self.read_only:
                self._save_index_file(root)
            self._loaded = True
        print(
            f"[exa-cache] fuzzy index loaded: {self.stats['index_entries']} entries ({from_file or 0} from the "
            f"index file, {from_store} from the store, {dropped} dropped as gone) in "
            f"{self.stats['index_load_ms']} ms",
            flush=True,
        )

    def _set_ready(self, started: float) -> None:
        if self._ready.is_set():
            return
        self.stats["index_ready_ms"] = int((time.time() - started) * 1000)
        self._ready.set()
        print(
            f"[exa-cache] fuzzy index ready: {self._index_size()} entries after {self.stats['index_ready_ms']} ms",
            flush=True,
        )

    def _index_size(self) -> int:
        return sum(len(rows) for rows in list(self._index.values()))

    def _add(self, kind: str, key: str, norm: str) -> bool:
        """Index one entry. False if the key is not an entry name or is indexed already."""
        if not _KEY_RE.fullmatch(key):
            return False
        tokens = _tokens(norm)
        with self._lock:
            indexed = self._indexed.setdefault(kind, set())
            if key in indexed:
                return False
            indexed.add(key)
            self._index.setdefault(kind, []).append((norm, tokens, key))
        return True

    def _walk(self) -> Tuple[int, Dict[str, set]]:
        """Index every stored entry not indexed yet.

        Returns how many were added and, for each kind whose directory exists, the keys that were
        indexed when its walk started but that it did not list. Serial on purpose: no worker pool
        whose queued reads could hold up interpreter exit.
        """
        n = 0
        unlisted: Dict[str, set] = {}
        for kind in ("search", "browse"):
            base = self.root / kind
            if not base.is_dir():
                continue  # a missing directory is not evidence that its entries are gone
            indexed = self._indexed[kind]
            with self._lock:
                not_listed = unlisted[kind] = set(indexed)
            for path in base.glob("*/*/*.json"):
                key = path.stem
                if key in indexed:
                    not_listed.discard(key)
                    continue  # listed in the index file or put() already; no need to read it
                try:
                    rec = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue  # a torn or deleted entry is a miss, never an error
                norm = rec.get("norm") if isinstance(rec, dict) else None
                if not isinstance(norm, str) or not norm:
                    continue
                if self._add(kind, key, norm):
                    n += 1
        return n, unlisted

    def _drop_missing(self, unlisted: Dict[str, set]) -> int:
        """Drop the index rows whose entries are gone, so they stop hiding near-duplicates; return how many.

        get() serves only the best fuzzy match, so a row whose entry is gone turns a hit on the
        next-best entry into a miss. Each key the walk did not list is checked once more, because
        put() may have stored it after the walk passed its directory.
        """
        n = 0
        for kind, keys in unlisted.items():
            gone = {key for key in keys if not os.path.exists(self._path(kind, key))}
            if not gone:
                continue
            with self._lock:
                self._indexed[kind] -= gone
                # A new list, not removal in place: a lookup still scanning the old one is unaffected.
                self._index[kind] = [row for row in self._index[kind] if row[2] not in gone]
            n += len(gone)
        return n

    def _read_index_file(self, root: str) -> Optional[int]:
        """Index the entries the index file lists and return how many; None if there is no usable file."""
        if self.index_path is None:
            return None
        try:
            data = json.loads(self.index_path.read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            print(f"[exa-cache] index file unreadable ({type(e).__name__}); rebuilding from the store", flush=True)
            return None
        entries = (
            data.get("entries") if isinstance(data, dict) and data.get("version") == _INDEX_FILE_VERSION else None
        )
        if not isinstance(entries, dict) or not all(isinstance(pairs, dict) for pairs in entries.values()):
            print("[exa-cache] index file has an unknown format; rebuilding from the store", flush=True)
            return None
        if data.get("root") != root:
            # Its keys would be fuzzy candidates whose entries are not in this store.
            print("[exa-cache] index file was saved for another exa_cache_root; rebuilding from the store", flush=True)
            return None
        n = 0
        for kind, pairs in entries.items():
            for key, norm in pairs.items():
                if isinstance(norm, str) and norm and self._add(kind, key, norm):
                    n += 1
        return n

    def _save_index_file(self, root: str) -> None:
        """Write {"version", "root", "entries": {kind: {key: norm}}} to index_path via a tmp file + os.replace."""
        # put() only appends, so these copies are a consistent prefix and put() is never blocked.
        entries = {kind: {key: norm for norm, _, key in list(rows)} for kind, rows in list(self._index.items())}
        tmp = None
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            # mkstemp creates the file 0600, as it does for every entry: the file holds the same queries.
            fd, tmp = tempfile.mkstemp(dir=str(self.index_path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump({"version": _INDEX_FILE_VERSION, "root": root, "entries": entries}, fh)
            os.replace(tmp, self.index_path)
        except OSError as e:
            print(f"[exa-cache] index file not saved ({type(e).__name__}): {e}", flush=True)
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    # ---- read/write --------------------------------------------------------------------------
    def get(self, kind: str, text: str) -> Tuple[Optional[Any], str]:
        """Return (payload, how) where how is 'exact' | 'legacy' | 'fuzzy' | 'miss'.

        Exact and legacy_root keys are read first and never wait for the index. Fuzzy matching runs
        only when exact_only is off and the index is ready; until it is ready, a lookup that would
        need it is a miss, counted as fuzzy_skipped, and starts the build if nothing has.
        """
        norm = normalize(text, kind)
        key = self._key(norm)
        rec = self._read(kind, key)
        if rec is not None:
            self.stats["hit_exact"] += 1
            return rec.get("payload"), "exact"

        legacy = self._read_legacy(kind, key)
        if legacy is not None:
            self.stats["hit_legacy"] += 1
            return legacy, "legacy"

        if self.exact_only:
            self.stats["miss"] += 1
            return None, "miss"
        if not self._ready.is_set():
            self.load_in_background()
            self.stats["fuzzy_skipped"] += 1
            self.stats["miss"] += 1
            return None, "miss"

        toks = _tokens(norm)
        best, best_score = None, 0.0
        for cand_norm, cand_toks, cand_key in self._index.get(kind, ()):
            score = _jaccard(toks, cand_toks)
            if score > best_score:
                best, best_score = cand_key, score
        if best is not None and best_score >= self.threshold:
            rec = self._read(kind, best)
            if rec is not None:
                self.stats["hit_fuzzy"] += 1
                return rec.get("payload"), "fuzzy"
        self.stats["miss"] += 1
        return None, "miss"

    def put(self, kind: str, text: str, payload: Any) -> bool:
        """Store one entry and index it; return True once it is on disk, False if nothing was written."""
        if self.read_only:
            self.stats["put_skipped_read_only"] += 1
            return False
        norm = normalize(text, kind)
        key = self._key(norm)
        path = self._path(kind, key)
        rec = {
            "query" if kind == "search" else "url": text,
            "norm": norm,
            "provider": self.provider,
            "payload": payload,
            "ts": int(time.time()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # tmp + replace: atomic within the directory, so a concurrent reader never sees a
            # partial file and two writers of the same key are last-wins on identical content.
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
            os.replace(tmp, path)
        except OSError as e:
            # A cache that cannot write must not break a rollout. Lustre returns EIO on roughly 1%
            # of creates here, which is exactly the case this swallows.
            self.stats["put_error"] += 1
            print(f"[exa-cache] put failed ({type(e).__name__}): {e}", flush=True)
            return False
        if not self.exact_only:
            self._add(kind, key, norm)
        self.stats["put"] += 1
        return True

    def _read_legacy(self, kind: str, key: str) -> Optional[str]:
        """Exact-key read from the legacy store, re-enveloped into what the plugin parses.

        Legacy records store `results`, not `payload`: a rendered SERP string for Exa entries, a
        list of structured results for Tavily ones. Both have to be wrapped in the one-frame SSE
        envelope before the plugin's parseMcpBody can read them.
        """
        if self.legacy_root is None:
            return None
        path = self.legacy_root / kind / key[:2] / key[2:4] / f"{key}.json"
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        results = rec.get("results")
        provider = rec.get("provider") or "tavily"  # untagged legacy entries predate the field
        if isinstance(results, str):
            text = results  # exa: already the rendered SERP
        elif isinstance(results, list):
            if not self.use_tavily and provider != "exa":
                self.stats["legacy_tavily_skipped"] += 1
                return None
            text = render_tavily(results)  # tavily: structured -> rendered
            self.stats["legacy_tavily_served"] += 1
        else:
            return None
        if not text.strip():
            return None
        return to_sse(text)

    def _read(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(kind, key)
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        # Never serve an entry produced by a different provider: the rendered text differs.
        if rec.get("provider") not in (None, self.provider):
            return None
        return rec


def collections_counter():
    from collections import Counter

    return Counter()
