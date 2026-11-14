# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Global, persistent flat-file cache for search/browse provider results.

Ported from the audited flat-file cache of an internal reference harness, which
also holds the design record and the cross-node POC evidence.

Why flat files (not SQLite) for the VALUE store: rollouts run across many nodes
on lustre. A shared SQLite file written from many nodes corrupts / serializes
catastrophically (cross-node POSIX locks are unreliable; WAL's -shm can't cross
nodes). Content-addressed files sidestep the whole problem: each key is its own
file, writes are atomic (temp + os.replace), reads need no lock, and a same-key
race is last-writer-wins (redundant, never corrupt). This also gives LIVE
cross-run and cross-rollout sharing for free — critical for GRPO, where N
repeats of a sample issue near-identical queries at the same time.

The FUZZY INDEX is a different access pattern — similarity search, not key
lookup — and is NOT flat-file. It lives in a read-only, immutable SQLite FTS5
master (sqlite_fuzzy_index.py), which never takes a lock and never writes a
-wal/-shm, so the cross-node objection above does not apply to it. Writes stay
on the flat-file side: this process's new norms go to its own append-only .txt
shard, which build_cache_index.py merges into the master offline.

Layout (under <root>):
    search/<ab>/<cd>/<sha256(norm_query)>.json -> {"query","norm","provider","results":[...], ...extra}
    browse/<ab>/<cd>/<sha256(norm_url)>.json   -> {"url","norm","provider","content": "...", "max_characters"?}

Single key namespace across providers. Each entry is tagged with the provider
that produced it ("tavily" | "exa"); entries written by the reference harness
have no "provider" field and read as "tavily". Write rule: exa entries are
preferred — a tavily write does not replace an existing exa entry unless that
entry is a page the provider cut at its character cap. A tavily write that finds
no entry publishes create-only (a hard link), so an exa entry written meanwhile
stays. A tavily write that replaces an older entry can still race an exa write
of the same key and leave the tavily entry; both are valid entries.

Store-raw / format-on-read: we cache the raw provider result list (search) and
raw page content (browse). All block-formatting + truncation stays in the
caller. The caller decides whether a stored entry fits the request (see the
`accept` argument of get_search_entry and `min_characters` of get_extract);
an entry that does not fit, or that is malformed, reads as a miss.

The cache is best-effort: a failed write (no space, quota, permission, stale
handle) is logged and counted, and the caller still returns the provider's
result.

Invariant: only provider responses enter the cache, never model-authored text —
otherwise a rollout could plant answers for its GRPO siblings.
"""

import hashlib
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Iterator
from os import PathLike
from time import perf_counter, time
from typing import Literal

from pydantic import JsonValue

from resources_servers.browsecomp_advanced_harness.rollout_context import (
    current_rollout_id,
    log_line,
    record_phase,
)
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_TOKENIZER,
    SqliteFuzzyIndex,
    master_db_path,
)


LOG = logging.getLogger(__name__)
CacheProvider = Literal["tavily", "exa"]
SearchResults = list[dict[str, JsonValue]]
SearchEntry = dict[str, JsonValue]

_WS_RE = re.compile(r"\s+")

INDEX_DIR = "index"
INDEX_BASE = "search_norms.txt"  # written by build_cache_index.py (offline scan)


def normalize_query(query: str | None) -> str:
    """Lowercase, strip surrounding quotes, collapse whitespace."""
    q = (query or "").strip()
    # strip a single layer of matching surrounding quotes
    if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
        q = q[1:-1]
    q = _WS_RE.sub(" ", q).strip().lower()
    return q


def normalize_url(url: str | None) -> str:
    """Drop #fragment and a single trailing slash."""
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    if u.endswith("/"):
        u = u[:-1]
    return u


def _page_was_capped(entry: SearchEntry) -> bool:
    """True when a stored page filled the provider's character cap, so the page may go on."""
    cap = entry.get("max_characters")
    return isinstance(cap, int) and len(entry["content"]) >= cap


class SearchCache:
    """Content-addressed flat-file cache. Pass root=None/"" to disable (no-op)."""

    def __init__(
        self,
        root: str | PathLike[str] | None,
        *,
        normalize_queries: bool = True,
        fuzzy_threshold: float | None = None,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        tokenizer: str = DEFAULT_TOKENIZER,
        verbose_phase_log: bool = False,
        read_only: bool = False,
    ) -> None:
        """fuzzy_threshold: 0-1 similarity cutoff for SEARCH-only fuzzy matching
        (None = exact only). On an exact miss, the best token_sort_ratio match
        over the norms index is served if >= threshold. Serve-only: fuzzy hits
        are never re-written under the new key. Browse is always exact.

        candidate_limit/tokenizer tune FTS5 candidate generation; see
        sqlite_fuzzy_index.py. They affect RECALL only, never match quality.

        read_only: serve from this root, never write to it. For pointing a run
        at a cache root owned by someone else -- entries are 0644 and dirs 0755,
        so reads work and writes do not. A writable cache on such a root is
        refused here at startup; without that check every write would fail and
        be logged. Reads are untouched, including fuzzy: the FTS5 master is
        already opened mode=ro&immutable=1, and fuzzy has always been serve-only.

        Raises ValueError for a fuzzy_threshold outside (0, 1], a read-only root
        that does not exist, or a writable root this process cannot write."""
        if fuzzy_threshold is not None and not 0 < fuzzy_threshold <= 1:
            raise ValueError(f"fuzzy_threshold must be in (0, 1] or None, got {fuzzy_threshold!r}")
        self.root = str(root) if root else None
        self.enabled = bool(self.root)
        self.read_only = bool(read_only)
        # Per-CALL get/put lines. Off by default: gets run several per turn, so
        # this is the one knob here that can multiply driver-log size. The
        # per-rollout phase totals accumulate either way.
        self.verbose_phase_log = bool(verbose_phase_log)
        self.normalize_queries = normalize_queries
        self.fuzzy_threshold = fuzzy_threshold
        self._lock = threading.Lock()
        self._stats = {
            "search_hits": 0,
            "search_misses": 0,
            "search_fuzzy_hits": 0,
            "browse_hits": 0,
            "browse_misses": 0,
            # In-process serve-latency accounting (get_search/get_extract wall
            # time, INCLUDING lustre reads and the fuzzy scan). An earlier run
            # had no way to answer "did cache hits queue up under load?" — averages hide bursts, so track max and a >0.5s counter.
            "search_get_s": 0.0,
            "search_get_max_s": 0.0,
            "search_get_slow": 0,  # calls > 0.5s
            "browse_get_s": 0.0,
            "browse_get_max_s": 0.0,
            "browse_get_slow": 0,
            "put_errors": 0,
        }
        self._fuzzy_index = None
        self._shard_path = None
        # Read-only creates nothing: the dirs we need are the ones already
        # there, and makedirs is the first call that fails on a foreign root.
        if self.enabled and self.read_only and not os.path.isdir(self.root):
            raise ValueError(f"read-only search cache root {self.root!r} does not exist")
        if self.enabled and not self.read_only:
            os.makedirs(os.path.join(self.root, "search"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "browse"), exist_ok=True)
            unwritable = [
                d for d in ("search", "browse") if not os.access(os.path.join(self.root, d), os.W_OK | os.X_OK)
            ]
            if unwritable:
                raise ValueError(
                    f"search cache root {self.root!r} is not writable ({', '.join(unwritable)}); "
                    "set search_cache_read_only to serve from it without writing"
                )
        if self.enabled and self.fuzzy_threshold:
            index_dir = os.path.join(self.root, INDEX_DIR)
            if not self.read_only:
                os.makedirs(index_dir, exist_ok=True)
            self._fuzzy_index = SqliteFuzzyIndex(
                self.root, threshold=self.fuzzy_threshold, candidate_limit=candidate_limit, tokenizer=tokenizer
            )
            self._fuzzy_index.add_many(self._unmerged_norms(index_dir))
            # No shard when read-only: a shard is a write, and a process that
            # caches nothing has nothing for build_cache_index.py to merge.
            if not self.read_only:
                self._shard_path = os.path.join(index_dir, f"search_norms_{os.uname().nodename}_{os.getpid()}.txt")

    @property
    def index_size(self) -> int:
        """Norms reachable by fuzzy matching: FTS5 master + this run's additions."""
        if self._fuzzy_index is None:
            return 0
        return self._fuzzy_index.master_count + self._fuzzy_index.local_count

    def _unmerged_norms(self, index_dir: str) -> Iterator[str]:
        """Norms from the .txt index that master.db does not already contain.

        With no master (a cold cache root), that is the whole text index. With a
        master, it is only the append shards written after the master was built
        — the sidecar model: the master holds everything merged offline by
        build_cache_index.py, the shards hold what has been cached since.

        This is what keeps startup cheap. Loading every .txt would rebuild the
        ~1 GB in-memory list the FTS5 index exists to eliminate.

        mtime is the merge marker. A shard written *during* a master build can be
        skipped; the cost is a recall miss (one live provider call), never a
        wrong answer.
        """
        master = master_db_path(self.root)
        cutoff = os.path.getmtime(master) if os.path.exists(master) else None
        if not os.path.isdir(index_dir):
            return  # read-only against a root with no index yet: exact-only
        for fname in sorted(os.listdir(index_dir)):
            if not fname.endswith(".txt"):
                continue
            if cutoff is not None and (
                fname == INDEX_BASE or os.path.getmtime(os.path.join(index_dir, fname)) <= cutoff
            ):
                continue
            try:
                with open(os.path.join(index_dir, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        norm = line.rstrip("\n")
                        if norm:
                            yield norm
            except OSError:
                continue  # a shard mid-write/unreadable only shrinks fuzzy recall

    # ---- internals --------------------------------------------------------
    def _path(self, kind, norm_key):
        h = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()
        d = os.path.join(self.root, kind, h[:2], h[2:4])
        return d, os.path.join(d, h + ".json")

    def _read(self, kind, norm_key):
        """The stored entry, or None when it is missing, unreadable or malformed."""
        _, path = self._path(kind, norm_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, ValueError):
            return None  # missing, unreadable, invalid UTF-8 or invalid JSON -> miss
        value_key, value_type = ("results", list) if kind == "search" else ("content", str)
        if not isinstance(obj, dict) or not isinstance(obj.get(value_key), value_type):
            return None  # an entry of another shape is a miss, never an empty hit
        return obj

    def _write(self, kind, norm_key, payload, *, create_only: bool = False) -> bool:
        """Publish one entry atomically. A failure is logged and counted, never raised:
        the caller already holds the provider's result and must still return it.

        `create_only` publishes only if no entry exists yet and returns False when
        another writer got there first. On a filesystem without hard links it falls
        back to a plain replace."""
        _t0 = perf_counter()
        d, path = self._path(kind, norm_key)
        # A per-write random suffix: pid and thread id repeat across nodes sharing one root.
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            os.makedirs(d, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            if not create_only:
                os.replace(tmp, path)  # atomic publish
            else:
                try:
                    os.link(tmp, path)  # atomic publish that fails if an entry exists
                except FileExistsError:
                    os.unlink(tmp)
                    return False
                except OSError:
                    os.replace(tmp, path)  # no hard links on this filesystem
                else:
                    os.unlink(tmp)
        except (OSError, TypeError, ValueError) as error:
            self._bump("put_errors")
            LOG.warning("[cache][put_error] rid=%s kind=%s error=%r", current_rollout_id(), kind, error)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        # A lustre write sitting on the rollout's critical path, previously silent.
        _elapsed = perf_counter() - _t0
        record_phase(f"cache_put_{kind}", _elapsed)
        if self.verbose_phase_log:
            log_line("[cache][put]", dur=_elapsed, kind=kind)
        return True

    def _put(self, kind, norm_key, provider, payload) -> bool:
        """Write one entry under the prefer-exa rule: a tavily write does not replace an
        exa entry, except a browse page exa cut at its character cap (tavily's copy is
        fuller). A tavily write that finds no entry publishes create-only, so an exa
        entry written meanwhile stays. Returns whether this write published."""
        if provider != "tavily":
            return self._write(kind, norm_key, payload)
        existing = self._read(kind, norm_key)
        if existing is None:
            # A malformed entry also reads as None; only a missing one is published create-only.
            create_only = not os.path.exists(self._path(kind, norm_key)[1])
            return self._write(kind, norm_key, payload, create_only=create_only)
        if existing.get("provider", "tavily") == "exa" and not (kind == "browse" and _page_was_capped(existing)):
            return False
        return self._write(kind, norm_key, payload)

    def _bump(self, name):
        with self._lock:
            self._stats[name] += 1

    def _time_add(self, kind, elapsed_s):
        """Accumulate get-path wall time; log outliers so bursts are visible live."""
        with self._lock:
            self._stats[f"{kind}_get_s"] += elapsed_s
            if elapsed_s > self._stats[f"{kind}_get_max_s"]:
                self._stats[f"{kind}_get_max_s"] = elapsed_s
            if elapsed_s > 0.5:
                self._stats[f"{kind}_get_slow"] += 1
        record_phase(f"cache_get_{kind}", elapsed_s)
        if self.verbose_phase_log:
            log_line("[cache][get]", dur=elapsed_s, kind=kind)
        if elapsed_s > 0.5:
            LOG.info("[cache][slow] rid=%s %s get took %.2fs", current_rollout_id(), kind, elapsed_s)

    # ---- search -----------------------------------------------------------
    def _norm_q(self, query):
        return normalize_query(query) if self.normalize_queries else (query or "").strip()

    def _fuzzy_lookup(self, norm, accept=None):
        """Best token_sort_ratio-equivalent match over the norms index, or None.
        Word-order insensitive (query + candidates compared in token-sorted
        form); the 0-1 threshold maps onto rapidfuzz's 0-100 score.

        Candidate generation moved to FTS5 (sqlite_fuzzy_index.py); the scorer
        and threshold are unchanged, so a candidate that reaches rapidfuzz here
        is judged exactly as the reference harness's full scan judged it."""
        # Two clocks on purpose. perf_counter measures the DURATION (monotonic,
        # immune to wall-clock steps); time() stamps the START so the line can be
        # drawn as the span [t, t+dur]. Emitting a perf_counter value as `t=` is
        # the specific bug the gantt cannot detect -- it just draws fuzzy near
        # the epoch and silently ruins the timeline.
        _wall0 = time()
        _t0 = perf_counter()
        matched_norm = self._fuzzy_index.lookup(norm)
        _elapsed = perf_counter() - _t0
        # Fuzzy is the path that once starved a training run, so its cost is recorded on
        # every lookup -- including the misses, which a hit-only print could
        # never show.
        record_phase("cache_fuzzy", _elapsed)

        obj = None if matched_norm is None else self._read("search", matched_norm)
        if matched_norm is None:
            status = "miss"
        elif obj is None:
            status = "stale"  # index entry survived, the entry itself did not
        elif accept is not None and not accept(obj):
            obj, status = None, "unfit"  # stored for a request of another shape
        else:
            status = "hit"

        # log_line, not print: it is what stamps rid= and an absolute t=, and it
        # is what makes this line joinable with [browsecomp][provider]/[gen]/
        # [bash][timing] on one timeline. Logged on EVERY outcome now -- a miss
        # and a stale entry both pay the full FTS5 lookup and return nothing, so
        # they are exactly the cases worth seeing on the chart.
        log_line(
            "[cache][fuzzy]",
            t=_wall0,
            dur=_elapsed,
            status=status,
            norm=repr(norm),
            via=repr(matched_norm) if matched_norm is not None else None,
        )
        return obj

    def get_search_entry(
        self, query: str, *, accept: Callable[[SearchEntry], bool] | None = None
    ) -> SearchEntry | None:
        """Return the stored entry for this query, or None on miss/disabled.

        `accept` lets the caller refuse an entry stored for a request of another shape
        (for example another exa search type); a refused entry is a miss.

        Fuzzy matching serves a DIFFERENT query's results when the two score at or
        above the threshold (for example a query about 2018 can be served the cached
        results for 2019), shown under the agent's own query. It changes what the
        agent retrieves; leave it off where retrieval must match a live run."""
        if not self.enabled:
            return None
        _t0 = perf_counter()
        try:
            norm = self._norm_q(query)
            obj = self._read("search", norm)
            if obj is not None and accept is not None and not accept(obj):
                obj = None
            if obj is None and self.fuzzy_threshold:
                obj = self._fuzzy_lookup(norm, accept)
                if obj is not None:
                    self._bump("search_fuzzy_hits")
            if obj is None:
                self._bump("search_misses")
                return None
            self._bump("search_hits")
            return obj
        finally:
            self._time_add("search", perf_counter() - _t0)

    def get_search(self, query: str) -> tuple[CacheProvider, SearchResults] | None:
        """Return (provider, raw result list), or None on miss/disabled."""
        obj = self.get_search_entry(query)
        return None if obj is None else (obj.get("provider", "tavily"), obj["results"])

    def _index_add(self, norm):
        """Record a newly cached norm: in-memory + this process's append shard.

        The shard .txt is unchanged — it stays the durable sidecar that
        build_cache_index.py merges into master.db offline."""
        if not self._fuzzy_index.add(norm):
            return
        if self._shard_path:
            try:
                with open(self._shard_path, "a", encoding="utf-8") as f:
                    f.write(norm + "\n")
            except OSError:
                pass  # index is best-effort; the entry itself is already cached

    def put_search(
        self,
        query: str,
        results: SearchResults,
        provider: CacheProvider,
        *,
        extra: dict[str, JsonValue] | None = None,
    ) -> None:
        """Store one provider result list. `extra` holds further response fields the
        caller formats from (exa's deep-search `output`) and the request shape it
        checks on read; it cannot replace the fixed fields."""
        if not self.enabled or self.read_only:
            return
        norm = self._norm_q(query)
        entry = {**(extra or {}), "query": query, "norm": norm, "provider": provider, "results": results}
        if self._put("search", norm, provider, entry) and self.fuzzy_threshold:
            self._index_add(norm)

    # ---- browse / extract -------------------------------------------------
    def get_extract(self, url: str, *, min_characters: int | None = None) -> tuple[CacheProvider, str] | None:
        """Return (provider, page content), or None on miss/disabled.

        `min_characters`: the page length the caller would request from a provider
        that caps pages (exa's maxCharacters). A page stored under a smaller cap
        that it filled is a miss, so a short copy never stands in for a longer fetch."""
        if not self.enabled:
            return None
        _t0 = perf_counter()
        try:
            obj = self._read("browse", normalize_url(url))
            if obj is not None and min_characters is not None and _page_was_capped(obj):
                if obj["max_characters"] < min_characters:
                    obj = None
            if obj is None:
                self._bump("browse_misses")
                return None
            self._bump("browse_hits")
            return obj.get("provider", "tavily"), obj["content"]
        finally:
            self._time_add("browse", perf_counter() - _t0)

    def put_extract(
        self, url: str, content: str, provider: CacheProvider, *, max_characters: int | None = None
    ) -> None:
        """Store one page. `max_characters` is the cap the provider applied, if any."""
        if not self.enabled or self.read_only:
            return
        norm = normalize_url(url)
        entry = {"url": url, "norm": norm, "provider": provider, "content": content}
        if max_characters is not None:
            entry["max_characters"] = max_characters
        self._put("browse", norm, provider, entry)

    # ---- stats ------------------------------------------------------------
    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._stats)

    def stats_report(self) -> dict[str, int | float]:
        """Stats dict enriched with totals + hit percentages."""
        s = self.stats()
        sh, sm = s["search_hits"], s["search_misses"]
        bh, bm = s["browse_hits"], s["browse_misses"]
        return {
            **s,
            "search_total": sh + sm,
            "browse_total": bh + bm,
            "search_hit_pct": round(100.0 * sh / (sh + sm), 1) if (sh + sm) else 0.0,
            "browse_hit_pct": round(100.0 * bh / (bh + bm), 1) if (bh + bm) else 0.0,
        }

    def summary_str(self) -> str:
        """Compact one-line summary, e.g. 'cache: search 12/40 (30%), browse 7/10 (70%)'."""
        if not self.enabled:
            return ""
        r = self.stats_report()
        fuzzy = " [{search_fuzzy_hits} fuzzy]".format(**r) if self.fuzzy_threshold else ""
        lat = ""
        if r["search_total"] or r["browse_total"]:
            s_avg = 1000.0 * r["search_get_s"] / r["search_total"] if r["search_total"] else 0.0
            b_avg = 1000.0 * r["browse_get_s"] / r["browse_total"] if r["browse_total"] else 0.0
            lat = (
                f" | get-lat search avg {s_avg:.0f}ms max {r['search_get_max_s']:.2f}s"
                f" ({r['search_get_slow']} >0.5s), browse avg {b_avg:.0f}ms"
                f" max {r['browse_get_max_s']:.2f}s ({r['browse_get_slow']} >0.5s)"
            )
        errors = f" | put errors {r['put_errors']}" if r["put_errors"] else ""
        return (
            (
                "cache: search {search_hits}/{search_total} ({search_hit_pct:.0f}%)" + fuzzy + ", "
                "browse {browse_hits}/{browse_total} ({browse_hit_pct:.0f}%)"
            ).format(**r)
            + lat
            + errors
        )
