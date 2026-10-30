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
    search/<ab>/<cd>/<sha256(norm_query)>.json -> {"query","norm","provider","results":[...]}
    browse/<ab>/<cd>/<sha256(norm_url)>.json   -> {"url","norm","provider","content": "..."}
    pack/                                      optional read-only pack of the same entries
                                               (search_cache_pack.py, build_cache_pack.py)

Reads go to the flat file first; only when it does not exist is the pack
consulted (when pack/MANIFEST.json exists). Writes always go to flat files.

Single key namespace across providers. Each entry is tagged with the provider
that produced it ("tavily" | "exa"); legacy entries migrated from the reference
harness have no "provider" field and read as "tavily". Write rule: exa entries are
preferred — a tavily write never clobbers an existing exa entry.

Store-raw / format-on-read: we cache the raw provider result list (search) and
raw page content (browse). All block-formatting + truncation stays in the
caller, so a cache HIT produces byte-identical output to a MISS.

Invariant: only provider responses enter the cache, never model-authored text —
otherwise a rollout could plant answers for its GRPO siblings.
"""

import hashlib
import json
import os
import re
import threading
from time import perf_counter, time

from resources_servers.browsecomp_advanced_harness.rollout_context import (
    current_rollout_id,
    log_line,
    record_phase,
)
from resources_servers.browsecomp_advanced_harness.search_cache_pack import open_pack
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_TOKENIZER,
    SqliteFuzzyIndex,
    master_db_path,
)


_WS_RE = re.compile(r"\s+")

INDEX_DIR = "index"
INDEX_BASE = "search_norms.txt"  # written by build_cache_index.py (offline scan)


def normalize_query(query):
    """Lowercase, strip surrounding quotes, collapse whitespace."""
    q = (query or "").strip()
    # strip a single layer of matching surrounding quotes
    if len(q) >= 2 and q[0] == q[-1] and q[0] in ("'", '"'):
        q = q[1:-1]
    q = _WS_RE.sub(" ", q).strip().lower()
    return q


def normalize_url(url):
    """Drop #fragment and a single trailing slash."""
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    if u.endswith("/"):
        u = u[:-1]
    return u


class SearchCache:
    """Content-addressed flat-file cache. Pass root=None/"" to disable (no-op)."""

    def __init__(
        self,
        root,
        normalize_queries=True,
        fuzzy_threshold=None,
        candidate_limit=DEFAULT_CANDIDATE_LIMIT,
        tokenizer=DEFAULT_TOKENIZER,
        verbose_phase_log=False,
        read_only=False,
    ):
        """fuzzy_threshold: 0-1 similarity cutoff for SEARCH-only fuzzy matching
        (None = exact only). On an exact miss, the best token_sort_ratio match
        over the norms index is served if >= threshold. Serve-only: fuzzy hits
        are never re-written under the new key. Browse is always exact.

        candidate_limit/tokenizer tune FTS5 candidate generation; see
        sqlite_fuzzy_index.py. They affect RECALL only, never match quality.

        read_only: serve from this root, never write to it. For pointing a run
        at a cache root owned by someone else -- entries are 0644 and dirs 0755,
        so reads work and writes do not. Not merely tidiness: `_write` does not
        catch OSError, and its call site in `_search_one_to_disk` sits OUTSIDE
        that function's try/except, so an unwritable root turns a search that
        SUCCEEDED into "Search failed: [Errno 13] Permission denied" for the
        model -- after the provider call was already paid for. Reads are
        untouched, including fuzzy: the FTS5 master is already opened
        mode=ro&immutable=1, and fuzzy has always been serve-only."""
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
            # Hits the pack served because the flat file was absent (fuzzy
            # included); subsets of search_hits / browse_hits. pack_errors is
            # the reader's own counter, merged in by stats().
            "search_pack_hits": 0,
            "browse_pack_hits": 0,
        }
        # On iff <root>/pack/MANIFEST.json exists. open_pack raises when the
        # manifest lists a shard file that is missing or the wrong size: failing
        # here beats serving silent misses that each become a paid provider call.
        self._pack = open_pack(self.root) if self.enabled else None
        self._fuzzy_index = None
        self._shard_path = None
        # Read-only creates nothing: the dirs we need are the ones already
        # there, and makedirs is the first call that fails on a foreign root.
        if self.enabled and not self.read_only:
            os.makedirs(os.path.join(self.root, "search"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "browse"), exist_ok=True)
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
    def pack_generation(self):
        """Generation id of the pack being served, or None when there is no pack."""
        return self._pack.generation if self._pack is not None else None

    @property
    def index_size(self):
        """Norms reachable by fuzzy matching: FTS5 master + this run's additions."""
        if self._fuzzy_index is None:
            return 0
        return self._fuzzy_index.master_count + self._fuzzy_index.local_count

    def _unmerged_norms(self, index_dir):
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

    def _read(self, kind, norm_key, count_pack_hit=False):
        """Flat file first; the pack only when the flat file does not exist.

        A flat file that exists but is corrupt or unreadable stays a miss: the
        pack never masks it. count_pack_hit bumps <kind>_pack_hits when the pack
        serves the entry -- the get paths pass it, the prefer-exa check does not."""
        _, path = self._path(kind, norm_key)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/unreadable -> treat as miss
        if self._pack is None:
            return None
        obj = self._pack.load(kind, hashlib.sha256(norm_key.encode("utf-8")).digest())
        if obj is not None and count_pack_hit:
            self._bump(f"{kind}_pack_hits")
        return obj

    def _write(self, kind, norm_key, payload):
        _t0 = perf_counter()
        d, path = self._path(kind, norm_key)
        os.makedirs(d, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic publish
        # A lustre write sitting on the rollout's critical path, previously silent.
        _elapsed = perf_counter() - _t0
        record_phase(f"cache_put_{kind}", _elapsed)
        if self.verbose_phase_log:
            log_line("[cache][put]", dur=_elapsed, kind=kind)

    def _tavily_would_clobber_exa(self, kind, norm_key, provider):
        """The prefer-exa write rule: a tavily write never replaces an exa entry."""
        if provider != "tavily":
            return False
        existing = self._read(kind, norm_key)
        return existing is not None and existing.get("provider", "tavily") == "exa"

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
            print(
                f"[cache][slow] rid={current_rollout_id()} {kind} get took {elapsed_s:.2f}s",
                flush=True,
            )

    # ---- search -----------------------------------------------------------
    def _norm_q(self, query):
        return normalize_query(query) if self.normalize_queries else (query or "").strip()

    def _fuzzy_lookup(self, norm):
        """Best token_sort_ratio-equivalent match over the norms index, or None.
        Word-order insensitive (query + candidates compared in token-sorted
        form); the 0-1 threshold maps onto rapidfuzz's 0-100 score.

        Candidate generation moved to FTS5 (sqlite_fuzzy_index.py); the scorer
        and threshold are unchanged, so a candidate that reaches rapidfuzz here
        is judged exactly as the old full scan judged it."""
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
        # every lookup -- including the misses, which the old hit-only print
        # could never show.
        record_phase("cache_fuzzy", _elapsed)

        obj = None if matched_norm is None else self._read("search", matched_norm, count_pack_hit=True)
        if matched_norm is None:
            status = "miss"
        elif obj is None:
            status = "stale"  # index entry survived, the entry itself did not
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

    def get_search(self, query):
        """Return (provider, raw result list), or None on miss/disabled."""
        if not self.enabled:
            return None
        _t0 = perf_counter()
        try:
            norm = self._norm_q(query)
            obj = self._read("search", norm, count_pack_hit=True)
            if obj is None and self.fuzzy_threshold:
                obj = self._fuzzy_lookup(norm)
                if obj is not None:
                    self._bump("search_fuzzy_hits")
            if obj is None:
                self._bump("search_misses")
                return None
            self._bump("search_hits")
            return obj.get("provider", "tavily"), obj.get("results", [])
        finally:
            self._time_add("search", perf_counter() - _t0)

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

    def put_search(self, query, results, provider):
        if not self.enabled or self.read_only:
            return
        norm = self._norm_q(query)
        if self._tavily_would_clobber_exa("search", norm, provider):
            return
        self._write("search", norm, {"query": query, "norm": norm, "provider": provider, "results": results})
        if self.fuzzy_threshold:
            self._index_add(norm)

    # ---- browse / extract -------------------------------------------------
    def get_extract(self, url):
        """Return (provider, page content), or None on miss/disabled."""
        if not self.enabled:
            return None
        _t0 = perf_counter()
        try:
            obj = self._read("browse", normalize_url(url), count_pack_hit=True)
            if obj is None:
                self._bump("browse_misses")
                return None
            self._bump("browse_hits")
            return obj.get("provider", "tavily"), obj.get("content", "")
        finally:
            self._time_add("browse", perf_counter() - _t0)

    def put_extract(self, url, content, provider):
        if not self.enabled or self.read_only:
            return
        norm = normalize_url(url)
        if self._tavily_would_clobber_exa("browse", norm, provider):
            return
        self._write("browse", norm, {"url": url, "norm": norm, "provider": provider, "content": content})

    # ---- stats ------------------------------------------------------------
    def stats(self):
        with self._lock:
            s = dict(self._stats)
        s["pack_errors"] = self._pack.errors if self._pack is not None else 0
        return s

    def stats_report(self):
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

    def summary_str(self):
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
        pack = ""
        if self._pack is not None:
            pack = (
                f" | pack hits search {r['search_pack_hits']}, browse {r['browse_pack_hits']},"
                f" errors {r['pack_errors']}"
            )
        return (
            (
                "cache: search {search_hits}/{search_total} ({search_hit_pct:.0f}%)" + fuzzy + ", "
                "browse {browse_hits}/{browse_total} ({browse_hit_pct:.0f}%)"
            ).format(**r)
            + lat
            + pack
        )
