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
"""FTS5 candidate generation for the search-norms fuzzy index.

Replaces the brute-force `rapidfuzz.extractOne` over an in-memory list of every
cached norm. That scan was O(n) per MISS, single-threaded, and GIL-held: at
1,863,474 norms it measured a 2.2 s mean and ~55,000 CPU-seconds inside a
~6,300 s training leg, which oversubscribed the one interpreter ~9x, starved the
shared thread pool that pocketshell also uses, and produced 668 stalls / zero
steps in that leg.

Two-stage retrieval. SQLite generates candidates through an inverted index;
rapidfuzz still does the scoring:

    1. FTS5 MATCH over master.db  -> <= candidate_limit candidates, bm25-ranked
    2. norms cached by THIS process since startup -> all of them (small)
    3. rapidfuzz token_sort_ratio over the union; best >= threshold wins

Match QUALITY is therefore unchanged -- same scorer, same threshold, same
token-order insensitivity. The only new failure mode is RECALL: FTS5 may not
surface a candidate the full scan would have found. That costs one live provider
call, never a wrong answer. `bench/parity_recall.py` measures it.

Two properties are load-bearing on Lustre:

- The master is opened `mode=ro&immutable=1`, so SQLite skips locking entirely
  and never creates `-wal`/`-shm` files. Those cannot cross nodes: the
  cross-node POC (32 tasks x 4 nodes) had shared-write SQLite crash 7/32 workers
  with a 34 s write p99, versus 0 crashes for flat files. NEVER open this
  read-write.
- `sqlite3` releases the GIL around query execution, so unlike the Python-list
  scan these lookups genuinely run in parallel across the harness thread pool.

The master is published by atomic rename, so a reader holding an open fd keeps
reading its old inode while a rebuild lands -- no coordination needed.

The flat-file VALUE store (`search/`, `browse/`) is untouched by this module. It
stays content-addressed, lock-free, and live-shared across nodes.
"""

import os
import sqlite3
import threading


try:
    from rapidfuzz import fuzz as _rf_fuzz
    from rapidfuzz import process as _rf_process
except ImportError:  # fuzzy matching becomes unavailable; exact lookups unaffected
    _rf_fuzz = _rf_process = None


INDEX_DIR = "index"
MASTER_DB = "master.db"

# Trigram FTS5 rejects terms shorter than 3 characters.
_MIN_TERM = {"trigram": 3, "unicode61": 2}

# Both defaults are measured, not guessed. bench/parity_recall.py over the live
# 1,892,440-norm index, 1000 perturbed real queries, threshold 0.8:
#
#   tokenizer  limit   recall    p50       p90       speedup
#   trigram      200   98.49%   31.0 ms   64.5 ms      104x
#   trigram     3000  100.00%   36.9 ms   83.9 ms      100x
#   unicode61   1000   99.90%    3.0 ms   20.1 ms    1,070x
#   unicode61   3000  100.00%    1.2 ms    4.3 ms    2,682x   <- default
#
# unicode61 indexes whole words, so a term is one posting-list lookup. trigram
# matches substrings, which costs 3x the disk (525 MB vs 223 MB) and an order of
# magnitude in query time, and buys nothing here: normalize_query already
# lowercases and collapses whitespace, and rapidfuzz handles the typo tolerance
# that trigram would otherwise provide.
#
# The limit matters because AND-of-2-terms can match thousands of rows and bm25
# ranks them on those two terms alone, so the true match can sit deep in the
# list. 3000 is where recall reached 100%; rescoring 3000 candidates in rapidfuzz
# is still microseconds.
DEFAULT_TOKENIZER = "unicode61"
DEFAULT_CANDIDATE_LIMIT = 3000


def master_db_path(cache_root):
    return os.path.join(cache_root, INDEX_DIR, MASTER_DB)


def token_sorted(norm):
    """ratio(sorted_a, sorted_b) == token_sort_ratio(a, b), pre-sorted once."""
    return " ".join(sorted(norm.split()))


def build_master_db(path, norms, tokenizer=DEFAULT_TOKENIZER):
    """Build the FTS5 master from `norms`. Returns the deduped row count.

    Writes to a temp file and atomically renames, so readers never observe a
    half-built index. `journal_mode=DELETE` keeps the result a single portable
    file with no `-wal`/`-shm` companions.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    for leftover in (tmp, f"{tmp}-journal"):
        if os.path.exists(leftover):
            os.remove(leftover)

    seen, ordered = set(), []
    for norm in norms:
        if norm and norm not in seen:
            seen.add(norm)
            ordered.append(norm)

    conn = sqlite3.connect(tmp)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(f"CREATE VIRTUAL TABLE norms USING fts5(norm, tokenize='{tokenizer}')")
        conn.executemany("INSERT INTO norms(norm) VALUES (?)", ((n,) for n in ordered))
        conn.commit()
    finally:
        conn.close()

    os.replace(tmp, path)
    return len(ordered)


class SqliteFuzzyIndex:
    """Read-only FTS5 master + a process-local tier for norms cached this run.

    The local tier exists because the master is a snapshot: entries this process
    caches mid-run are not in it, but must still be fuzzy-matchable (the shipped
    behaviour).

    It is an in-memory FTS5 table, NOT a list. An earlier version kept a plain
    list on the theory that "one run's misses" stays small enough to rescan.
    Production disproved that: one run cached ~50k norms in 1.3h and
    search-get latency went 17ms -> 92ms, slow-get rate 0.19% -> 2.03%, still
    climbing -- the same O(n) shape this module exists to remove, just with a
    smaller constant. Both tiers now go through the same indexed cascade.

    The in-memory DB is per-process and never touches Lustre, so it carries none
    of the cross-node locking concerns that keep the master read-only.
    """

    def __init__(self, root, threshold, candidate_limit=DEFAULT_CANDIDATE_LIMIT, tokenizer=DEFAULT_TOKENIZER):
        if _rf_process is None:
            raise ImportError("fuzzy matching requested but rapidfuzz is not installed")
        self.root = str(root)
        self.threshold = threshold
        self.candidate_limit = candidate_limit
        self.tokenizer = tokenizer
        self.last_candidate_count = 0

        self._db_path = master_db_path(self.root)
        self._tl = threading.local()
        self._broken = not os.path.exists(self._db_path)

        # Local tier: an in-memory FTS5 index over norms cached since startup.
        # check_same_thread=False + one lock: SQLite serializes writes anyway, and
        # a single shared connection keeps every thread reading the same rows
        # (thread-local connections cannot see each other's in-memory data).
        self._lock = threading.Lock()
        self._local_set = set()  # dedup, and the source of local_count
        self._local_db = sqlite3.connect(":memory:", check_same_thread=False)
        self._local_db.execute(f"CREATE VIRTUAL TABLE norms USING fts5(norm, tokenize='{self.tokenizer}')")

    # ---- master ----------------------------------------------------------
    def _conn(self):
        """Thread-local read-only connection, or None if the master is unusable."""
        if self._broken:
            return None
        conn = getattr(self._tl, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(f"file:{self._db_path}?mode=ro&immutable=1", uri=True)
                conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
            except sqlite3.Error:
                self._broken = True  # missing/corrupt master is a clean miss, forever
                return None
            self._tl.conn = conn
        return conn

    def _terms(self, norm):
        """The query's most selective tokens, most selective first.

        Length is a cheap stand-in for rarity: long tokens are the distinctive
        ones, short ones ("the", "of") have posting lists spanning most of the
        corpus and discriminate nothing.
        """
        min_len = _MIN_TERM.get(self.tokenizer, 3)
        return sorted({t for t in norm.split() if len(t) >= min_len}, key=len, reverse=True)

    @staticmethod
    def _stages(terms):
        """Progressively wider MATCH expressions, narrowest first.

        Narrowness is the whole game. `ORDER BY rank` computes bm25 for EVERY
        matching row before the LIMIT applies, so the cost is set by how many
        rows the expression matches, not by how many we keep. Measured over the
        live 1.89M-row master:

            OR of 12 terms  + rank    p50 331 ms   <- the obvious formulation
            OR of 2 terms   + rank    p50  69 ms
            AND of 2 terms  + rank    p50 1.7 ms   <- 200x, still ranked
            OR of 12 terms, no rank   p50 0.9 ms   <- fast but arbitrary rows

        So: AND the two most selective terms and let bm25 rank the small result.
        Widen only when that returns nothing, which happens when one of those
        terms is misspelled -- hence the any-two-of-three stage.
        """
        quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
        if len(quoted) >= 2:
            yield " AND ".join(quoted[:2])
        top3 = quoted[:3]
        if len(top3) >= 2:
            # Any two of the top three: survives one bad token.
            yield " OR ".join(f"({a} AND {b})" for i, a in enumerate(top3) for b in top3[i + 1 :])
        if quoted:
            yield quoted[0]

    def _candidates(self, conn, norm):
        """Run the narrowest-first cascade against one FTS5 connection."""
        for expr in self._stages(self._terms(norm)):
            try:
                rows = conn.execute(
                    "SELECT norm FROM norms WHERE norms MATCH ? ORDER BY rank LIMIT ?", (expr, self.candidate_limit)
                ).fetchall()
            except sqlite3.Error:
                return []  # malformed expression or corrupt page -> treat as no candidates
            if rows:
                return [r[0] for r in rows]
        return []

    def _master_candidates(self, norm):
        conn = self._conn()
        return [] if conn is None else self._candidates(conn, norm)

    def _local_candidates(self, norm):
        with self._lock:
            return self._candidates(self._local_db, norm)

    # ---- process-local tier ----------------------------------------------
    def add(self, norm):
        """Record a norm cached by this process so it becomes matchable now.

        Returns True if it was new, so callers can skip a redundant shard write.
        """
        with self._lock:
            if norm in self._local_set:
                return False
            self._local_set.add(norm)
            self._local_db.execute("INSERT INTO norms(norm) VALUES (?)", (norm,))
            return True

    def add_many(self, norms):
        for norm in norms:
            self.add(norm)

    @property
    def local_count(self):
        return len(self._local_set)

    @property
    def master_count(self):
        """Rows in the FTS5 master, or 0 if there is no usable master."""
        conn = self._conn()
        if conn is None:
            return 0
        try:
            return conn.execute("SELECT count(*) FROM norms").fetchone()[0]
        except sqlite3.Error:
            return 0

    # ---- lookup ----------------------------------------------------------
    def lookup(self, norm):
        """Best token_sort_ratio match >= threshold over master + local, or None."""
        # Both tiers are indexed, so each returns at most candidate_limit rows —
        # the rescore set is bounded no matter how long the run goes.
        originals = self._master_candidates(norm) + self._local_candidates(norm)
        sorted_forms = [token_sorted(c) for c in originals]
        self.last_candidate_count = len(originals)
        if not originals:
            return None

        match = _rf_process.extractOne(
            token_sorted(norm), sorted_forms, scorer=_rf_fuzz.ratio, score_cutoff=self.threshold * 100
        )
        return None if match is None else originals[match[2]]
