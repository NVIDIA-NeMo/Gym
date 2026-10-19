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
"""Tests for the FTS5-backed fuzzy index that replaces the brute-force norms scan.

Background. The shipped fuzzy path is a rapidfuzz `extractOne` over an in-memory
list of every cached norm — O(n) per MISS, single-threaded, GIL-held. At
1,863,474 norms that measured a 2.2 s mean scan and ~55,000 CPU-seconds inside a
~6,300 s training leg, which starved the shared thread pool and produced 668
stalls / zero steps in that leg. This module replaces candidate GENERATION
with an FTS5 index while leaving candidate SCORING on rapidfuzz, so match
quality is unchanged and only recall is at risk.

Design under test (index-only swap; the flat-file value store is untouched):

    <cache_root>/index/master.db                     FTS5, read-only+immutable
    <cache_root>/index/search_norms_<host>_<pid>.txt append shard (UNCHANGED)

Two-tier lookup, union-then-rescore:
  1. FTS5 master  -> up to `candidate_limit` candidates (indexed, GIL-released)
  2. process-local norms added since startup -> all of them (bounded, small)
  3. rapidfuzz token_sort_ratio over the union; best >= threshold wins

Invariants pinned here:
- IDENTICAL SEMANTICS to the brute-force scan for any candidate that FTS5
  returns. The threshold, the scorer, and the token-sort insensitivity do not
  change. Only recall (did FTS5 surface it at all) is new, and a recall miss is
  a cache miss, never a wrong answer.
- READ-ONLY master: a lookup must never create -wal/-shm sidecar files, because
  those cannot cross Lustre nodes (cross-node POC: shared-write SQLite crashed
  7/32 workers, write p99 34 s).
- A missing or corrupt master is a clean miss, mirroring the flat-file cache's
  defensive reads.

`tests/test_search_cache.py::TestSearchCacheFuzzy` is deliberately NOT forked: it
is the behavioural contract and must keep passing verbatim against the new
backend. This file adds what that contract cannot see — the storage mechanics,
the recall knob, and brute-force parity.
"""

import sqlite3
import threading
from pathlib import Path

import pytest


pytest.importorskip("rapidfuzz")

from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache  # noqa: E402
from resources_servers.browsecomp_advanced_harness.sqlite_fuzzy_index import (  # noqa: E402
    SqliteFuzzyIndex,
    build_master_db,
    master_db_path,
)


# A corpus with deliberate near-neighbours: `chemistry` differs from `physics`
# by one token and must be REJECTED at the rescore stage even though FTS5
# happily returns it as a candidate.
CORPUS = [
    "who won the 1997 nobel prize in physics",
    "who won the 1997 nobel prize in chemistry",
    "1997 nobel prize physics winner",
    "nobel peace prize 1997 landmines treaty",
    "capital of burkina faso population 2019",
    "countries elected 68th session csw 2024",
    "president of france 1997",
]


@pytest.fixture
def master(tmp_path):
    """A built, read-only-ready master.db over CORPUS."""
    build_master_db(master_db_path(str(tmp_path)), CORPUS)
    return str(tmp_path)


class TestBuildMasterDb:
    def test_build_returns_row_count(self, tmp_path):
        assert build_master_db(master_db_path(str(tmp_path)), CORPUS) == len(CORPUS)

    def test_build_is_idempotent_and_dedupes(self, tmp_path):
        path = master_db_path(str(tmp_path))
        build_master_db(path, CORPUS)
        assert build_master_db(path, CORPUS + CORPUS) == len(CORPUS)

    def test_build_publishes_atomically(self, tmp_path):
        """A half-written DB must never be visible under the live name."""
        path = master_db_path(str(tmp_path))
        build_master_db(path, CORPUS)
        leftovers = [p.name for p in (tmp_path / "index").iterdir() if p.name.startswith("master.db.")]
        assert leftovers == []

    def test_built_db_has_no_wal_sidecars(self, tmp_path):
        """journal_mode=DELETE keeps the master a single portable file."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS)
        names = {p.name for p in (tmp_path / "index").iterdir()}
        assert "master.db-wal" not in names and "master.db-shm" not in names


class TestLookupSemantics:
    def test_word_reorder_hits(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        assert idx.lookup("68th session csw 2024 countries elected") == "countries elected 68th session csw 2024"

    def test_below_threshold_is_miss(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        assert idx.lookup("completely different topic entirely") is None

    def test_near_neighbour_rejected_by_rescore(self, master):
        """FTS5 returns `chemistry` as a candidate; rapidfuzz must reject it."""
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        assert idx.lookup("who won the 1997 nobel prize in chemistry which year") != (
            "who won the 1997 nobel prize in physics"
        )

    def test_best_match_wins_not_first_candidate(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.5)
        assert idx.lookup("1997 physics nobel prize who won it") == "who won the 1997 nobel prize in physics"

    def test_threshold_is_inclusive_boundary(self, master):
        """>= threshold, matching the brute-force scorer's score_cutoff."""
        idx = SqliteFuzzyIndex(master, threshold=1.0)
        assert idx.lookup("who won the 1997 nobel prize in physics") == "who won the 1997 nobel prize in physics"


class TestDefensiveReads:
    def test_missing_master_is_clean_miss(self, tmp_path):
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        assert idx.lookup("anything at all") is None

    def test_corrupt_master_is_clean_miss(self, tmp_path):
        path = master_db_path(str(tmp_path))
        build_master_db(path, CORPUS)
        with open(path, "r+b") as f:
            f.write(b"not a sqlite file at all")
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        assert idx.lookup("who won the 1997 nobel prize in physics") is None

    def test_empty_master_is_clean_miss(self, tmp_path):
        build_master_db(master_db_path(str(tmp_path)), [])
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        assert idx.lookup("who won the 1997 nobel prize in physics") is None

    def test_lookup_creates_no_wal_or_shm(self, master):
        """immutable=1 -> zero locking, zero sidecar files. Load-bearing on Lustre."""
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        idx.lookup("1997 physics nobel prize who won it")
        names = {p.name for p in Path(master, "index").iterdir()}
        assert "master.db-wal" not in names and "master.db-shm" not in names

    def test_master_is_opened_read_only(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        idx.lookup("anything")
        with pytest.raises(sqlite3.OperationalError):
            idx._conn().execute("INSERT INTO norms(norm) VALUES ('sneaky write')")


class TestProcessLocalAdditions:
    """Norms cached during the run are matchable without rebuilding the master."""

    def test_added_norm_is_matchable(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        idx.add("tallest building in shanghai completed 2015")
        assert idx.lookup("shanghai tallest building completed 2015") == "tallest building in shanghai completed 2015"

    def test_add_is_deduped(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        idx.add("some brand new query")
        idx.add("some brand new query")
        assert idx.local_count == 1

    def test_master_match_preferred_when_better(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.5)
        idx.add("nobel prize 1997 physics who won")
        # The exact-ish master row scores higher than the local paraphrase.
        assert idx.lookup("who won the 1997 nobel prize in physics") == "who won the 1997 nobel prize in physics"


class TestLocalTierScaling:
    """The process-local tier must be INDEXED, not rescanned.

    The master is an immutable snapshot, so norms cached during a run land in a
    process-local tier. The first implementation kept that tier as a plain list
    and rescored all of it on every lookup — the same O(n) shape as the
    brute-force scan this module replaced. A production run showed the consequence in
    production: search-get latency 17ms -> 92ms and slow-get rate 0.19% -> 2.03%
    within 1.3h as the process cached ~50k norms.

    The offline benchmarks missed it because they built a fresh index and never
    grew `local`. These tests close that gap.

    They assert on `last_candidate_count`, not wall time: the rescore-set size is
    the defect itself and is deterministic, whereas timing is noisy under load.
    """

    LOCAL_N = 20_000

    def test_rescore_set_stays_bounded_as_local_grows(self, tmp_path):
        """The regression test. Candidates must not track the local tier size."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8, candidate_limit=50)

        idx.lookup("who won the 1997 nobel prize in physics")
        cold = idx.last_candidate_count

        for i in range(self.LOCAL_N):
            idx.add(f"cached during this run {i} about assorted topics")
        idx.lookup("who won the 1997 nobel prize in physics")

        assert idx.local_count == self.LOCAL_N
        # Master candidates + local candidates, each capped by candidate_limit.
        assert idx.last_candidate_count <= 2 * 50, (
            f"rescore set grew to {idx.last_candidate_count} with {self.LOCAL_N} local norms "
            f"(was {cold} cold) — the local tier is being rescanned linearly"
        )

    def test_local_norm_still_found_among_many(self, tmp_path):
        """Indexing the local tier must not cost local recall."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8, candidate_limit=50)

        needle = "tallest building in shanghai completed 2015"
        idx.add(needle)
        for i in range(self.LOCAL_N):
            idx.add(f"unrelated filler query {i} concerning other matters")

        assert idx.lookup("shanghai tallest building completed 2015") == needle

    def test_master_and_local_still_compete(self, tmp_path):
        """Both tiers must reach the rescore stage; best score wins regardless."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.5, candidate_limit=50)
        idx.add("nobel prize 1997 physics who won")
        for i in range(1000):
            idx.add(f"noise entry {i} about unrelated subjects")
        assert idx.lookup("who won the 1997 nobel prize in physics") == "who won the 1997 nobel prize in physics"

    def test_concurrent_add_and_lookup_on_local_index(self, tmp_path):
        """The local index is shared mutable state; adds and reads must not race."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
        errors = []

        def adder(base):
            try:
                for i in range(200):
                    idx.add(f"concurrent norm {base}-{i} about various things")
            except Exception as e:  # pragma: no cover - surfaced via errors
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    idx.lookup("concurrent norm 3-17 about various things")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=adder, args=(b,)) for b in range(8)]
        threads += [threading.Thread(target=reader) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert idx.local_count == 8 * 200


class TestRecallKnob:
    def test_candidate_limit_bounds_the_rescore_set(self, tmp_path):
        norms = [f"query number {i} about nobel prize physics" for i in range(500)]
        build_master_db(master_db_path(str(tmp_path)), norms)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8, candidate_limit=10)
        idx.lookup("query number 250 about nobel prize physics")
        assert idx.last_candidate_count <= 10

    @pytest.mark.parametrize("tokenizer", ["unicode61", "trigram"])
    def test_both_tokenizers_work(self, tmp_path, tokenizer):
        """unicode61 is the measured default; trigram stays supported for A/B."""
        build_master_db(master_db_path(str(tmp_path)), CORPUS, tokenizer=tokenizer)
        idx = SqliteFuzzyIndex(str(tmp_path), threshold=0.8, tokenizer=tokenizer)
        assert idx.lookup("68th session csw 2024 countries elected") == "countries elected 68th session csw 2024"


class TestThreadSafety:
    def test_concurrent_lookups_are_correct(self, master):
        """32 threads share the index; each gets a thread-local connection."""
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        results, errors = [], []

        def worker():
            try:
                results.append(idx.lookup("68th session csw 2024 countries elected"))
            except Exception as e:  # pragma: no cover - failure surfaces via errors
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert results == ["countries elected 68th session csw 2024"] * 32

    def test_concurrent_add_and_lookup(self, master):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        errors = []

        def adder(i):
            try:
                idx.add(f"newly cached query {i} about physics")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        def reader():
            try:
                idx.lookup("newly cached query 5 about physics")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=adder, args=(i,)) for i in range(16)]
        threads += [threading.Thread(target=reader) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestBruteForceParity:
    """The gate: FTS5+rescore must decide the same as the O(n) scan it replaces.

    Recall loss shows up here as a disagreement. Precision loss cannot occur by
    construction (same scorer, same threshold), so any disagreement is FTS5
    failing to surface a candidate -> tune `candidate_limit`/tokenizer.
    """

    @staticmethod
    def _brute(norms, query, threshold):
        from rapidfuzz import fuzz, process

        sorted_norms = [" ".join(sorted(n.split())) for n in norms]
        match = process.extractOne(
            " ".join(sorted(query.split())), sorted_norms, scorer=fuzz.ratio, score_cutoff=threshold * 100
        )
        return None if match is None else norms[match[2]]

    @pytest.mark.parametrize(
        "query",
        [
            "68th session csw 2024 countries elected",
            "1997 physics nobel prize who won it",
            "burkina faso capital population 2019",
            "who won the 1997 nobel prize in chemistry",
            "an entirely unrelated question about penguins",
            "president france 1997",
        ],
    )
    def test_agrees_with_brute_force(self, master, query):
        idx = SqliteFuzzyIndex(master, threshold=0.8)
        assert idx.lookup(query) == self._brute(CORPUS, query, 0.8)


class TestSearchCacheIntegration:
    """The public surface must not change: same ctor, same stats, same semantics."""

    RESULTS = [{"title": "T", "url": "https://x.com", "raw_content": "BODY"}]

    def test_fuzzy_hit_through_search_cache(self, tmp_path):
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        assert cache.get_search("68th session CSW 2024 countries elected") == ("tavily", self.RESULTS)
        assert cache.stats()["search_fuzzy_hits"] == 1

    def test_no_master_db_still_serves_own_puts(self, tmp_path):
        """A cold cache root has no master.db; within-run fuzzy must still work."""
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        assert not (tmp_path / "index" / "master.db").exists()
        cache.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        assert cache.get_search("68th session CSW 2024 countries elected") == ("tavily", self.RESULTS)

    def test_build_master_from_text_index(self, tmp_path):
        """Rebuild the master from the .txt shards, with no JSON rescan.

        This is how a cache root that predates master.db gets one: the shards
        are the durable record, so the master can be regenerated from them in
        seconds instead of walking 430k entry files.
        """
        from resources_servers.browsecomp_advanced_harness.build_cache_index import build_master_from_text_index

        writer = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        writer.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        (tmp_path / "index" / "master.db").unlink(missing_ok=True)

        assert build_master_from_text_index(str(tmp_path)) == 1
        assert (tmp_path / "index" / "master.db").exists()

        reader = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        assert reader.get_search("68th session CSW 2024 countries elected") == ("tavily", self.RESULTS)

    def test_exact_path_untouched_when_fuzzy_disabled(self, tmp_path):
        """No index dir, no DB, no rapidfuzz import cost when fuzzy is off."""
        cache = SearchCache(str(tmp_path))
        cache.put_search("q", self.RESULTS, provider="tavily")
        assert cache.get_search("q") == ("tavily", self.RESULTS)
        assert not (tmp_path / "index").exists()

    def test_stale_index_entry_is_a_miss(self, tmp_path):
        """A norm in the master whose JSON entry was deleted must not crash."""
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        build_master_db(master_db_path(str(tmp_path)), ["countries elected 68th session csw 2024"])
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        assert cache.get_search("68th session CSW 2024 countries elected") is None
