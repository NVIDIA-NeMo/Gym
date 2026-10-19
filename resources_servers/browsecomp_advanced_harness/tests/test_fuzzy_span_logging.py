# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fuzzy lookup must be drawable as a span, like every other phase.

`[browsecomp][provider]`, `[browsecomp][gen]` and `[browsecomp][bash][timing]`
all go through `log_line()`, so each carries an ABSOLUTE `t=`. The fuzzy line
was a bare `print()`: it carried `rid=` and `dur=` but no `t=`, so the gantt
could show fuzzy only as a per-rollout total and never as a span on the
timeline. It also printed on a HIT alone -- the function returns before the
print on a miss and on a stale index entry -- so the two outcomes that cost
time without returning anything were invisible per call.

Pinned here:
  1. every outcome (hit / miss / stale) emits exactly one line
  2. the line carries an absolute epoch `t=`, not a perf_counter value
  3. `t=` is the START of the lookup, so [t, t+dur] is the span
  4. `record_phase("cache_fuzzy", ...)` still fires on every lookup
"""

import re
from time import time
from unittest.mock import MagicMock

import pytest

from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache


LINE = re.compile(r"\[cache\]\[fuzzy\] rid=(?P<rid>\S+) t=(?P<t>[0-9.]+) dur=(?P<dur>[0-9.]+) status=(?P<status>\w+)")


def _cache(tmp_path, matched, read_returns) -> SearchCache:
    c = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
    c._fuzzy_index = MagicMock()
    c._fuzzy_index.lookup = MagicMock(return_value=matched)
    c._read = MagicMock(return_value=read_returns)
    return c


def _one_line(capsys):
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "[cache][fuzzy]" in ln]
    assert len(lines) == 1, f"expected exactly one fuzzy line, got {lines}"
    m = LINE.search(lines[0])
    assert m, f"line does not match the span format: {lines[0]}"
    return m


class TestEveryOutcomeIsLogged:
    def test_hit_is_logged(self, tmp_path, capsys) -> None:
        cache = _cache(tmp_path, matched="widget review", read_returns={"results": []})

        assert cache._fuzzy_lookup("widget reviews") is not None

        assert _one_line(capsys).group("status") == "hit"

    def test_miss_is_logged(self, tmp_path, capsys) -> None:
        # Previously silent: the function returned before reaching the print.
        cache = _cache(tmp_path, matched=None, read_returns=None)

        assert cache._fuzzy_lookup("nothing like this") is None

        assert _one_line(capsys).group("status") == "miss"

    def test_stale_index_entry_is_logged(self, tmp_path, capsys) -> None:
        # The index matched but the entry is gone or corrupt. Also previously
        # silent, and the one outcome that pays the FTS5 cost twice.
        cache = _cache(tmp_path, matched="widget review", read_returns=None)

        assert cache._fuzzy_lookup("widget reviews") is None

        assert _one_line(capsys).group("status") == "stale"


class TestTheLineIsASpan:
    def test_t_is_an_absolute_epoch(self, tmp_path, capsys) -> None:
        cache = _cache(tmp_path, matched=None, read_returns=None)
        before = time()

        cache._fuzzy_lookup("q")

        t = float(_one_line(capsys).group("t"))
        # A perf_counter value is small (seconds since an arbitrary origin); an
        # epoch is ~1.7-1.8e9. This is the check that catches a monotonic leak.
        assert t > 1_700_000_000, f"t={t} is not an absolute epoch"
        assert before - 1 <= t <= time() + 1

    def test_t_is_the_start_not_the_end(self, tmp_path, capsys) -> None:
        # If t were stamped at the END, [t, t+dur] would draw the span in the
        # wrong place -- shifted right by its own duration.
        cache = _cache(tmp_path, matched=None, read_returns=None)
        before = time()
        cache._fuzzy_lookup("q")
        after = time()

        m = _one_line(capsys)
        t, dur = float(m.group("t")), float(m.group("dur"))
        # log_line formats floats as %.3f, so t is rounded to the millisecond and
        # can round UP past `after` on a fast lookup. Compare with a 1ms slack;
        # tightening this to `t <= after` makes the test flake ~2 runs in 3.
        assert t <= after + 0.001
        assert t + dur <= after + 0.5
        assert t >= before - 0.5

    def test_rid_is_carried(self, tmp_path, capsys) -> None:
        cache = _cache(tmp_path, matched=None, read_returns=None)

        cache._fuzzy_lookup("q")

        # No rollout in scope here, so the sentinel; the point is the field exists.
        assert _one_line(capsys).group("rid")


class TestPhaseAccountingUnchanged:
    @pytest.mark.parametrize("matched,read_returns", [("m", {"results": []}), (None, None), ("m", None)])
    def test_record_phase_fires_on_every_outcome(self, tmp_path, capsys, monkeypatch, matched, read_returns) -> None:
        import resources_servers.browsecomp_advanced_harness.search_cache as sc

        seen = []
        monkeypatch.setattr(sc, "record_phase", lambda name, secs: seen.append((name, secs)))
        cache = _cache(tmp_path, matched=matched, read_returns=read_returns)

        cache._fuzzy_lookup("q")

        assert [n for n, _ in seen] == ["cache_fuzzy"]
        assert seen[0][1] >= 0.0
