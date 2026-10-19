# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-rollout telemetry context.

The rollout id is minted by the AGENT and arrives as an HTTP header, because the
harness's own `session_id` lives in a signed cookie the agent cannot read back.
Everything here exists so a `[tavily_call]`/`[bash]`/`[cache]` line printed deep
inside a helper can name the rollout that caused it.
"""

import pytest

from resources_servers.browsecomp_advanced_harness import rollout_context as RC


@pytest.fixture(autouse=True)
def _clean():
    RC.reset_all()
    yield
    RC.reset_all()


class _Req:
    """Minimal stand-in for a Starlette Request."""

    def __init__(self, headers=None, session=None):
        self.headers = headers or {}
        self.session = session if session is not None else {}


# ----------------------------------------------------------------- identity


def test_rollout_id_defaults_to_placeholder_when_never_set():
    assert RC.current_rollout_id() == RC.UNKNOWN_ROLLOUT_ID


def test_reads_the_agent_minted_header():
    assert RC.rollout_id_from_request(_Req({"x-rollout-id": "r42-ab12"})) == "r42-ab12"


def test_header_lookup_is_case_insensitive():
    """Starlette lower-cases header keys; a raw dict in a test may not."""
    assert RC.rollout_id_from_request(_Req({"X-Rollout-Id": "r7"})) == "r7"


def test_falls_back_to_session_id_when_header_absent():
    """Eval, tests and direct curl calls do not send the header; those requests
    must still be attributable rather than collapsing to a shared '-'."""
    req = _Req(headers={}, session={"session_id": "sess-abc"})
    assert RC.rollout_id_from_request(req) == "sess-abc"


def test_falls_back_to_placeholder_when_neither_present():
    assert RC.rollout_id_from_request(_Req()) == RC.UNKNOWN_ROLLOUT_ID


def test_set_rollout_id_is_visible_to_a_deep_helper():
    RC.set_rollout_id("r9")
    assert RC.current_rollout_id() == "r9"


# ------------------------------------------------------------------ totals


def test_record_phase_accumulates_seconds_and_counts():
    RC.set_rollout_id("r1")
    RC.record_phase("tavily_search", 1.5)
    RC.record_phase("tavily_search", 0.5)
    RC.record_phase("bash_exec", 2.0)
    t = RC.peek_phase_totals("r1")
    assert t["tavily_search_s"] == 2.0
    assert t["tavily_search_n"] == 2
    assert t["bash_exec_s"] == 2.0
    assert t["bash_exec_n"] == 1


def test_totals_are_kept_separate_per_rollout():
    RC.record_phase("judge", 1.0, rid="rA")
    RC.record_phase("judge", 4.0, rid="rB")
    assert RC.peek_phase_totals("rA")["judge_s"] == 1.0
    assert RC.peek_phase_totals("rB")["judge_s"] == 4.0


def test_pop_returns_and_clears_so_a_long_leg_does_not_leak():
    """One harness process serves ~944 concurrent rollouts for a 4h leg; totals
    must not accumulate for the process lifetime."""
    RC.record_phase("judge", 1.0, rid="rX")
    assert RC.pop_phase_totals("rX")["judge_s"] == 1.0
    assert RC.pop_phase_totals("rX") == {}
    assert "rX" not in RC._TOTALS


def test_unknown_rollout_is_not_tracked():
    """Without an id there is nothing to attribute to, and accumulating under
    '-' would grow without bound and mix every rollout together."""
    RC.record_phase("judge", 1.0)  # no rid set -> UNKNOWN
    assert RC.peek_phase_totals(RC.UNKNOWN_ROLLOUT_ID) == {}


# ----------------------------------------------------------------- rendering


def test_format_phase_totals_is_stable_and_parseable():
    totals = {"tavily_search_s": 12.34, "tavily_search_n": 3, "bash_exec_s": 1.0, "bash_exec_n": 2}
    s = RC.format_phase_totals(totals)
    assert s == "bash_exec=1.00s/2 tavily_search=12.34s/3"


def test_format_phase_totals_empty():
    assert RC.format_phase_totals({}) == "-"


def test_log_line_includes_rollout_id_and_absolute_time(capsys):
    RC.set_rollout_id("r5")
    RC.log_line("[browsecomp][judge]", t=1700000000.5, dur=2.25, status="success")
    out = capsys.readouterr().out.strip()
    assert out.startswith("[browsecomp][judge] rid=r5 t=1700000000.500")
    assert "dur=2.250" in out
    assert "status=success" in out


def test_log_line_defaults_time_to_now(capsys):
    RC.log_line("[x]")
    assert "t=" in capsys.readouterr().out
