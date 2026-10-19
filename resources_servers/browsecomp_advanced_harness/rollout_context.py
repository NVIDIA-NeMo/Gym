# SPDX-License-Identifier: Apache-2.0
"""Per-rollout telemetry context for the browsecomp harness.

WHY THIS EXISTS. Every tool line the harness printed used to be unattributable:
`[tavily_call]`, `[judge_call]`, `[browsecomp][bash][timing]` and `[cache]` carry
no rollout key, and ONE NemoGym process serves the whole run with up to ~944
rollouts in flight. So "how long did THIS rollout spend in search vs bash?" could
not be answered from a driver log at all — only leg-wide totals could.

The rollout id is minted by the AGENT and arrives as the `X-Rollout-Id` header.
It is not derived from the harness's own `session_id` because that lives in a
signed session cookie which the agent cannot read back, so the agent could not
tag its own generation/tokenize lines with the same key. Requests without the
header (eval, tests, direct curl) fall back to the session id, then to "-".

A ContextVar rather than a parameter: the print sites are several frames deep
inside helpers and inside `search_cache.py`, which has no request at all. This
mirrors the existing `_PROVIDER_RETRY_COUNTS` ContextVar in `app.py`, which
solves the same attribution problem for provider retry counts.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from time import time
from typing import Any, Dict, Optional


ROLLOUT_ID_HEADER = "X-Rollout-Id"
UNKNOWN_ROLLOUT_ID = "-"

_ROLLOUT_ID: ContextVar[str] = ContextVar("_ROLLOUT_ID", default=UNKNOWN_ROLLOUT_ID)

# rid -> {"<phase>_s": seconds, "<phase>_n": count}. Emitted once per rollout on
# the `[browsecomp][phases]` line at /verify, then popped — see pop_phase_totals.
_TOTALS: Dict[str, Dict[str, float]] = {}
_LOCK = threading.Lock()


def set_rollout_id(rid: str):
    """Bind the rollout id for the current asyncio task. Returns the reset token."""
    return _ROLLOUT_ID.set(rid or UNKNOWN_ROLLOUT_ID)


def current_rollout_id() -> str:
    return _ROLLOUT_ID.get()


def rollout_id_from_request(request: Any) -> str:
    """Header first, session id second, placeholder last."""
    headers = getattr(request, "headers", None) or {}
    rid = None
    getter = getattr(headers, "get", None)
    if getter is not None:
        # Starlette lower-cases header keys; a plain dict in a test may not.
        rid = getter(ROLLOUT_ID_HEADER.lower()) or getter(ROLLOUT_ID_HEADER)
    if rid:
        return rid
    session = getattr(request, "session", None) or {}
    try:
        rid = session.get("session_id")
    except AttributeError:  # pragma: no cover - defensive
        rid = None
    return rid or UNKNOWN_ROLLOUT_ID


def record_phase(phase: str, dur_s: float, *, rid: Optional[str] = None) -> None:
    """Accumulate one phase sample against a rollout.

    Always called, even when per-call logging is off — the totals are the cheap,
    always-on product (one summary line per rollout) and the per-call lines are
    the expensive optional detail.

    `rid` is KEYWORD-ONLY on purpose: the agent's twin helper takes rid FIRST
    (it has no ContextVar to default to), and calling this one in that order
    silently passed a phase name where a duration belonged.
    """
    rid = rid or current_rollout_id()
    if rid == UNKNOWN_ROLLOUT_ID:
        # Nothing to attribute to. Accumulating under "-" would grow unbounded
        # for the process lifetime and blend every rollout together.
        return
    with _LOCK:
        t = _TOTALS.setdefault(rid, {})
        t[f"{phase}_s"] = t.get(f"{phase}_s", 0.0) + float(dur_s)
        t[f"{phase}_n"] = t.get(f"{phase}_n", 0) + 1


def peek_phase_totals(rid: str) -> Dict[str, float]:
    with _LOCK:
        return dict(_TOTALS.get(rid, {}))


def pop_phase_totals(rid: str) -> Dict[str, float]:
    """Read and drop. Called at /verify, which is the end of a rollout."""
    with _LOCK:
        return _TOTALS.pop(rid, {})


def format_phase_totals(totals: Dict[str, float]) -> str:
    """`bash_exec=1.00s/2 tavily_search=12.34s/3` — sorted for stable diffing."""
    phases = sorted({k[:-2] for k in totals if k.endswith("_s")})
    if not phases:
        return "-"
    return " ".join(f"{p}={totals.get(f'{p}_s', 0.0):.2f}s/{int(totals.get(f'{p}_n', 0))}" for p in phases)


def log_line(tag: str, t: Optional[float] = None, **kv: Any) -> None:
    """Print one telemetry line carrying the rollout id and an ABSOLUTE timestamp.

    The timestamp matters as much as the id: harness prints have never had one,
    so reconstructing a timeline meant interpolating between the sparse `ts=`
    anchors nemo-rl timers emit — and there is no anchor at all during the first
    rollout wave (~65 min).
    """
    parts = [tag, f"rid={current_rollout_id()}", f"t={(time() if t is None else t):.3f}"]
    for k, v in kv.items():
        if v is None:
            continue
        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    print(" ".join(parts), flush=True)


def reset_all() -> None:
    """Test hook: drop all accumulated state."""
    with _LOCK:
        _TOTALS.clear()
    _ROLLOUT_ID.set(UNKNOWN_ROLLOUT_ID)
