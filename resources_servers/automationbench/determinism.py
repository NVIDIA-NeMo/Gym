# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic tool execution for the automationbench server (`deterministic: true`).

The simulator draws record ids from ``uuid4`` and ``random`` and stamps records with the wall clock (often through
``Field(default_factory=datetime.now)``, bound at import). Replaying a captured trajectory then mints different ids,
and every later call that references a record the trajectory created diverges. Under this module each tool call of an
episode runs with the episode's own ``random`` state, ``uuid4`` drawn from that state, and the clock frozen at
``clock_base + calls`` seconds. Building the episode's world is call 0, so a replay that starts from the same record
reproduces the trajectory's ids, timestamps and tool text.

A record is ``{"seed": <hex>, "clock_base": <ISO time>, "calls": <int>}``. Prefix pass@K rows carry the record the
captured trajectory started from; other rows get one seeded from their ``task_id``.

Tools are synchronous and the server runs them under the session lock, so no other call runs inside the window. The
clock is frozen at the C level (time-machine), which also covers default factories bound at import.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import random
import sys
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


RNG_STATE_KEY = "_determinism_rng_state"
_ORIGINAL_UUID4 = uuid.uuid4
_inside_deterministic_call = False


def _uuid4() -> uuid.UUID:
    if _inside_deterministic_call:
        return uuid.UUID(int=random.getrandbits(128), version=4)
    return _ORIGINAL_UUID4()


def _install_uuid4_hook() -> None:
    """Route uuid4 through `_uuid4`, including names bound by `from uuid import uuid4`."""
    uuid.uuid4 = _uuid4
    for name, module in list(sys.modules.items()):
        if name.startswith("automationbench") and getattr(module, "uuid4", None) is _ORIGINAL_UUID4:
            module.uuid4 = _uuid4


def new_record(seed_material: str, clock_base: Optional[datetime] = None) -> Dict[str, Any]:
    """A fresh record: the seed hashed from `seed_material`, the clock at `clock_base` (now) to the second."""
    base = (clock_base or datetime.now(timezone.utc)).replace(microsecond=0)
    return {"seed": hashlib.sha256(seed_material.encode()).hexdigest(), "clock_base": base.isoformat(), "calls": 0}


@contextlib.contextmanager
def deterministic_call(record: Dict[str, Any], rng_holder: Dict[str, Any]) -> Iterator[None]:
    """Run one call under the episode's RNG and frozen clock, then advance `record["calls"]`.

    `rng_holder` carries the episode's `random` state between calls and is seeded from `record["seed"]` on first
    use, so a replay needs only a fresh copy of the starting record.
    """
    import time_machine

    global _inside_deterministic_call
    _install_uuid4_hook()
    rng_state = rng_holder.get(RNG_STATE_KEY)
    if rng_state is None:
        rng_state = random.Random(int(record["seed"], 16)).getstate()
    saved = random.getstate()
    random.setstate(rng_state)
    moment = datetime.fromisoformat(record["clock_base"]) + timedelta(seconds=record["calls"])
    _inside_deterministic_call = True
    try:
        with time_machine.travel(moment, tick=False):
            yield
    finally:
        _inside_deterministic_call = False
        rng_holder[RNG_STATE_KEY] = random.getstate()
        random.setstate(saved)
        record["calls"] += 1


def deterministic_world(initial_state: Dict[str, Any], record: Dict[str, Any], rng_holder: Dict[str, Any]):
    """Build the task's WorldState under the record (call 0 of the episode).

    Records the task leaves without ids or timestamps get them from default factories while the world is built.
    """
    from automationbench.schema.world import WorldState

    with deterministic_call(record, rng_holder):
        return WorldState(**copy.deepcopy(initial_state))
