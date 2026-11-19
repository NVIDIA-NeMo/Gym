# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The oracle distance: how many actions a level still needs from a game state.

The games of a pivot pool are written as pure functions over a state ``(h, z)`` (``get_initial_state``,
``transition``, ``predict``, ``check_level_complete``, optionally ``is_game_over``) plus ``policy(h, z, level)``: the
next action of a shortest completion of ``level`` (``RESET`` when restarting is shorter). The distance rolls the policy
on a copy of the state until the level completes: every action counts 1, and a game-over state takes only RESET. It is
None past ``cap`` actions or when the policy revisits a state. Standard library only (the worker imports it).
"""

import copy
import hashlib
import inspect
import pickle
import threading
import types
from typing import Any


DEFAULT_CAP = 200
PREDICT_SEED = 0
RESET = 0

# Some games fill module-level policy caches lazily: one lock per game module.
_MODULE_LOCKS: dict = {}
_TABLE_LOCK = threading.Lock()


def _module_lock(namespace: dict) -> threading.RLock:
    with _TABLE_LOCK:
        return _MODULE_LOCKS.setdefault(id(namespace), (namespace, threading.RLock()))[1]


def oracle_for(game: Any) -> types.SimpleNamespace:
    """The oracle over an engine game's own functions (``game.FUNCTIONS`` and its module's ``policy``)."""
    functions = dict(getattr(game, "FUNCTIONS", {}) or {})
    module = getattr(functions.get("transition"), "__globals__", None) or {}
    policy = module.get("policy")
    needed = ("transition", "predict", "check_level_complete", "get_initial_state")
    missing = [k for k in needed if not callable(functions.get(k))] + ([] if callable(policy) else ["policy"])
    if missing:
        raise LookupError(f"game {type(game).__name__} has no {missing}: it has no oracle")
    predict = functions["predict"]
    seeded = len(inspect.signature(predict).parameters) >= 2
    is_over = functions.get("is_game_over") or module.get("is_game_over")
    lock = _module_lock(module)

    def locked_policy(h: Any, z: Any, level: int) -> Any:
        with lock:
            return policy(h, z, level)

    return types.SimpleNamespace(
        transition=functions["transition"],
        predict=(lambda h, seed: predict(h, seed)) if seeded else (lambda h, seed: predict(h)),
        check_level_complete=functions["check_level_complete"],
        is_game_over=is_over if callable(is_over) else (lambda h, z: False),
        get_initial_state=functions["get_initial_state"],
        policy=locked_policy,
        reset_action=module.get("RESET_ACTION", RESET),
    )


def _is_reset(oracle: Any, action: Any) -> bool:
    try:
        return bool(action == oracle.reset_action)
    except Exception:  # e.g. a list action compared to an int
        return False


def _state_key(h: Any, z: Any) -> bytes | None:
    try:
        return hashlib.sha256(pickle.dumps((h, z), protocol=4)).digest()
    except Exception:  # an unpicklable state: the cap alone ends a loop
        return None


def distance_to_go(oracle: Any, h: Any, z: Any, level: int, *, cap: int = DEFAULT_CAP) -> int | None:
    """Actions the oracle policy needs from ``(h, z)`` to complete ``level`` (None past ``cap`` or on a loop)."""
    h, z = copy.deepcopy(h), copy.deepcopy(z)
    seen = set()
    steps = 0
    while True:
        if oracle.check_level_complete(h, z, level):
            return steps
        if steps >= cap:
            return None
        key = _state_key(h, z)
        if key is not None:
            if key in seen:
                return None
            seen.add(key)
        action = oracle.reset_action if oracle.is_game_over(h, z) else oracle.policy(h, z, level)
        if _is_reset(oracle, action):
            h, z = copy.deepcopy(oracle.get_initial_state(level))
        else:  # a click policy returns [6, x, y]; transition takes the tuple
            h = oracle.transition(h, z, tuple(action) if isinstance(action, list) else action)
            z = oracle.predict(h, PREDICT_SEED)
        steps += 1


def distance(game: Any, level: int, *, cap: int = DEFAULT_CAP) -> int | None:
    """The distance of an engine game in its current state (``h_t``, ``z_t``) to completing ``level``."""
    oracle = oracle_for(game)
    return distance_to_go(oracle, game.h_t, game.z_t, level, cap=cap)
