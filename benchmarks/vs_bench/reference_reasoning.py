# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke unmodified VS-Bench action extraction in its own Python environment.

External source: https://github.com/zelaix/VS-Bench, MIT license, revision
2c3531cdf91d1b620b4a9015126a242eb305a4d0, ``agents/prompt_agent.py`` and
``envs/<game>.py``. No third-party source is copied into this adapter.

Only reasoning action parsing is supported here. In particular, this is not a
safe execution boundary for the original perception scorers, which use eval().
"""

import importlib
import json
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path


GAMES = frozenset(
    (
        "hanabi",
        "overcooked",
        "knights_archers_zombies",
        "breakthrough",
        "kuhn_poker",
        "atari_pong",
        "simple_push",
        "coin_dilemma",
        "monster_hunt",
        "battle_of_the_colors",
    )
)


def reference_score(author_root: Path, request: dict) -> dict:
    game = request["game"]
    if game not in GAMES:
        raise ValueError("Unknown VS-Bench game")
    # Upstream's eager package initializers require unrelated optional providers
    # and Atari dependencies. Load the requested author modules without editing them.
    for name in ("agents", "envs", "models"):
        package = types.ModuleType(name)
        package.__path__ = [str(author_root / name)]
        sys.modules[name] = package
    sys.path.insert(0, str(author_root))
    from agents.prompt_agent import PromptAgent
    from envs.base_env import BaseEnv, RegexError

    importlib.import_module("envs." + game)
    env = object.__new__(BaseEnv._registry[game])
    agent = object.__new__(PromptAgent)
    try:
        value = agent._extract_action(request["answer"], env.regex_patterns)
        action = int(agent._parse_action(value, request["legal_actions"]))
    except RegexError:
        return {"valid": False, "outcome": "retryable_format_error"}
    except ValueError:
        # Author's predict() breaks immediately and returns this sentinel.
        return {"valid": False, "outcome": "illegal_action", "action": -99999999, "correct": False}
    return {"valid": True, "outcome": "parsed", "action": action, "correct": action in request["next_action"]}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: reference_reasoning.py <upstream-root>")
    with redirect_stdout(sys.stderr):
        result = reference_score(Path(sys.argv[1]).resolve(), json.load(sys.stdin))
    print(json.dumps(result, allow_nan=False))
