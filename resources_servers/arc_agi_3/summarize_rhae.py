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
"""RHAE table of ARC-AGI-3 runs over an expected game set: overall and per game, at the end and at fixed hours.

    python resources_servers/arc_agi_3/summarize_rhae.py --expected resources_servers/arc_agi_3/data/public25.jsonl \\
        <run> [<run> ...] --hours 1 2 [--json summary.json]

A ``<run>`` is a run directory, a ``benchmark.json`` or a Gym rollouts JSONL:

- run directory: scores come from its ``benchmark.json`` (``.`` or ``runs/*/``), on the run clock (the kit's: a
  level counts at ``T`` if it was cleared before the run start + ``T``). Its ``rollouts.jsonl`` and Gym's
  ``rollouts_failures.jsonl``, when present, add the verifier's flags (agent and persistence errors; a game whose
  scorecard disagrees with its agent fails its rollout and is counted as ``rollout_failed``).
- ``benchmark.json``: the same, without the verifier's flags.
- rollouts JSONL: the verifier's scores. Rows without a run clock offset (``clock_offset_seconds``: older runs timed
  per game, or a verify request without the run's origin) give no timed score.

A game has a value in a column only if it is a valid result: present, not cancelled, timing valid (for timed
columns), and not flagged by the verifier. RHAE is the mean over the expected games and is printed only when every
expected game has a value; otherwise the column reads ``INCOMPLETE n/N``. Nothing is zero-filled.

A value is final when it can no longer change: the game's attempt has ended, or (for ``@T``) the snapshot was saved
at or after run start + T. A run directory with a ``rollouts.jsonl`` also needs the verifier's row for each game. A
column whose values are all present but not all final reads ``<value> not final``: the score so far of a live or
aborted run, not the run's score. ``complete`` is true only when every column is final.
"""

import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


try:
    from resources_servers.arc_agi_3.rhae import TimingError, clock_offset_seconds, rhae_at, validate_timing
except ImportError:  # run as a script from this directory
    from rhae import TimingError, clock_offset_seconds, rhae_at, validate_timing


# Flags that withhold a value only from the timed columns or only make it not final (the others invalidate the game).
NO_RUN_CLOCK = "no run clock"
NOT_VERIFIED = "not_verified"
SOFT_FLAGS = ("timing_invalid", NO_RUN_CLOCK, NOT_VERIFIED)
# Attempt states of a game that is still being played (its record can still change).
LIVE_STATES = (None, "", "playing")


def _hour_key(hours: float) -> str:
    return f"rhae_{hours:g}h"


def _is_run_clock_offset(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _observed_seconds(start: str | None, saved_at: str | None) -> float | None:
    """Seconds of the run clock that a snapshot covers (``saved_at - start_time``); None when unknown."""
    if not start or not saved_at:
        return None
    try:
        return clock_offset_seconds(start, saved_at)
    except (TimingError, ValueError):
        return None


def game_key(meta: dict[str, Any]) -> str | None:
    """A game's key in the summary: its full id (``ls20-9607627b``); several versions of a game can be expected."""
    return meta.get("full_game_id") or meta.get("game_id")


def load_expected(path: Path) -> list[str]:
    games = [game_key(json.loads(line)["verifier_metadata"]) for line in path.read_text().splitlines() if line.strip()]
    if len(set(games)) != len(games):
        raise ValueError(f"{path}: duplicate game ids")
    return games


def _new_entry() -> dict[str, Any]:
    return {"scores": {}, "flags": [], "final": {}}


def scores_from_benchmark(path: Path, hours: list[float]) -> dict[str, dict[str, Any]]:
    bench = json.loads(path.read_text())
    start = bench.get("start_time")
    observed = _observed_seconds(start, bench.get("saved_at"))
    games: dict[str, dict[str, Any]] = {}
    for run in bench["game_runs"]:
        game = run["game_id"]
        if game in games:
            raise ValueError(f"{path}: game {game} appears twice; summarize one attempt per game")
        entry = games[game] = _new_entry()
        apl = run.get("actions_per_level") or []
        args = (
            apl,
            run.get("base_actions_per_level"),
            run.get("number_of_levels") or 0,
            run.get("levels_completed") or 0,
        )
        if run.get("state") == "cancelled":
            entry["flags"].append("cancelled")
        ended = run.get("state") not in LIVE_STATES
        entry["final"]["end"] = ended
        for h in hours:
            entry["final"][_hour_key(h)] = ended or (observed is not None and observed >= h * 3600.0)
        entry["scores"]["end"] = rhae_at(*args, [])
        try:
            times = validate_timing(run.get("history") or [], apl)
            if not start or not run.get("started_at"):
                raise TimingError("no run start_time / game started_at")
            offset = clock_offset_seconds(start, run["started_at"])
            for h in hours:
                entry["scores"][_hour_key(h)] = rhae_at(*args, times, t_seconds=h * 3600.0, offset_seconds=offset)
        except TimingError as exc:
            entry["flags"].append(f"timing_invalid: {exc}")
    return games


def _row_flags(row: dict[str, Any]) -> list[str]:
    flags = []
    # The verifier's model-error mask: the game was played without the model, so its score does not measure it.
    if row.get("model_error"):
        flags.append(f"model_error: {str(row['model_error'])[:120]}")
    if row.get("mask_sample"):
        flags.append(f"masked: {row.get('failure_kind') or 'unknown'}")
    if row.get("agent_error"):
        flags.append(f"agent_error: {str(row['agent_error'])[:120]}")
    if row.get("persistence_error"):
        flags.append(f"persistence_error: {str(row['persistence_error'])[:120]}")
    state = row.get("agent_state") or row.get("state")
    if state in ("not_started", "cancelled"):
        flags.append(state)
    if row.get("timing_valid") is False:
        flags.append(f"timing_invalid: {row.get('timing_error')}")
    return flags


def scores_from_rollouts(path: Path, hours: list[float]) -> dict[str, dict[str, Any]]:
    games: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        game = (row.get("verifier_metadata") or {}).get("full_game_id") or row.get("game_id")
        if game in games:
            raise ValueError(f"{path}: game {game} appears twice; summarize one rollout per game")
        entry = games[game] = _new_entry()
        entry["flags"] = _row_flags(row)
        entry["scores"]["end"] = row.get("rhae")
        entry["final"] = {"end": True, **{_hour_key(h): True for h in hours}}  # a verified row is an ended attempt
        if _is_run_clock_offset(row.get("clock_offset_seconds")):
            for h in hours:
                entry["scores"][_hour_key(h)] = row.get(_hour_key(h))
        else:
            entry["flags"].append(
                f"{NO_RUN_CLOCK}: no clock_offset_seconds (older run or no run origin); use the run's benchmark.json"
            )
    return games


def failed_rollout_games(path: Path) -> list[str]:
    """Games of Gym's ``<stem>_failures.jsonl`` (rollouts that raised before a verify response)."""
    games = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            meta = row.get("verifier_metadata") or (row.get("input") or {}).get("verifier_metadata") or {}
            games.append(game_key(meta) or "?")
    return games


def _find_benchmark(run_dir: Path) -> Path | None:
    found = [p for p in (run_dir / "benchmark.json", *sorted(run_dir.glob("runs/*/benchmark.json"))) if p.is_file()]
    if len(found) > 1:
        raise ValueError(f"{run_dir}: several benchmark.json files {found}; pass one")
    return found[0] if found else None


def _is_benchmark_json(path: Path) -> bool:
    """A taaf benchmark.json (one JSON object with ``game_runs``), not a rollouts JSONL."""
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return False
    return isinstance(data, dict) and "game_runs" in data


def load_run(path: Path, hours: list[float]) -> dict[str, dict[str, Any]]:
    if path.is_dir():
        bench = _find_benchmark(path)
        rollouts = path / "rollouts.jsonl"
        if bench is None and not rollouts.is_file():
            raise FileNotFoundError(
                f"{path}: no benchmark.json or rollouts.jsonl. An official-harness run directory has neither (its "
                "scores are the verifier's rows): pass the evaluation's rollouts JSONL instead"
            )
        games = scores_from_benchmark(bench, hours) if bench else scores_from_rollouts(rollouts, hours)
        failures = path / "rollouts_failures.jsonl"
        if bench and rollouts.is_file():  # the verifier's flags on top of the run-clock scores
            verified = scores_from_rollouts(rollouts, hours)
            failed = set(failed_rollout_games(failures)) if failures.is_file() else set()
            for game, entry in verified.items():
                flags = [f for f in entry["flags"] if not f.startswith(NO_RUN_CLOCK)]
                games.setdefault(game, _new_entry())["flags"] += flags
            for game, entry in games.items():
                if game not in verified and game not in failed:  # no verifier row yet: live, or the run was killed
                    entry["flags"].append(NOT_VERIFIED)
                    entry["final"] = {column: False for column in entry["final"]}
    elif _is_benchmark_json(path):
        games, failures = scores_from_benchmark(path, hours), None
    else:
        games = scores_from_rollouts(path, hours)
        failures = path.with_name(path.stem + "_failures.jsonl")
    if failures is not None and failures.is_file():
        for game in failed_rollout_games(failures):
            games.setdefault(game, _new_entry())["flags"].append("rollout_failed")
    return games


def summarize(name: str, games: dict[str, dict[str, Any]], expected: list[str], hours: list[float]) -> dict[str, Any]:
    columns = ["end"] + [_hour_key(h) for h in hours]
    counts: dict[str, int] = {}
    per_game: dict[str, dict[str, Any]] = {}
    for game in expected:
        entry = games.get(game)
        if entry is None:
            per_game[game] = {"status": "missing", "flags": ["missing"], **{c: None for c in columns}}
            counts["missing"] = counts.get("missing", 0) + 1
            continue
        flags = entry["flags"]
        for flag in flags:
            key = flag.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
        hard = [f for f in flags if not f.startswith(SOFT_FLAGS)]
        values = {}
        for column in columns:
            value = entry["scores"].get(column)
            values[column] = None if hard else value
        final = {column: bool((entry.get("final") or {}).get(column)) for column in columns}  # unknown = not final
        per_game[game] = {"status": "invalid" if hard else "valid", "flags": flags, "final": final, **values}
    unexpected = sorted(set(games) - set(expected))
    overall: dict[str, Any] = {}
    coverage: dict[str, int] = {}
    column_status: dict[str, str] = {}
    for column in columns:
        values = [per_game[g][column] for g in expected if per_game[g][column] is not None]
        coverage[column] = len(values)
        present = len(values) == len(expected) and not unexpected
        overall[column] = statistics.mean(values) if present else None
        if not present:
            column_status[column] = "incomplete"
        else:
            column_status[column] = "final" if all(per_game[g]["final"][column] for g in expected) else "not_final"
    return {
        "run": name,
        "expected_games": len(expected),
        "present_games": len(set(games) & set(expected)),
        "complete": all(status == "final" for status in column_status.values()),
        "column_status": column_status,
        "coverage": coverage,
        "overall": overall,
        "counts": counts,
        "unexpected_games": unexpected,
        "per_game": per_game,
    }


def print_table(summaries: list[dict[str, Any]], hours: list[float]) -> None:
    columns = ["end"] + [_hour_key(h) for h in hours]
    labels = ["end"] + [f"@{h:g}h" for h in hours]
    print("RHAE (0-100), mean over the expected games; INCOMPLETE n/N = only n of N games have a valid value;")
    print("not final = every game has a value, but some were still playing at the snapshot or have no verifier row")
    print("(the score so far of a live or aborted run, not the run's score)")
    print("run".ljust(40), *[label.rjust(18) for label in labels], " problems")
    for s in summaries:
        cells = []
        for column in columns:
            value, covered, status = s["overall"][column], s["coverage"][column], s["column_status"][column]
            if status == "final":
                cells.append(f"{value:18.2f}")
            elif status == "not_final":
                cells.append(f"{value:.2f} not final".rjust(18))
            else:
                cells.append(f"INCOMPLETE {covered}/{s['expected_games']}".rjust(18))
        problems = ", ".join(f"{k} {v}" for k, v in sorted(s["counts"].items())) or "-"
        if s["unexpected_games"]:
            problems += f", unexpected games {s['unexpected_games']}"
        print(s["run"][-40:].ljust(40), *cells, " " + problems)
    games = list(summaries[0]["per_game"]) if summaries else []
    for column, label in zip(columns, labels):
        print(f"\nper game {label} (* = not final)")
        print("game".ljust(6), *[s["run"][-14:].rjust(14) for s in summaries])
        for game in games:
            cells = []
            for s in summaries:
                entry = s["per_game"][game]
                value = entry[column]
                if value is None:
                    cells.append(entry["status"].rjust(14))
                else:
                    cells.append((f"{value:.2f}" + ("" if entry["final"][column] else "*")).rjust(14))
            print(game.ljust(6), *cells)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directory, benchmark.json or rollouts JSONL")
    parser.add_argument("--expected", type=Path, required=True, help="the dataset JSONL that defines the game set")
    parser.add_argument("--hours", nargs="*", type=float, default=[1.0, 2.0])
    parser.add_argument("--json", type=Path, help="also write the summaries as JSON")
    args = parser.parse_args(argv)
    expected = load_expected(args.expected)
    summaries = [summarize(str(path), load_run(path, args.hours), expected, args.hours) for path in args.runs]
    print_table(summaries, args.hours)
    if args.json is not None:
        args.json.write_text(
            json.dumps({"generated_at": datetime.now().isoformat(), "runs": summaries}, indent=1) + "\n"
        )


if __name__ == "__main__":
    main()
