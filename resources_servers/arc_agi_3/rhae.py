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
"""ARC-AGI-3 RHAE (relative human action efficiency) for one game attempt.

The formula is the ARC-AGI-3 scorecard's (``arc_agi.scorecard.EnvironmentScoreCalculator`` v0.9.8, the same as taaf
``GameRun`` and the Kaggle kit's scorer):

- level ``i`` (1-based) of ``n`` scores ``min(115, (baseline_i / actions_i) ** 2 * 100)`` once cleared, else 0;
- the attempt score is the mean of the level scores weighted by ``i``, capped at
  ``(weight of the levels that scored) / (total weight) * 100``.

``rhae_at`` is the kit's score at a time limit ``T``: a level counts only if the action that cleared it happened
before ``T`` seconds from the run's start (the kit's benchmark start; each game adds its start offset). Times must be
complete, finite, non-negative and non-decreasing (:func:`validate_timing`); otherwise there is no timed score.

Scores are on the kit's 0-100 scale.
"""

import math
from collections.abc import Sequence


LEVEL_SCORE_CAP = 115.0


def game_score(
    actions_per_level: Sequence[int],
    base_actions_per_level: Sequence[int] | None,
    number_of_levels: int,
    levels_completed: int,
) -> float:
    """Score of one attempt on the 0-100 scale (kit ``game_score``)."""
    if not base_actions_per_level or not number_of_levels:
        return 0.0
    total = total_weight = scored_weight = 0.0
    for i in range(number_of_levels):
        weight = i + 1
        total_weight += weight
        actions = actions_per_level[i] if i < len(actions_per_level) else 0
        score = 0.0
        if i < levels_completed and actions and actions > 0:
            score = min(LEVEL_SCORE_CAP, (base_actions_per_level[i] / actions) ** 2 * 100)
        if score > 0:
            scored_weight += weight
        total += score * weight
    if total_weight == 0:
        return 0.0
    return min(total / total_weight, scored_weight / total_weight * 100)


def levels_completed_at(
    actions_per_level: Sequence[int],
    action_wallclock_seconds: Sequence[float],
    levels_completed: int,
    t_seconds: float,
    offset_seconds: float = 0.0,
) -> int:
    """Levels whose clearing action happened at or before ``t_seconds`` (kit ``rhae_at`` loop).

    ``action_wallclock_seconds[j]`` is the time of action ``j`` since the game started; ``offset_seconds`` is the
    game's start relative to the clock origin of ``t_seconds``.
    """
    if not levels_completed:
        return 0
    actions_so_far = 0
    for level in range(levels_completed):
        actions_so_far += actions_per_level[level] or 0
        if actions_so_far and actions_so_far <= len(action_wallclock_seconds):
            wall = action_wallclock_seconds[actions_so_far - 1] or 0
        else:
            return level
        if offset_seconds + wall > t_seconds:
            return level
    return levels_completed


def rhae_at(
    actions_per_level: Sequence[int],
    base_actions_per_level: Sequence[int] | None,
    number_of_levels: int,
    levels_completed: int,
    action_wallclock_seconds: Sequence[float],
    t_seconds: float | None = None,
    offset_seconds: float = 0.0,
) -> float:
    """Attempt score at ``t_seconds`` of play, or at the end when ``t_seconds`` is None."""
    done = levels_completed
    if t_seconds is not None:
        done = levels_completed_at(
            actions_per_level, action_wallclock_seconds, levels_completed, t_seconds, offset_seconds
        )
    return game_score(actions_per_level, base_actions_per_level, number_of_levels, done)


class TimingError(ValueError):
    """An attempt's action times cannot support a timed (RHAE@T) score."""


def wallclock_seconds(history: list[dict]) -> list[float]:
    """Per-action times (seconds since the game started) from a taaf ``GameRun.to_json_dict()`` history.

    Raises :class:`TimingError` unless every action has a finite, non-negative time and the times never decrease.
    """
    times: list[float] = []
    for index, entry in enumerate(history):
        value = entry.get("wallclock_seconds") if isinstance(entry, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TimingError(f"action {index}: wallclock_seconds is {value!r}, not a number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise TimingError(f"action {index}: wallclock_seconds {value!r} is not finite and non-negative")
        if times and value < times[-1]:
            raise TimingError(
                f"action {index}: wallclock_seconds {value} is earlier than the previous action's {times[-1]}"
            )
        times.append(value)
    return times


def validate_timing(history: list[dict], actions_per_level: Sequence[int]) -> list[float]:
    """Checked per-action times, also requiring one time per counted action."""
    times = wallclock_seconds(history)
    if len(times) != sum(int(a) for a in actions_per_level):
        raise TimingError(f"{len(times)} timed actions but {sum(actions_per_level)} counted in actions_per_level")
    return times


def clock_offset_seconds(benchmark_start_time: str, game_started_at: str) -> float:
    """Seconds from the run's clock origin to the game's start (the kit's ``started_at - start_time``)."""
    from datetime import datetime

    offset = (datetime.fromisoformat(game_started_at) - datetime.fromisoformat(benchmark_start_time)).total_seconds()
    if not math.isfinite(offset) or offset < 0:
        raise TimingError(f"game start {game_started_at} is before the run start {benchmark_start_time}")
    return offset


def recorded_play(events: list[dict], number_of_levels: int) -> tuple[list[int], list[float], int, str]:
    """One game run as the game server recorded it (``save_all_recordings``): ``(actions_per_level, action times in
    seconds since the run's first event, levels completed, the first event's UTC time)``.

    The first event is the RESET of ``make()`` (not an action); every later event is one action, counted on the level
    the game was on before it. Raises :class:`TimingError` when the events cannot give a timed score.
    """
    from datetime import datetime

    if not events:
        raise TimingError("the game server's recording of the run is empty")
    first = events[0]
    if ((first.get("data") or {}).get("action_input") or {}).get("id") != "RESET":
        raise TimingError("the recording does not start with the game's initial RESET")
    try:
        start = datetime.fromisoformat(first["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimingError(f"the recording's first event has no valid timestamp: {exc}") from exc
    if start.tzinfo is None:
        raise TimingError("the recording's timestamps carry no UTC offset")
    actions_per_level = [0] * number_of_levels
    times: list[float] = []
    level = int(first["data"].get("levels_completed") or 0)
    levels = level
    for index, event in enumerate(events[1:]):
        if level >= number_of_levels:
            raise TimingError(f"action {index} was recorded after all {number_of_levels} levels were cleared")
        seconds = (datetime.fromisoformat(event["timestamp"]) - start).total_seconds()
        if not math.isfinite(seconds) or seconds < 0 or (times and seconds < times[-1]):
            raise TimingError(f"action {index}: recorded time {event['timestamp']} is out of order")
        actions_per_level[level] += 1
        times.append(seconds)
        level = int(event["data"].get("levels_completed") or 0)
        levels = max(levels, level)
    return actions_per_level, times, levels, first["timestamp"]


def level_clear_steps(events: list[dict]) -> list[int]:
    """The game-wide action number (1-based) at which each level was first cleared, from a recording that
    :func:`recorded_play` accepted (the first event is the initial RESET; every later event is one action)."""
    steps: list[int] = []
    cleared = int(((events[0].get("data") or {}).get("levels_completed")) or 0)
    for step, event in enumerate(events[1:], start=1):
        level = int((event.get("data") or {}).get("levels_completed") or 0)
        while level > cleared:
            steps.append(step)
            cleared += 1
    return steps


def utc_clock_offset_seconds(run_start_utc: str, game_start_utc: str) -> float:
    """Seconds from the run's clock origin to a game's first server event; both must carry a UTC offset."""
    from datetime import datetime

    origin, start = datetime.fromisoformat(run_start_utc), datetime.fromisoformat(game_start_utc)
    if origin.tzinfo is None:
        raise TimingError(f"the run clock origin {run_start_utc} carries no UTC offset")
    offset = (start - origin).total_seconds()
    if not math.isfinite(offset) or offset < 0:
        raise TimingError(f"game start {game_start_utc} is before the run start {run_start_utc}")
    return offset
