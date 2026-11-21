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
import random
from datetime import datetime, timedelta

import pytest

from resources_servers.arc_agi_3.rhae import game_score, levels_completed_at, rhae_at, wallclock_seconds


# The reference: an independent NVIDIA-authored implementation of the Kaggle kit's scorer (``game_score`` and
# ``rhae_at``; not third-party code, Apache-2.0 here), in this repository's formatting (quotes, line breaks); rhae_at
# returns the per-attempt scores (the original returns their mean and the attempt count) so that the tests can compare
# per game.
def kit_game_score(apl, base, nlev, done):
    if not base or not nlev:
        return 0.0
    tot = tw = mw = 0.0
    for i in range(nlev):
        w = i + 1
        tw += w
        a = apl[i] if i < len(apl) else 0
        s = min(115.0, (base[i] / a) ** 2 * 100) if i < done and a and a > 0 else 0.0
        if s > 0:
            mw += w
        tot += s * w
    return 0.0 if tw == 0 else min(tot / tw, mw / tw * 100)


def kit_rhae_at(bench, t_seconds=None):
    start = datetime.fromisoformat(bench["start_time"])
    scores, levels = [], 0
    for r in bench["game_runs"]:
        apl = r.get("actions_per_level") or []
        done = r.get("levels_completed") or 0
        history = r.get("history") or []
        if t_seconds is not None:
            if r.get("started_at"):
                g0 = (datetime.fromisoformat(r["started_at"]) - start).total_seconds()
                n = k = 0
                for k in range(done):
                    n += apl[k] or 0
                    wall = history[n - 1].get("wallclock_seconds") or 0 if n and n <= len(history) else None
                    if wall is None or g0 + wall > t_seconds:
                        break
                else:
                    k = done
                done = k if done else 0
            else:
                done = 0
        levels += done
        scores.append(kit_game_score(apl, r.get("base_actions_per_level"), r.get("number_of_levels") or 0, done))
    return scores, levels


def random_run(rng: random.Random, start: datetime):
    nlev = rng.randint(1, 9)
    base = [rng.randint(5, 200) for _ in range(nlev)]
    done = rng.randint(0, nlev)
    apl = [rng.randint(1, 400) for _ in range(done)]
    if done < nlev:
        apl.append(rng.randint(0, 300))
    apl += [0] * (nlev - len(apl))
    t, history = 0.0, []
    for _ in range(sum(apl)):
        t += rng.uniform(0.5, 30.0)
        history.append({"action": {"id": "ACTION1", "data": {}}, "wallclock_seconds": t})
    offset = rng.uniform(0, 30)
    return {
        "number_of_levels": nlev,
        "base_actions_per_level": base,
        "actions_per_level": apl,
        "levels_completed": done,
        "history": history,
        "started_at": (start + timedelta(seconds=offset)).isoformat(),
    }, offset


def test_game_score_known_values() -> None:
    # Two levels, both cleared at the human baseline: 100.
    assert game_score([10, 20], [10, 20], 2, 2) == pytest.approx(100.0)
    # Level 1 cleared twice as slow as the human (25), level 2 not cleared: weighted (25*1 + 0*2)/3, cap 1/3*100.
    assert game_score([20, 5], [10, 20], 2, 1) == pytest.approx(25.0 / 3)
    # A level faster than the human is capped at 115; the attempt is capped at 100.
    assert game_score([1, 1], [10, 20], 2, 2) == pytest.approx(100.0)
    assert game_score([0, 0], [10, 20], 2, 0) == 0.0
    assert game_score([5], None, 1, 1) == 0.0


def test_matches_kit_scorer_on_random_runs() -> None:
    rng = random.Random(1234)
    start = datetime(2026, 11, 4, 12, 0, 0)
    runs, offsets = zip(*(random_run(rng, start) for _ in range(300)))
    bench = {"start_time": start.isoformat(), "game_runs": list(runs)}
    for t in (None, 600.0, 3600.0, 7200.0):
        kit_scores, _ = kit_rhae_at(bench, t)
        for run, offset, kit_score in zip(runs, offsets, kit_scores):
            ours = rhae_at(
                run["actions_per_level"],
                run["base_actions_per_level"],
                run["number_of_levels"],
                run["levels_completed"],
                wallclock_seconds(run["history"]),
                t_seconds=t,
                offset_seconds=offset,
            )
            assert ours == pytest.approx(kit_score, abs=1e-9)


def test_levels_completed_at_edges() -> None:
    walls = [10.0, 20.0, 30.0, 40.0]
    assert levels_completed_at([2, 2], walls, 2, 25.0) == 1
    assert levels_completed_at([2, 2], walls, 2, 40.0) == 2
    assert levels_completed_at([2, 2], walls, 2, 5.0) == 0
    assert levels_completed_at([2, 2], walls, 0, 100.0) == 0
    # A level cleared with no recorded action time is not counted (kit: wall None -> stop).
    assert levels_completed_at([2, 5], walls, 2, 100.0) == 1
