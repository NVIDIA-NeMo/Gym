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
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from resources_servers.arc_agi_3 import summarize_rhae


GAMES = ["aa11", "bb22", "cc33"]
HOURS = [1.0, 2.0]


def meta(game, version="v1"):
    return {"game_id": game, "full_game_id": f"{game}-{version}"}


def write_expected(path: Path, games=GAMES) -> Path:
    path.write_text("".join(json.dumps({"verifier_metadata": meta(g)}) + "\n" for g in games))
    return path


def game_run(game, apl, done, step_s, start="2026-11-04T12:00:00", state="gave_up", version="v1"):
    return {
        "game_id": f"{game}-{version}",
        "number_of_levels": 2,
        "base_actions_per_level": [10, 20],
        "actions_per_level": apl,
        "levels_completed": done,
        "state": state,
        "history": [{"wallclock_seconds": step_s * (i + 1)} for i in range(sum(apl))],
        "started_at": start,
    }


def write_benchmark(path: Path, runs) -> Path:
    path.write_text(json.dumps({"start_time": "2026-11-04T12:00:00", "game_runs": runs}))
    return path


def rollout(game, rhae, h1, h2, **extra):
    row = {"game_id": game, "verifier_metadata": meta(game), "rhae": rhae, "rhae_1h": h1, "rhae_2h": h2}
    row["clock_offset_seconds"] = 0.0
    row.update(extra)
    return json.dumps(row) + "\n"


def full_benchmark(path: Path) -> Path:
    return write_benchmark(
        path,
        [
            # Both levels at the human baseline, the second cleared after 1 h: 100 at the end, 100/3 at 1 h.
            game_run("aa11", [10, 20], 2, 150.0),
            game_run("bb22", [20, 3], 1, 10.0),
            game_run("cc33", [0, 0], 0, 1.0),
        ],
    )


def test_complete_benchmark_run(tmp_path: Path) -> None:
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl"))
    summary = summarize_rhae.summarize(
        "r", summarize_rhae.load_run(full_benchmark(tmp_path / "b.json"), HOURS), expected, HOURS
    )
    assert summary["complete"] and summary["coverage"] == {"end": 3, "rhae_1h": 3, "rhae_2h": 3}
    assert summary["overall"]["end"] == pytest.approx((100.0 + 25.0 / 3 + 0.0) / 3)
    assert summary["per_game"]["aa11-v1"]["rhae_1h"] == pytest.approx(100.0 / 3)


def test_partial_set_has_no_score(tmp_path: Path) -> None:
    # F2: one winning game out of the expected set is not a benchmark score.
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl"))
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(rollout("aa11", 100.0, 100.0, 100.0))
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(rollouts, HOURS), expected, HOURS)
    assert not summary["complete"] and summary["overall"] == {"end": None, "rhae_1h": None, "rhae_2h": None}
    assert summary["coverage"]["end"] == 1 and summary["counts"]["missing"] == 2
    assert summary["per_game"]["bb22-v1"]["status"] == "missing"


def test_flagged_rows_and_failures_make_the_run_incomplete(tmp_path: Path) -> None:
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl"))
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(
        rollout("aa11", 100.0, 100.0, 100.0)
        + rollout("bb22", 50.0, 50.0, 50.0, persistence_error="OSError: disk full")
        + rollout("cc33", 0.0, None, None, agent_state="not_started", agent_error="GameStartError: x")
    )
    (tmp_path / "rollouts_failures.jsonl").write_text(json.dumps({"verifier_metadata": meta("cc33")}) + "\n")
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(rollouts, HOURS), expected, HOURS)
    assert not summary["complete"] and summary["overall"]["end"] is None
    counts = summary["counts"]
    assert counts["persistence_error"] == 1 and counts["not_started"] == 1 and counts["agent_error"] == 1
    assert counts["rollout_failed"] == 1
    assert summary["per_game"]["bb22-v1"]["end"] is None and summary["per_game"]["aa11-v1"]["end"] == 100.0


def test_legacy_rollouts_have_no_timed_scores(tmp_path: Path) -> None:
    # F7: rows timed on a per-game clock (no clock_offset_seconds) give no @1h/@2h value.
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11"]))
    rollouts = tmp_path / "rollouts.jsonl"
    row = {"game_id": "aa11", "verifier_metadata": meta("aa11"), "rhae": 40.0, "rhae_1h": 40.0, "rhae_2h": 40.0}
    rollouts.write_text(json.dumps(row) + "\n")
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(rollouts, HOURS), expected, HOURS)
    assert summary["overall"]["end"] == 40.0 and summary["overall"]["rhae_1h"] is None
    assert summary["coverage"]["rhae_1h"] == 0


def test_run_dir_scores_from_benchmark_with_verifier_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "runs" / "x").mkdir(parents=True)
    full_benchmark(run_dir / "runs" / "x" / "benchmark.json")
    (run_dir / "rollouts.jsonl").write_text(
        rollout("aa11", 100.0, 1.0, 1.0) + rollout("bb22", 0.0, 0.0, 0.0, persistence_error="OSError: full")
    )
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl"))
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(run_dir, HOURS), expected, HOURS)
    # Scores come from benchmark.json (the run clock), flags from the rollouts.
    assert summary["per_game"]["aa11-v1"]["rhae_1h"] == pytest.approx(100.0 / 3)
    assert summary["counts"]["persistence_error"] == 1 and not summary["complete"]


def test_benchmark_timing_and_clock(tmp_path: Path) -> None:
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11", "bb22"]))
    bad = game_run("aa11", [1, 0], 1, 1.0)
    bad["history"][0]["wallclock_seconds"] = "nan"
    # Started 10 s after the run start, cleared at 3595 s of its own play = 3605 s on the run clock.
    late = game_run("bb22", [1, 0], 1, 3595.0, start="2026-11-04T12:00:10")
    summary = summarize_rhae.summarize(
        "r", summarize_rhae.load_run(write_benchmark(tmp_path / "b.json", [bad, late]), HOURS), expected, HOURS
    )
    assert summary["per_game"]["aa11-v1"]["rhae_1h"] is None and summary["per_game"]["aa11-v1"]["end"] is not None
    assert summary["counts"]["timing_invalid"] == 1
    assert summary["per_game"]["bb22-v1"]["rhae_1h"] == 0.0 and summary["per_game"]["bb22-v1"]["rhae_2h"] > 0
    assert summary["overall"]["end"] is not None and summary["overall"]["rhae_1h"] is None


def test_cancelled_and_unexpected_games(tmp_path: Path) -> None:
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11"]))
    runs = [game_run("aa11", [1, 0], 1, 1.0, state="cancelled"), game_run("zz99", [1, 0], 1, 1.0)]
    summary = summarize_rhae.summarize(
        "r", summarize_rhae.load_run(write_benchmark(tmp_path / "b.json", runs), HOURS), expected, HOURS
    )
    assert summary["counts"]["cancelled"] == 1 and summary["unexpected_games"] == ["zz99-v1"]
    assert not summary["complete"]


def test_duplicates_rejected(tmp_path: Path) -> None:
    rollouts = tmp_path / "r.jsonl"
    rollouts.write_text(rollout("aa11", 1.0, 1.0, 1.0) * 2)
    with pytest.raises(ValueError, match="twice"):
        summarize_rhae.load_run(rollouts, HOURS)
    with pytest.raises(ValueError, match="duplicate"):
        summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11", "aa11"]))


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(rollout("aa11", 100.0, 100.0, 100.0))
    out = tmp_path / "summary.json"
    summarize_rhae.main(
        [
            str(rollouts),
            str(full_benchmark(tmp_path / "benchmark.json")),
            "--expected",
            str(write_expected(tmp_path / "e.jsonl")),
            "--json",
            str(out),
        ]
    )
    printed = capsys.readouterr().out
    assert "INCOMPLETE 1/3" in printed and "missing 2" in printed
    runs = json.loads(out.read_text())["runs"]
    assert runs[0]["complete"] is False and runs[1]["complete"] is True


def snapshot(path: Path, runs, saved_after_s: float) -> Path:
    start = datetime(2026, 11, 4, 12, 0, 0)
    path.write_text(
        json.dumps(
            {
                "start_time": start.isoformat(),
                "saved_at": (start + timedelta(seconds=saved_after_s)).isoformat(),
                "play_seconds": 7200.0,
                "game_runs": runs,
            }
        )
    )
    return path


def test_live_snapshot_is_not_final(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # F2 residual: a snapshot saved 60 s into play (the game still playing, one of two levels cleared) is the score
    # so far, not a complete run with an end / @1h / @2h score.
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11"]))
    bench = snapshot(tmp_path / "b.json", [game_run("aa11", [10, 2], 1, 5.0, state="playing")], 60.0)
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(bench, HOURS), expected, HOURS)
    assert summary["complete"] is False and summary["overall"]["end"] == pytest.approx(100.0 / 3)
    assert summary["column_status"] == {"end": "not_final", "rhae_1h": "not_final", "rhae_2h": "not_final"}
    summarize_rhae.print_table([summary], HOURS)
    printed = capsys.readouterr().out
    assert "33.33 not final" in printed and "33.33*" in printed


def test_snapshot_past_an_hour_makes_that_hour_final(tmp_path: Path) -> None:
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11"]))
    bench = snapshot(tmp_path / "b.json", [game_run("aa11", [10, 2], 1, 5.0, state="playing")], 3700.0)
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(bench, HOURS), expected, HOURS)
    assert summary["column_status"] == {"end": "not_final", "rhae_1h": "final", "rhae_2h": "not_final"}
    assert not summary["complete"]


def test_killed_run_without_verifier_rows_is_not_final(tmp_path: Path) -> None:
    # Every game ended in the snapshot, but the run was killed before the verifier saw two of them.
    run_dir = tmp_path / "run"
    (run_dir / "runs" / "x").mkdir(parents=True)
    full_benchmark(run_dir / "runs" / "x" / "benchmark.json")
    (run_dir / "rollouts.jsonl").write_text(rollout("aa11", 100.0, 100.0 / 3, 100.0))
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl"))
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(run_dir, HOURS), expected, HOURS)
    assert summary["counts"]["not_verified"] == 2 and not summary["complete"]
    assert summary["column_status"]["end"] == "not_final" and summary["overall"]["end"] is not None
    assert summary["per_game"]["aa11-v1"]["final"]["end"] and not summary["per_game"]["bb22-v1"]["final"]["end"]


@pytest.mark.parametrize("offset", [None, "0", float("nan"), -5.0, True])
def test_rollout_without_a_run_clock_offset_has_no_timed_score(tmp_path: Path, offset) -> None:
    # F7 residual: the offset key with no real value (verified without the run origin) is not a run clock.
    expected = summarize_rhae.load_expected(write_expected(tmp_path / "e.jsonl", ["aa11"]))
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(rollout("aa11", 100.0, 100.0, 100.0, clock_offset_seconds=offset))
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(rollouts, HOURS), expected, HOURS)
    assert summary["overall"]["end"] == 100.0 and summary["overall"]["rhae_1h"] is None
    assert summary["counts"]["no run clock"] == 1 and summary["column_status"]["rhae_1h"] == "incomplete"


def test_two_versions_of_a_game_are_two_games(tmp_path: Path) -> None:
    # Synthetic sets hold several versions of one game name (sg0005-aee9edbb, sg0005-2b3d7198): keyed by full id.
    expected_path = tmp_path / "e.jsonl"
    expected_path.write_text("".join(json.dumps({"verifier_metadata": meta("aa11", v)}) + "\n" for v in ("v1", "v2")))
    expected = summarize_rhae.load_expected(expected_path)
    bench = write_benchmark(
        tmp_path / "b.json", [game_run("aa11", [10, 20], 2, 1.0), game_run("aa11", [0, 0], 0, 1.0, version="v2")]
    )
    summary = summarize_rhae.summarize("r", summarize_rhae.load_run(bench, HOURS), expected, HOURS)
    assert expected == ["aa11-v1", "aa11-v2"] and summary["complete"]
    assert summary["per_game"]["aa11-v1"]["end"] == 100.0 and summary["per_game"]["aa11-v2"]["end"] == 0.0
