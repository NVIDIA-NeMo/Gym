# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract checks for BioML-bench's Harbor task writer and its in-sandbox grading helper."""

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.biomlbench import biomlbench_task
from benchmarks.biomlbench.prepare import CHECKSUMS_FPATH, DOCKER_IMAGE, TASK_IDS, instruction, task_slug, write_task


def test_write_task_runs_agent_as_nonroot_and_grades_in_a_separate_sandbox(tmp_path: Path) -> None:
    pins = {"public": {"train.csv": "a" * 32}, "private": {"answers.csv": "b" * 32}}
    task_dir = tmp_path / task_slug("polarishub/tdcommons-herg")
    write_task(task_dir, "polarishub/tdcommons-herg", "GENERAL", "TASK", {"time_limit": 28800}, pins)

    config = tomllib.loads((task_dir / "task.toml").read_text())
    assert task_dir.name == "polarishub__tdcommons-herg"
    assert config["agent"] == {"timeout_sec": 28800, "user": "nonroot"}
    assert config["verifier"]["user"] == "root"
    # The answers exist only in the verifier's sandbox, which receives just the submission.
    assert config["verifier"]["environment_mode"] == "separate"
    assert config["artifacts"] == ["/home/submission"]
    assert config["environment"]["docker_image"] == DOCKER_IMAGE
    assert (config["environment"]["cpus"], config["environment"]["memory_mb"]) == (16, 65536)
    assert config["environment"]["gpus"] == 0
    for folder in ("environment", "tests"):
        assert json.loads((task_dir / folder / "task.json").read_text()) == {
            "task_id": "polarishub/tdcommons-herg",
            "checksums": pins,
        }
        assert (task_dir / folder / "biomlbench_task.py").is_file()
    test_sh = (task_dir / "tests" / "test.sh").read_text()
    prepare_line = "prepare --task-id polarishub/tdcommons-herg --data-dir /private/data"
    grade_line = "grade --task-id polarishub/tdcommons-herg --data-dir /private/data"
    assert prepare_line in test_sh and "--task-json /tests/task.json" in test_sh
    assert test_sh.index(prepare_line) < test_sh.index(grade_line)
    assert "--output /logs/verifier/reward.json" in test_sh
    assert "sample_submission" in (task_dir / "solution" / "solve.sh").read_text()


def test_instruction_follows_upstream_start_script_order() -> None:
    prompt = instruction("GENERAL", "TASK DESCRIPTION", 28800)
    assert prompt.startswith("GENERAL\nADDITIONAL NOTES\n")
    assert "a maximum of 8hrs 0mins 0secs hours" in prompt
    assert "Total Steps" not in prompt
    assert prompt.endswith("\nCOMPETITION INSTRUCTIONS\n------\n\nTASK DESCRIPTION")


def test_checksums_pin_every_task_except_random_sample_submissions() -> None:
    checksums = json.loads(CHECKSUMS_FPATH.read_text())
    assert sorted(checksums) == sorted(TASK_IDS)
    assert not any(task_id.startswith("kaggle/") for task_id in TASK_IDS)
    unpinned = sorted(
        f"{task_id}:{side}/{name}"
        for task_id, sides in checksums.items()
        for side, files in sides.items()
        for name, md5 in files.items()
        if md5 is None
    )
    assert unpinned == [
        "manual/open-problems-predict-modality:public/sample_submission.csv",
        "manual/open-problems-spatially-variable-genes:public/sample_submission.csv",
    ]
    assert all(checksums[task_id]["private"] for task_id in TASK_IDS)


@pytest.mark.parametrize(
    ("actual", "mismatches"),
    [
        ({"public": {"train.csv": "x", "sample_submission.csv": "random"}, "private": {"answers.csv": "y"}}, []),
        (
            {"public": {"train.csv": "changed", "sample_submission.csv": "r"}, "private": {"answers.csv": "y"}},
            ["public/train.csv"],
        ),
        ({"public": {"train.csv": "x"}, "private": {"answers.csv": "y"}}, ["public/sample_submission.csv"]),
        (
            {
                "public": {"train.csv": "x", "sample_submission.csv": "r", "extra.csv": "z"},
                "private": {"answers.csv": "y"},
            },
            ["public/extra.csv"],
        ),
    ],
)
def test_checksum_mismatches_require_every_file_and_skip_null_pins(actual: dict, mismatches: list[str]) -> None:
    expected = {"public": {"train.csv": "x", "sample_submission.csv": None}, "private": {"answers.csv": "y"}}
    assert biomlbench_task.checksum_mismatches(actual, expected) == mismatches


def test_rewards_report_medals_and_penalized_percentile() -> None:
    report = {
        "valid_submission": True,
        "any_medal": True,
        "gold_medal": False,
        "silver_medal": True,
        "bronze_medal": False,
        "above_median": True,
        "leaderboard_percentile": 91.5,
        "score": 0.71,
    }
    assert biomlbench_task.rewards(report) == {
        "reward": 1.0,
        "gold_medal": 0.0,
        "silver_medal": 1.0,
        "bronze_medal": 0.0,
        "above_median": 1.0,
        "valid_submission": 1.0,
        "leaderboard_percentile": 91.5,
        "score": 0.71,
    }
    invalid = biomlbench_task.rewards({"valid_submission": False, "leaderboard_percentile": 40.0, "score": 0.3})
    assert invalid["reward"] == 0.0
    assert invalid["leaderboard_percentile"] == 0.0
    assert "score" not in invalid


def test_rewards_drop_non_finite_scores() -> None:
    # A constant prediction scores NaN on Spearman; Harbor rejects non-finite reward values.
    result = biomlbench_task.rewards({"valid_submission": True, "score": float("nan"), "leaderboard_percentile": None})
    assert result["valid_submission"] == 1.0
    assert result["leaderboard_percentile"] == 0.0
    assert "score" not in result
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_submission_path_prefers_the_sample_submission_format(tmp_path: Path) -> None:
    assert biomlbench_task.submission_path(Path("sample_submission.h5ad"), tmp_path) is None
    (tmp_path / "submission.csv").write_text("id,y\n")
    assert biomlbench_task.submission_path(Path("sample_submission.h5ad"), tmp_path) == tmp_path / "submission.csv"
    (tmp_path / "submission.h5ad").write_text("")
    assert biomlbench_task.submission_path(Path("sample_submission.h5ad"), tmp_path) == tmp_path / "submission.h5ad"


def test_submission_path_ignores_symlinks(tmp_path: Path) -> None:
    answers = tmp_path / "answers.csv"
    answers.write_text("id,y\n0,1\n")
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    (submission_dir / "submission.csv").symlink_to(answers)
    assert biomlbench_task.submission_path(Path("sample_submission.csv"), submission_dir) is None


@pytest.mark.parametrize(
    ("content", "valid", "message"),
    [
        ("id,y\n0,0.5\n1,0.1\n", True, "Submission is valid."),
        ("id,y\n1,0.1\n0,0.5\n", True, "Submission is valid."),
        ("id,score\n0,0.5\n1,0.1\n", False, "Expected columns"),
        ("id,y\n0,0.5\n", False, "Expected 2 rows"),
        ("id,y\n0,0.5\n7,0.1\n", False, "differ from the sample submission"),
        ("id,y\n0,0.5\n1,\n", False, "missing values"),
    ],
)
def test_validate_checks_format_against_the_public_sample(tmp_path: Path, content: str, valid: bool, message: str):
    pytest.importorskip("pandas")
    sample = tmp_path / "sample_submission.csv"
    sample.write_text("id,y\n0,0.0\n1,0.0\n")
    submission = tmp_path / "submission.csv"
    submission.write_text(content)
    ok, text = biomlbench_task.validate(submission, sample)
    assert ok is valid
    assert message in text


class _Report:
    def __init__(self, **fields: object) -> None:
        self.fields = fields

    def to_dict(self) -> dict:
        return self.fields


@pytest.fixture
def upstream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A stand-in for the image's BioML-bench package: one task and recorded grader calls."""
    task = SimpleNamespace(
        id="polarishub/tdcommons-herg",
        sample_submission=tmp_path / "public" / "sample_submission.csv",
        public_dir=tmp_path / "public",
        private_dir=tmp_path / "private",
    )
    state = SimpleNamespace(task=task, graded=[], grade_error=None, checksums={})

    def grade_submission(path: Path, _task: object) -> _Report:
        state.graded.append(path)
        if state.grade_error is not None:
            raise state.grade_error
        return _Report(valid_submission=path.is_file(), any_medal=False, score=0.5 if path.is_file() else None)

    registry = SimpleNamespace(set_data_dir=lambda _dir: SimpleNamespace(get_task=lambda _id: task))
    modules = {
        "biomlbench": ModuleType("biomlbench"),
        "biomlbench.registry": SimpleNamespace(registry=registry),
        "biomlbench.grade": SimpleNamespace(grade_submission=grade_submission),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _grade_args(tmp_path: Path, submission_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task_id="polarishub/tdcommons-herg",
        data_dir=str(tmp_path / "data"),
        submission_dir=str(submission_dir),
        output=str(tmp_path / "verifier" / "reward.json"),
    )


def test_grade_scores_a_missing_submission_as_invalid(tmp_path: Path, upstream: SimpleNamespace) -> None:
    (tmp_path / "submission").mkdir()
    biomlbench_task.grade(_grade_args(tmp_path, tmp_path / "submission"))
    assert not upstream.graded[0].exists()
    reward = json.loads((tmp_path / "verifier" / "reward.json").read_text())
    assert reward["valid_submission"] == 0.0 and reward["reward"] == 0.0


def test_grade_propagates_grader_failures_without_a_reward_file(tmp_path: Path, upstream: SimpleNamespace) -> None:
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "submission.csv").write_text("id,y\n0,1\n")
    upstream.grade_error = ConnectionError("leaderboard download failed")
    with pytest.raises(ConnectionError, match="leaderboard download failed"):
        biomlbench_task.grade(_grade_args(tmp_path, tmp_path / "submission"))
    assert not (tmp_path / "verifier" / "reward.json").exists()


def _stub_preparer(monkeypatch: pytest.MonkeyPatch, upstream: SimpleNamespace) -> dict:
    """Make the stubbed upstream write one public and one private file; return their real pins."""
    task = upstream.task

    def download_and_prepare_dataset(_task: object, keep_raw: bool, skip_verification: bool) -> None:
        assert (keep_raw, skip_verification) == (False, True)
        task.public_dir.mkdir(parents=True)
        task.private_dir.mkdir(parents=True)
        (task.public_dir / "train.csv").write_text("id,y\n0,1\n")
        (task.private_dir / "answers.csv").write_text("id,y\n1,0\n")

    def generate_checksums(folder: Path) -> dict:
        return {path.name: hashlib.md5(path.read_bytes()).hexdigest() for path in sorted(folder.iterdir())}

    data = SimpleNamespace(
        download_and_prepare_dataset=download_and_prepare_dataset, generate_checksums=generate_checksums
    )
    monkeypatch.setitem(sys.modules, "biomlbench.data", data)
    return {
        "public": {"train.csv": hashlib.md5(b"id,y\n0,1\n").hexdigest()},
        "private": {"answers.csv": hashlib.md5(b"id,y\n1,0\n").hexdigest()},
    }


def _prepare_args(tmp_path: Path, pins: dict) -> SimpleNamespace:
    task_json = tmp_path / "task.json"
    task_json.write_text(json.dumps({"task_id": "polarishub/tdcommons-herg", "checksums": pins}))
    return SimpleNamespace(
        task_id="polarishub/tdcommons-herg",
        data_dir=str(tmp_path / "data"),
        public_dir=str(tmp_path / "home-data"),
        task_json=str(task_json),
    )


def test_prepare_publishes_the_public_split_when_the_pins_match(tmp_path, monkeypatch, upstream) -> None:
    pins = _stub_preparer(monkeypatch, upstream)
    biomlbench_task.prepare(_prepare_args(tmp_path, pins))
    assert (tmp_path / "home-data" / "train.csv").read_text() == "id,y\n0,1\n"
    assert not (tmp_path / "home-data" / "answers.csv").exists()


@pytest.mark.parametrize("side", ["public", "private"])
def test_prepare_refuses_changed_data_before_publishing_it(tmp_path, monkeypatch, upstream, side) -> None:
    pins = _stub_preparer(monkeypatch, upstream)
    name = next(iter(pins[side]))
    pins[side][name] = "0" * 32
    with pytest.raises(SystemExit, match=f"differs from its pinned checksums: {side}/{name}"):
        biomlbench_task.prepare(_prepare_args(tmp_path, pins))
    assert not (tmp_path / "home-data").exists()
