# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.path_utils import aggregate_metrics_path_for, failures_path_for, resolve_run_output_dir


class TestSiblingPaths:
    def test_failures_path_sits_beside_the_rollouts(self):
        assert failures_path_for(Path("results/rollouts.jsonl")) == Path("results/rollouts_failures.jsonl")

    def test_aggregate_metrics_path_switches_suffix(self):
        assert aggregate_metrics_path_for(Path("results/rollouts.jsonl")) == Path(
            "results/rollouts_aggregate_metrics.json"
        )


class TestResolveRunOutputDir:
    """`resolve_run_output_dir` picks the directory a compare/stat-test report is written into."""

    def test_defaults_to_the_runs_own_directory(self, tmp_path):
        run = tmp_path / "run_b" / "rollouts.jsonl"
        run.parent.mkdir(parents=True)
        run.write_text("{}")
        assert resolve_run_output_dir(str(run)) == tmp_path / "run_b"

    def test_absolute_output_dir_wins_over_the_run_directory(self, tmp_path):
        assert (
            resolve_run_output_dir("runs/b/r.jsonl", output_dirpath=str(tmp_path / "elsewhere"))
            == tmp_path / "elsewhere"
        )

    def test_relative_output_dir_is_anchored_at_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_run_output_dir("runs/b/r.jsonl", output_dirpath="out") == tmp_path / "out"

    def test_subdir_is_appended_to_either_branch(self, tmp_path):
        assert resolve_run_output_dir("x", output_dirpath=str(tmp_path), subdir="stats") == tmp_path / "stats"
        run = tmp_path / "run_b" / "rollouts.jsonl"
        run.parent.mkdir(parents=True)
        run.write_text("{}")
        assert resolve_run_output_dir(str(run), subdir="stats") == tmp_path / "run_b" / "stats"

    def test_a_relative_run_stays_under_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, raising=False)
        monkeypatch.chdir(tmp_path)
        assert resolve_run_output_dir("runs/b/r.jsonl") == tmp_path / "runs" / "b"

    def test_an_absolute_run_keeps_its_own_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path / "..")
        absolute = tmp_path / "runs" / "b" / "r.jsonl"
        assert resolve_run_output_dir(str(absolute)) == tmp_path / "runs" / "b"
