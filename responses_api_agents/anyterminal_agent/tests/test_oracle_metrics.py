# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the cgroup resource metrics, sandbox knobs, and the TB 2.1 prepare() entrypoint."""

import json
from pathlib import Path

import pytest

from responses_api_agents.anyterminal_agent import prepare as prepare_mod
from responses_api_agents.anyterminal_agent.app import (
    AnyTerminalAgentConfig,
    TerminalBenchMetrics,
    _parse_cgroup_stat,
)


AGENT_KWARGS = dict(
    host="127.0.0.1",
    port=1,
    name="anyterminal_agent",
    entrypoint="app.py",
    agent_server_module="responses_api_agents.hermes_agent.app",
    agent_server_class="HermesAgent",
    agent_config_class="HermesAgentConfig",
)


# ── _parse_cgroup_stat ────────────────────────────────────────────────────────────


class TestParseCgroupStat:
    def test_parses_all_three_counters(self) -> None:
        assert _parse_cgroup_stat("CG 12345678 90000 268435456\n") == {
            "cpu_usec": 12345678.0,
            "throttled_usec": 90000.0,
            "mem_peak_bytes": 268435456.0,
        }

    def test_missing_counters_become_none(self) -> None:
        parsed = _parse_cgroup_stat("CG - - 268435456")
        assert parsed["cpu_usec"] is None
        assert parsed["throttled_usec"] is None
        assert parsed["mem_peak_bytes"] == 268435456.0

    def test_ignores_unrelated_output_lines(self) -> None:
        assert _parse_cgroup_stat("warning: something\nCG 10 20 30\n") == {
            "cpu_usec": 10.0,
            "throttled_usec": 20.0,
            "mem_peak_bytes": 30.0,
        }

    @pytest.mark.parametrize("stdout", [None, "", "garbage", "CG 1 2"])
    def test_unparseable_returns_none(self, stdout) -> None:
        assert _parse_cgroup_stat(stdout) is None


# ── TerminalBenchMetrics resource fields ─────────────────────────────────────────


class TestResourceMetricsFields:
    def test_resource_fields_default_to_none(self) -> None:
        m = TerminalBenchMetrics()
        for field in (
            "cpu_solve_sec",
            "cpu_verify_sec",
            "cpu_total_sec",
            "cpu_util_solve",
            "cpu_util_verify",
            "cpu_throttled_sec",
            "mem_peak_mib",
        ):
            assert getattr(m, field) is None

    def test_resource_fields_round_trip(self) -> None:
        m = TerminalBenchMetrics(cpu_solve_sec=1.5, cpu_util_solve=0.25, mem_peak_mib=512.0)
        d = m.model_dump()
        assert d["cpu_solve_sec"] == 1.5
        assert d["cpu_util_solve"] == 0.25
        assert d["mem_peak_mib"] == 512.0


# ── Agent config knobs ────────────────────────────────────────────────────────────


class TestSandboxKnobs:
    def test_knob_defaults_preserve_legacy_behavior(self) -> None:
        cfg = AnyTerminalAgentConfig(**AGENT_KWARGS)
        assert cfg.cpu_pin_enabled is False
        assert cfg.override_cpus is None
        assert cfg.override_memory_mb is None
        assert cfg.ray_task_num_cpus is None
        assert cfg.sandbox_metadata == {}

    def test_knobs_accept_values(self) -> None:
        cfg = AnyTerminalAgentConfig(
            **AGENT_KWARGS,
            cpu_pin_enabled=True,
            override_cpus=2,
            override_memory_mb=4096,
            ray_task_num_cpus=0.25,
            sandbox_metadata={"nemo.nvidia.com/resources": "custom"},
        )
        assert cfg.cpu_pin_enabled is True
        assert cfg.override_cpus == 2.0
        assert cfg.override_memory_mb == 4096
        assert cfg.ray_task_num_cpus == 0.25
        assert cfg.sandbox_metadata == {"nemo.nvidia.com/resources": "custom"}


# ── prepare() ────────────────────────────────────────────────────────────────────


def _make_task(root: Path, name: str, cpus: int = 1, memory_mb: int = 2048) -> None:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text(
        f"""
[environment]
docker_image = "example/{name}:1"
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 10240
gpus = 0

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 600.0
"""
    )
    (task / "instruction.md").write_text(f"Solve {name}.")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")


class TestPrepare:
    def test_writes_one_row_per_task_with_agent_ref(self, tmp_path: Path, monkeypatch) -> None:
        tasks = tmp_path / "tasks"
        _make_task(tasks, "alpha-task")
        _make_task(tasks, "beta-task", cpus=4, memory_mb=8192)
        (tasks / "not-a-task").mkdir()  # no task.toml -> skipped
        monkeypatch.setenv("TERMINAL_BENCH_2_1_TASKS_DIR", str(tasks))

        out = prepare_mod.prepare(agent_name="my_oracle", output_fpath=tmp_path / "out.jsonl")

        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert [r["responses_create_params"]["metadata"]["task_name"] for r in rows] == ["alpha-task", "beta-task"]
        assert all(r["agent_ref"] == {"name": "my_oracle"} for r in rows)
        beta = rows[1]["responses_create_params"]["metadata"]
        assert beta["docker_image"] == "example/beta-task:1"
        assert beta["cpus"] == "4"
        assert beta["memory_mb"] == "8192"
        assert beta["agent_timeout_sec"] == "900.0"
        assert beta["verifier_timeout_sec"] == "600.0"
        assert beta["workdir"] == "/app"
        assert Path(beta["task_dir"]).is_absolute()
        assert rows[0]["responses_create_params"]["input"][0]["content"] == "Solve alpha-task."

    def test_missing_env_dir_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("TERMINAL_BENCH_2_1_TASKS_DIR", str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError):
            prepare_mod.prepare(output_fpath=tmp_path / "out.jsonl")

    def test_empty_tasks_dir_raises(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "tasks").mkdir()
        monkeypatch.setenv("TERMINAL_BENCH_2_1_TASKS_DIR", str(tmp_path / "tasks"))
        with pytest.raises(RuntimeError, match="No tasks"):
            prepare_mod.prepare(output_fpath=tmp_path / "out.jsonl")

    def test_pinned_checkout_is_a_commit_sha(self) -> None:
        assert len(prepare_mod.TB21_PINNED_COMMIT) == 40
        assert prepare_mod.TB21_REPO_URL.endswith("terminal-bench-2-1.git")
