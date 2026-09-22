# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from benchmarks.terminal_bench_2_1 import prepare_inkling_small as preparation
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def pinned_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "--quiet")
    git(origin, "config", "user.name", "Preparation Test")
    git(origin, "config", "user.email", "test@example.invalid")
    # Deliberately reverse creation order; preparation must use the manifest order.
    names = [f"task-{index:02d}" for index in range(89)]
    for name in reversed(names):
        directory = origin / "tasks" / name
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text(
            f'[task]\nname = "terminal-bench/{name}"\n[environment]\ndocker_image = "example/{name}:pinned"\n'
        )
        (directory / "instruction.md").write_text(f"Solve {name}.\nKeep the café heading.\n\n", encoding="utf-8")
    git(origin, "add", ".")
    git(origin, "commit", "--quiet", "-m", "Fixture tasks")
    rows = [
        {
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": f"Solve {name}.\nKeep the café heading.\n\n"},
                    {"role": "user", "content": preparation.TERMINAL_INTERACTION_GUIDANCE},
                ]
            },
            "task_name": f"terminal-bench/{name}",
            "docker_image": f"example/{name}:pinned",
            "task_folder": name,
        }
        for name in names
    ]
    normalized = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "task_repository": str(origin),
                "task_revision": git(origin, "rev-parse", "HEAD"),
                "task_order": names,
                "normalized_rows_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(preparation, "GYM_ROOT", tmp_path)
    monkeypatch.setattr(preparation, "REFERENCE_PATH", reference)
    monkeypatch.setattr(preparation, "SOURCE_PATH", tmp_path / "data" / "tasks")
    monkeypatch.setattr(preparation, "OUTPUT_PATH", tmp_path / "data" / "rows.jsonl")
    return rows


def test_pinned_clone_preserves_prompts_order_and_tags_and_repairs_cached_rows(pinned_tasks: list[dict]) -> None:
    output = preparation.prepare()
    original = output.read_bytes()
    rows = [json.loads(line) for line in original.splitlines()]
    for row, expected in zip(rows, pinned_tasks, strict=True):
        assert row["task_folder"] == f"data/tasks/tasks/{expected['task_folder']}"
        row["task_folder"] = Path(row["task_folder"]).name
        assert row == expected
    output.write_text("stale cached inputs\n")
    preparation.prepare()
    assert output.read_bytes() == original


@pytest.mark.parametrize("change", ["tracked", "untracked", "revision"])
def test_changed_task_checkout_is_rejected_without_rewriting_inputs(pinned_tasks: list[dict], change: str) -> None:
    output = preparation.prepare()
    original = output.read_bytes()
    source = preparation.SOURCE_PATH
    if change == "tracked":
        (source / "tasks/task-00/instruction.md").write_text("Changed instruction")
    elif change == "untracked":
        (source / "tasks/task-00/unexpected_test.py").write_text("unexpected verifier input")
    else:
        git(
            source,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "New",
        )
    with pytest.raises(ValueError, match="modified|Expected task revision"):
        preparation.prepare()
    assert output.read_bytes() == original


@pytest.mark.parametrize("change", ["order", "guidance", "missing_task"])
def test_input_drift_is_rejected(pinned_tasks: list[dict], monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    output = preparation.prepare()
    original = output.read_bytes()
    if change == "guidance":
        monkeypatch.setattr(preparation, "TERMINAL_INTERACTION_GUIDANCE", "Different guidance")
    else:
        reference = json.loads(preparation.REFERENCE_PATH.read_text())
        if change == "order":
            reference["task_order"].reverse()
        else:
            reference["task_order"].pop()
        preparation.REFERENCE_PATH.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="differ from the reference|89 unique tasks"):
        preparation.prepare()
    assert output.read_bytes() == original


def test_full_profile_enables_fixes_and_reference_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "unused.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/terminal_bench_2_1/inkling_small.yaml",
                        "benchmarks/nemotron_3.5_super/sandbox_utils.yaml",
                        "benchmarks/nemotron_3.5_super/policy_model_override.yaml",
                    ],
                    "policy_base_url": "http://127.0.0.1:1/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "offline-model",
                }
            ),
        )
    )
    config = OmegaConf.to_container(config, resolve=True)
    agent = config["terminal_bench_2_1_terminus_2_sandboxed_agent"]["responses_api_agents"][
        "terminus_2_sandboxed_agent"
    ]
    resources = config[agent["resources_server"]["name"]]["resources_servers"]["terminal_bench_2_1"]
    model = config["policy_model"]["responses_api_models"]["vllm_model"]
    assert agent["entrypoint"] == "app.py"
    assert agent["interleaved_thinking"] is model["uses_interleaved_reasoning"] is True
    assert agent["recover_stalled_interrupts"] is True
    assert agent["terminal_hidden_mounts"] == ["/mnt/s3-data", "/mnt/.s3-gate"]
    assert agent["remote_tmux_binary_path"] == "/mnt/s3-data/data/bxyu/tmux/tmux-3.7c-linux-x86_64"
    assert agent["model_context_limit"] == 1048576
    assert agent["sandbox_timeout"] == 10800
    assert agent["llm_request_timeout"] == 3600
    assert agent["max_turns"] is agent["model_output_limit"] is None
    assert resources["evaluation_timeout"] == 1800
    assert resources["is_verifying_golden_patch"] is False
    assert resources["sandbox_config"]["env"]["DEBIAN_FRONTEND"] == "noninteractive"
    assert resources["sandbox_config"]["resources"] == {"cpu": 4, "memory_mib": 16384, "disk_gib": 30}
    assert model["chat_template_kwargs"] == {"reasoning_effort": "max"}
    assert agent["num_workers"] == model["num_workers"] == 4
    assert config["num_samples_in_parallel"] == 256
    assert config["observability_enabled"] is True
    assert config["upload_rollouts"] is False
    assert "opensandbox" in config["sandbox"]
    assert agent["datasets"] == [
        {
            "name": "terminal_bench_2_1_inkling_small",
            "type": "benchmark",
            "jsonl_fpath": "benchmarks/terminal_bench_2_1/data/benchmark_inkling_small.jsonl",
            "prepare_script": "benchmarks/terminal_bench_2_1/prepare_inkling_small.py",
            "num_repeats": 8,
        }
    ]
