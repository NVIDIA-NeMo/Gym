# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test catalog metadata and README generation through its pre-commit hook."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import scripts.update_env_list as update_env_list
import yaml
from scripts.update_env_list import visit_agent_datasets


def _agent_config(train_dataset: dict) -> dict:
    return {
        "test_agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "datasets": [train_dataset],
                }
            }
        }
    }


def test_visit_agent_datasets_reads_huggingface_source() -> None:
    metadata = visit_agent_datasets(
        _agent_config(
            {
                "name": "train",
                "type": "train",
                "license": "Creative Commons Attribution 4.0 International",
                "source": {
                    "type": "huggingface",
                    "repo_id": "nvidia/example-dataset",
                },
            }
        )
    )

    assert metadata.types == ["train"]
    assert metadata.license == "Creative Commons Attribution 4.0 International"
    assert metadata.huggingface_repo_id == "nvidia/example-dataset"


def test_visit_agent_datasets_retains_legacy_huggingface_fallback() -> None:
    metadata = visit_agent_datasets(
        _agent_config(
            {
                "name": "train",
                "type": "train",
                "license": "Apache 2.0",
                "huggingface_identifier": {"repo_id": "nvidia/legacy-dataset"},
            }
        )
    )

    assert metadata.types == ["train"]
    assert metadata.license == "Apache 2.0"
    assert metadata.huggingface_repo_id == "nvidia/legacy-dataset"


def test_training_server_info_includes_benchmark_configs(tmp_path: Path, monkeypatch) -> None:
    resources_servers = tmp_path / "resources_servers"
    responses_api_agents = tmp_path / "responses_api_agents"
    benchmarks = tmp_path / "benchmarks"
    resources_servers.mkdir()
    responses_api_agents.mkdir()
    benchmark_dir = benchmarks / "desktop_bench"
    benchmark_dir.mkdir(parents=True)
    agent_dir = responses_api_agents / "desktop_agent" / "configs"
    agent_dir.mkdir(parents=True)
    agent_config = agent_dir / "desktop_agent.yaml"
    agent_config.write_text(
        """
desktop_agent:
  responses_api_agents:
    desktop_agent:
      domain: agent
      description: Desktop benchmark
      value: GUI evaluation
""".lstrip(),
        encoding="utf-8",
    )
    (benchmark_dir / "config.yaml").write_text(
        f"""
config_paths:
  - {agent_config}

desktop_agent:
  responses_api_agents:
    desktop_agent:
      datasets:
        - name: validation
          type: validation
          license: Apache 2.0
          huggingface_identifier:
            repo_id: example/desktop-bench
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(update_env_list, "RESOURCES_SERVERS_FOLDER", resources_servers)
    monkeypatch.setattr(update_env_list, "RESPONSES_API_AGENTS_FOLDER", responses_api_agents)
    monkeypatch.setattr(update_env_list, "BENCHMARKS_FOLDER", benchmarks)

    servers = update_env_list.get_training_server_info()

    assert len(servers) == 1
    server = servers[0]
    assert server.name == "desktop_bench"
    assert server.display_name == "Desktop Agent"
    assert server.config_path == "benchmarks/desktop_bench/config.yaml"
    assert server.readme_path == "benchmarks/desktop_bench/README.md"
    assert server.config_metadata.domain == "agent"
    assert server.config_metadata.description == "Desktop benchmark"
    assert server.config_metadata.value == "GUI evaluation"
    assert server.config_metadata.types == ["validation"]


def _prepare_readme_hook_repository(directory: Path) -> None:
    """Create a catalog large enough to exercise pre-commit filename batching."""
    root = Path(__file__).resolve().parents[3]
    configuration = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hook = next(h for repo in configuration["repos"] for h in repo["hooks"] if h["id"] == "update-readme-table")
    runner = directory / "generate_readme.py"
    runner.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "from scripts.update_env_list import main\n"
        "with Path('calls.jsonl').open('a') as log:\n"
        "    print(json.dumps(sys.argv[1:]), file=log)\n"
        "main()\n"
    )
    hook.update(language="system", entry=shlex.join([sys.executable, str(runner)]))
    hook.pop("additional_dependencies", None)
    (directory / ".pre-commit-config.yaml").write_text(yaml.safe_dump({"repos": [{"repo": "local", "hooks": [hook]}]}))
    (directory / "responses_api_agents").mkdir()
    configs = directory / "resources_servers" / "hook_example" / "configs"
    configs.mkdir(parents=True)
    for number in range(128):
        (configs / f"example_{number}.yaml").write_text(
            yaml.safe_dump(
                {"hook_example": {"resources_servers": {"hook_example": {"domain": "math"}}}}
                | _agent_config({"name": "train", "type": "train", "license": "Apache 2.0"})
            )
        )
    for arguments in (["init", "--quiet"], ["add", "."]):
        subprocess.run(["git", *arguments], cwd=directory, check=True, capture_output=True, timeout=30)


def test_readme_hook_generates_once_without_filenames_and_is_idempotent(tmp_path: Path) -> None:
    """Run the configured hook twice and check dispatch, generated content, and stability."""
    readme = tmp_path / "README.md"
    prefix = "# Catalog\n\n<!-- START_TRAINING_SERVERS_TABLE -->\n"
    suffix = "<!-- END_TRAINING_SERVERS_TABLE -->\n\nFooter stays here.\n"
    readme.write_text(prefix + "outdated table\n" + suffix)
    _prepare_readme_hook_repository(tmp_path)
    command = [sys.executable, "-m", "pre_commit", "run", "update-readme-table", "--all-files", "--color=never"]
    environment = {**os.environ, "PRE_COMMIT_HOME": str(tmp_path / "hook-cache")}
    first = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert first.returncode == 1, first.stdout + first.stderr
    assert "files were modified by this hook" in first.stdout
    generated = readme.read_text()
    assert generated.startswith(prefix)
    assert generated.endswith(suffix)
    assert "Hook Example" in generated
    assert "outdated table" not in generated
    assert [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()] == [[]]

    second = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert second.returncode == 0, second.stdout + second.stderr
    assert readme.read_text() == generated
    assert [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()] == [[], []]
