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
from pathlib import Path

import scripts.migrate_dataset_decoupling as mig


RS_AGENT_YAML = """\
math_env:
  resources_servers:
    math_env:
      entrypoint: app.py
math_env_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: math_env
      datasets:
      - name: train
        type: train
        jsonl_fpath: data/train.jsonl
      - name: bench
        type: benchmark
        jsonl_fpath: data/bench.jsonl
"""

SELF_CONTAINED_YAML = """\
tau_agent:
  responses_api_agents:
    tau:
      entrypoint: app.py
      datasets:
      - name: example
        type: example
        jsonl_fpath: data/example.jsonl
"""


def _run_config_migration(tmp_path: Path, yaml_text: str, dry_run: bool = True) -> tuple[mig.Report, Path]:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text)
    report = mig.Report(dry_run=dry_run)
    mig.migrate_config_file(path, report, pin_benchmarks=True, move_datasets=True)
    return report, path


def test_mover_moves_datasets_to_rs_block(tmp_path: Path) -> None:
    report, path = _run_config_migration(tmp_path, RS_AGENT_YAML, dry_run=False)

    cfg = mig.load_yaml(path)
    agent_cfg = cfg["math_env_simple_agent"]["responses_api_agents"]["simple_agent"]
    rs_cfg = cfg["math_env"]["resources_servers"]["math_env"]
    assert "datasets" not in agent_cfg
    assert [d["name"] for d in rs_cfg["datasets"]] == ["train", "bench"]
    assert report.datasets_moved == 2
    assert str(path) in report.configs_changed


def test_mover_pins_benchmark_agent(tmp_path: Path) -> None:
    _, path = _run_config_migration(tmp_path, RS_AGENT_YAML, dry_run=False)

    cfg = mig.load_yaml(path)
    bench = cfg["math_env"]["resources_servers"]["math_env"]["datasets"][1]
    assert bench["type"] == "benchmark"
    assert bench["agent"] == "math_env_simple_agent"


def test_mover_leaves_self_contained_agents(tmp_path: Path) -> None:
    report, path = _run_config_migration(tmp_path, SELF_CONTAINED_YAML, dry_run=False)

    cfg = mig.load_yaml(path)
    agent_cfg = cfg["tau_agent"]["responses_api_agents"]["tau"]
    assert [d["name"] for d in agent_cfg["datasets"]] == ["example"]
    assert report.datasets_moved == 0
    assert any("tau_agent" in entry for entry in report.self_contained_left)


def test_mover_dry_run_writes_nothing(tmp_path: Path) -> None:
    report, path = _run_config_migration(tmp_path, RS_AGENT_YAML, dry_run=True)

    assert report.datasets_moved == 2  # detected...
    cfg = mig.load_yaml(path)  # ...but file untouched
    assert "datasets" in cfg["math_env_simple_agent"]["responses_api_agents"]["simple_agent"]


def test_rewriter_strips_agent_ref_and_preserves_content_hashes(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [
        {
            "responses_create_params": {"input": [{"role": "user", "content": "q1"}]},
            "expected_answer": "1",
            "agent_ref": {"type": "responses_api_agents", "name": "old_agent"},
        },
        {"responses_create_params": {"input": []}, "expected_answer": "2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    report = mig.Report(dry_run=False)
    mig.rewrite_jsonl_file(path, report, fold_task_data=False)

    out = [json.loads(line) for line in path.read_text().splitlines()]
    assert all("agent_ref" not in r for r in out)
    assert out[0]["expected_answer"] == "1"
    assert report.agent_refs_stripped == 1

    manifest = report.manifests[str(path)]
    assert manifest["rows_before"] == manifest["rows_after"] == 2
    # The content hash ignores agent_ref, so the manifest proves a bijection.
    assert manifest["content_hashes_before"] == manifest["content_hashes_after"]


def test_rewriter_fold_task_data_moves_extras(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    row = {"responses_create_params": {"input": []}, "expected_answer": "42", "agent_ref": {"name": "a"}}
    path.write_text(json.dumps(row) + "\n")

    report = mig.Report(dry_run=False)
    mig.rewrite_jsonl_file(path, report, fold_task_data=True)

    out = json.loads(path.read_text())
    assert out["task_data"] == {"expected_answer": "42"}
    assert "expected_answer" not in out
    assert "agent_ref" not in out


def test_rewriter_dry_run_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    original = json.dumps({"responses_create_params": {}, "agent_ref": {"name": "a"}}) + "\n"
    path.write_text(original)

    report = mig.Report(dry_run=True)
    mig.rewrite_jsonl_file(path, report, fold_task_data=False)

    assert path.read_text() == original
    assert report.agent_refs_stripped == 1  # detected, not applied
