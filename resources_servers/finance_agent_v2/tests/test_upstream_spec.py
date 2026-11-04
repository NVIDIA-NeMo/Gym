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
"""Drift guard for benchmarks/finance_agent_v2/upstream_spec.json.

The benchmark reads the upstream prompts and tool schemas from a committed snapshot
rather than importing `finance_agent`, because `gym eval prepare` runs in the repo-root
venv where that package is absent. An import cannot go stale but a file can, so these
tests re-derive the snapshot from the installed package and compare. They live here
because this is the venv that has it.

On failure, re-export rather than editing the JSON:
    python scripts/export_upstream_spec.py
"""

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT_FPATH = Path(__file__).resolve().parents[1] / "scripts" / "export_upstream_spec.py"


@pytest.fixture(scope="module")
def exporter():
    """Import the exporter by path: `scripts/` is a directory of standalone CLI
    scripts, not an importable package (no server in this repo makes it one)."""
    spec = importlib.util.spec_from_file_location("export_upstream_spec", _SCRIPT_FPATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed(exporter) -> dict:
    return json.loads(exporter.SPEC_FPATH.read_text(encoding="utf-8"))


def test_committed_snapshot_matches_installed_package(exporter) -> None:
    """A mismatch means samples advertise a tool signature the installed tools do
    not implement. Byte equality, so it also pins the prompts and every schema."""
    assert exporter.SPEC_FPATH.read_text(encoding="utf-8") == exporter.serialize(exporter.build_spec()), (
        f"{exporter.SPEC_FPATH} is stale relative to the installed finance_agent package. "
        "Re-run scripts/export_upstream_spec.py from this venv."
    )


def test_one_upstream_version_across_the_pin_snapshot_and_dataset(exporter, committed) -> None:
    """Bumping the project dependency pin is what makes the snapshot stale, and nothing
    else. prepare.py refuses to run when it disagrees; catch it here, where the
    failure names the cause."""
    from benchmarks.finance_agent_v2 import prepare as prepare_module

    assert exporter._upstream_sha() == committed["upstream_commit_id"] == prepare_module._UPSTREAM_SHA


def test_snapshot_covers_every_upstream_tool(committed) -> None:
    """A tool upstream adds must reach the dataset; without it the agent simply
    cannot call it, which reads as a weak model."""
    from finance_agent.tools import VALID_TOOLS

    assert set(committed["valid_tools"]) == set(VALID_TOOLS)
    assert set(committed["tools"]) == set(VALID_TOOLS) | {committed["submit_tool"]}


def test_unknown_upstream_tool_fails_the_export(exporter, monkeypatch) -> None:
    """Rather than exporting a silently incomplete tool set."""
    monkeypatch.setattr(exporter, "VALID_TOOLS", list(exporter.VALID_TOOLS) + ["brand_new_tool"])

    with pytest.raises(ValueError, match="brand_new_tool"):
        exporter.build_spec()


@pytest.mark.parametrize("dependency_name", ["finance-agent", "finance_agent", "Finance-Agent"])
def test_upstream_sha_reads_only_the_direct_project_dependency(
    exporter, monkeypatch, tmp_path, dependency_name
) -> None:
    """Unrelated TOML metadata must not supply or duplicate the upstream pin."""
    sha = "a" * 40
    project = tmp_path / "pyproject.toml"
    project.write_text(
        "[project]\ndependencies = [\n"
        f'  "{dependency_name} @ git+https://github.com/vals-ai/finance-agent-v2.git@{sha}",\n'
        '  "model-library==0.1.25",\n]\n'
        '[tool.snapshot]\nexample = "finance-agent-v2.git@' + "b" * 40 + '"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter, "PROJECT_FPATH", project)

    assert exporter._upstream_sha() == sha


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        ["finance-agent==0.1.0"],
        ["finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@main"],
        ["finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@bb460b52"],
        ["finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@" + "a" * 41],
        ["finance-agent @ git+https://github.com/other/finance-agent-v2.git@" + "a" * 40],
        ["finance-agent @ git+https://github.com/vals-ai/finance-agent-v2.git@" + "a" * 40] * 2,
    ],
)
def test_upstream_sha_rejects_missing_ambiguous_or_unpinned_dependencies(
    exporter, monkeypatch, tmp_path, dependencies
) -> None:
    project = tmp_path / "pyproject.toml"
    project.write_text("[project]\ndependencies = " + json.dumps(dependencies) + "\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "PROJECT_FPATH", project)

    with pytest.raises(ValueError, match="expected .*finance-agent"):
        exporter._upstream_sha()
