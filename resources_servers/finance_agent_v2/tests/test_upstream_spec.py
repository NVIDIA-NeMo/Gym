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


_SERVER_DIR = Path(__file__).resolve().parents[1]
_SCRIPT_FPATH = _SERVER_DIR / "scripts" / "export_upstream_spec.py"


def _load_exporter():
    """Import the exporter by path: `scripts/` is a directory of standalone CLI
    scripts, not an importable package (no server in this repo makes it one)."""
    spec = importlib.util.spec_from_file_location("export_upstream_spec", _SCRIPT_FPATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    return _load_exporter()


@pytest.fixture(scope="module")
def committed(exporter) -> dict:
    return json.loads(exporter.SPEC_FPATH.read_text(encoding="utf-8"))


class TestSnapshotIsCurrent:
    def test_committed_snapshot_matches_installed_package(self, exporter) -> None:
        """A mismatch means samples advertise a tool signature the installed tools do
        not implement."""
        expected = exporter.serialize(exporter.build_spec())
        actual = exporter.SPEC_FPATH.read_text(encoding="utf-8")
        assert actual == expected, (
            f"{exporter.SPEC_FPATH} is stale relative to the installed finance_agent package. "
            "Re-run scripts/export_upstream_spec.py from this venv."
        )

    def test_check_mode_agrees(self, exporter) -> None:
        # The --check path is what a human (or CI) runs; keep it wired to the same
        # comparison rather than letting it rot into a no-op.
        assert exporter.main(["--check"]) == 0

    def test_snapshot_commit_matches_the_benchmark_pin(self, committed) -> None:
        """prepare.py refuses to run when these disagree; catch it here, where the
        failure names the cause."""
        from benchmarks.finance_agent_v2 import prepare as prepare_module

        assert committed["upstream_commit_id"] == prepare_module._UPSTREAM_SHA

    def test_snapshot_covers_every_upstream_tool(self, committed) -> None:
        """A tool upstream adds must reach the dataset; without it the agent simply
        cannot call it, which reads as a weak model."""
        from finance_agent.tools import VALID_TOOLS

        assert set(committed["valid_tools"]) == set(VALID_TOOLS)
        assert set(committed["tools"]) == set(VALID_TOOLS) | {committed["submit_tool"]}

    def test_every_schema_is_a_usable_function_tool(self, committed) -> None:
        # Guards against a tool class losing an attribute and exporting an empty
        # description or parameter block, which the responses API accepts silently.
        for name, schema in committed["tools"].items():
            assert schema["type"] == "function", name
            assert schema["name"] == name
            assert schema["description"].strip(), f"{name} has no description"
            assert schema["parameters"]["type"] == "object", name
            assert isinstance(schema["parameters"]["properties"], dict), name
            for required in schema["parameters"]["required"]:
                assert required in schema["parameters"]["properties"], (
                    f"{name}: required {required!r} is not a property"
                )

    def test_prompts_are_non_empty_and_take_the_question(self, committed) -> None:
        assert committed["system_prompt"].strip()
        assert "{question}" in committed["question_prompt"]


class TestExportGuards:
    def test_unknown_upstream_tool_fails_the_export(self, exporter, monkeypatch) -> None:
        """Rather than exporting a silently incomplete tool set."""
        monkeypatch.setattr(exporter, "VALID_TOOLS", list(exporter.VALID_TOOLS) + ["brand_new_tool"])

        with pytest.raises(ValueError, match="brand_new_tool"):
            exporter.build_spec()

    def test_sha_comes_from_the_requirements_pin(self, exporter, committed) -> None:
        # One source of truth for the upstream version: bumping requirements.txt is
        # what makes the snapshot stale, and nothing else.
        assert exporter._upstream_sha() == committed["upstream_commit_id"]
