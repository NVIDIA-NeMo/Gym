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

import pytest

from resources_servers.conversational_tool_use_simulation.scripts.build_conversational_tool_use_dataset import (
    build_sample_dataset,
)
from resources_servers.conversational_tool_use_simulation.scripts.build_full_conversational_tool_use_datasets import (
    BuildJob,
    _existing_output_is_current,
)


def _write_source(source_dir: Path) -> Path:
    domain_dir = source_dir / "0"
    scenario_dir = domain_dir / "scenarios" / "run"
    scenario_dir.mkdir(parents=True)
    (domain_dir / "policy.md").write_text("Authenticate before account changes.", encoding="utf-8")
    (domain_dir / "tools.jsonl").write_text(
        json.dumps(
            {
                "name": "lookup_account",
                "doc": "Look up an account.",
                "params": {"type": "object", "properties": {}},
                "returns": {"type": "object", "properties": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "scenarios_0.jsonl").write_text(
        json.dumps(
            {
                "customer_persona": "A customer",
                "reason_for_contact": "Check an account.",
                "customer_details": "Account A-1",
                "unknown_info": None,
                "task_instructions": "Ask for help.",
                "representative_domain": "account support",
                "outside_policy_scope": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return domain_dir


def _build_current_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BuildJob, Path, Path, Path]:
    source_dir = tmp_path / "source"
    _write_source(source_dir)
    monkeypatch.setenv("CONVERSATIONAL_TOOL_USE_GENERAL_SOURCE_DIR", str(source_dir))
    job = BuildJob(
        key="general",
        dataset_name="general",
        source_indexes=(0,),
        source_names=("general-source",),
        source_profiles=("general",),
        parallel_tool_calls=False,
    )
    output_path = tmp_path / "general.jsonl"
    report_path = tmp_path / "general.report.json"
    build_sample_dataset(
        source_dirs=[source_dir],
        output_path=output_path,
        report_path=report_path,
        max_rows=None,
        dataset_name=job.dataset_name,
        source_names=list(job.source_names),
        source_profiles=list(job.source_profiles),
        max_rows_per_domain=None,
        scan_domains_per_source=None,
        parallel_tool_calls=job.parallel_tool_calls,
    )
    return job, output_path, report_path, source_dir


def test_skip_existing_requires_current_contract_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, output_path, report_path, _ = _build_current_dataset(tmp_path, monkeypatch)
    assert _existing_output_is_current(job, output_path, report_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_profiles"] = ["proactive"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _existing_output_is_current(job, output_path, report_path)


def test_skip_existing_rejects_different_dataset_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, output_path, report_path, _ = _build_current_dataset(tmp_path, monkeypatch)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    report["metadata"]["dataset_name"] = "other"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _existing_output_is_current(job, output_path, report_path)

    report["metadata"]["dataset_name"] = job.dataset_name
    report["source_names"] = ["other-source"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not _existing_output_is_current(job, output_path, report_path)


def test_skip_existing_rejects_corrupt_output_or_changed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job, output_path, report_path, source_dir = _build_current_dataset(tmp_path, monkeypatch)
    original_output = output_path.read_text(encoding="utf-8")

    output_path.write_text(original_output + "not-json\n", encoding="utf-8")
    assert not _existing_output_is_current(job, output_path, report_path)

    output_path.write_text(original_output, encoding="utf-8")
    (source_dir / "0" / "policy.md").write_text("A changed policy.", encoding="utf-8")
    assert not _existing_output_is_current(job, output_path, report_path)
