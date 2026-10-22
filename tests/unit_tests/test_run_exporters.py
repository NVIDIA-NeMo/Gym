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
from pathlib import Path

import pytest

from nemo_gym.atif_export import ExportAtifConfig
from nemo_gym.config_types import ConfigError
from nemo_gym.exporters import load_run_exporter_class
from nemo_gym.exporters.atif import AtifExporter
from nemo_gym.exporters.base import ExportResult, ExportState
from nemo_gym.run_artifacts import RunArtifacts


def test_export_result_constructors() -> None:
    assert ExportResult.succeeded(n=1) == ExportResult(ExportState.SUCCEEDED, 0, None, {"n": 1})
    assert ExportResult.skipped().state is ExportState.SKIPPED
    failed = ExportResult.failed("boom", exit_code=3)
    assert (failed.state, failed.exit_code, failed.error) == (ExportState.FAILED, 3, "boom")


def test_registry_loads_atif_and_rejects_unknown_names() -> None:
    assert load_run_exporter_class("atif") is AtifExporter
    with pytest.raises(ConfigError, match="Unknown exporter 'nope'. Available exporters: atif"):
        load_run_exporter_class("nope")


def test_atif_exporter_reports_a_conversion_failure_instead_of_raising(tmp_path: Path) -> None:
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("not json\n")
    config = ExportAtifConfig(
        rollouts_jsonl_fpath=tmp_path / "ignored.jsonl",
        output_dirpath=tmp_path / "atif",
        session_id="s",
        agent_version="1",
    )

    result = AtifExporter(config).export(RunArtifacts(rollouts))

    assert result.state is ExportState.FAILED
    assert result.exit_code == 1
    assert "invalid JSON" in result.error
    assert not (tmp_path / "atif").exists()


def test_atif_exporter_reads_the_rollouts_from_the_run_artifacts(tmp_path: Path) -> None:
    config = ExportAtifConfig(
        rollouts_jsonl_fpath=tmp_path / "ignored.jsonl",
        output_dirpath=tmp_path / "atif",
        session_id="s",
        agent_version="1",
    )

    result = AtifExporter(config).export(RunArtifacts(tmp_path / "missing.jsonl"))

    assert result.state is ExportState.FAILED
    assert "missing.jsonl" in result.error
