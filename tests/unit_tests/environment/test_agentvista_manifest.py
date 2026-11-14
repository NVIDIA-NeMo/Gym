# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentVista manifest validation must select its resource verifier fixtures."""

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_manifest_uses_resource_verifier_fixtures():
    from nemo_gym.environment.onboarding import exercise_verifier_run, prepare_verifier_run
    from nemo_gym.environment.registry import manifest_workloads_for_server, resolve_catalog_entry
    from nemo_gym.environment.validation import validate_environment

    entry = resolve_catalog_entry("agentvista", "benchmark")
    assert entry.resources_server == "agentvista"
    server_dir = Path(__file__).resolve().parents[3] / "resources_servers/agentvista"
    assert entry in manifest_workloads_for_server("agentvista", server_dir)
    validation = validate_environment(entry.manifest_path, entry.config_path, validate_datasets=False)
    report = await exercise_verifier_run(prepare_verifier_run(entry, validation))
    assert report.resources_server == "agentvista"
    assert {case.kind for case in report.cases} == {"full_reward", "zero_reward", "malformed"}
    assert len(report.cases) == 4
