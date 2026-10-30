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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest, SWEBenchProResourcesServerConfig
from resources_servers.swebench_pro.verification import VerificationResult, grade_output
from resources_servers.swebench_pro_v2.app import SWEBenchProV2ResourcesServer
from resources_servers.swebench_pro_v2.verification import (
    TEST_FILES,
    VerificationInputsV2,
    assemble_workspace_files,
    create_entryscript,
)


POLICY = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example.com"}]}


def instance_request() -> SWEBenchProInstanceRequest:
    return SWEBenchProInstanceRequest.model_validate(
        {
            "repo": "example/repo",
            "instance_id": "instance_example",
            "base_commit": "abc123",
            "patch": "gold patch",
            "test_patch": "test patch",
            "problem_statement": "Fix it",
            "fail_to_pass": '["new_test"]',
            "pass_to_pass": '["old_test"]',
            "selected_test_files_to_run": '["tests"]',
            "dockerhub_tag": "example-tag",
            "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example",
            "run_script": "#!/bin/bash\nexit 0\n",
            "parser_script": "print('{}')\n",
            "test_sh": "#!/bin/bash\nexit 0\n",
            "test_config": '{"fail_to_pass": "[]"}',
        }
    )


def make_server(**overrides: object) -> SWEBenchProV2ResourcesServer:
    config = SWEBenchProResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swebench_pro_v2_resources_server",
        sandbox_provider="test",
        sandbox_config={"provider_options": {"resource_requests": {"cpu": 1}}},
        **overrides,
    )
    return SWEBenchProV2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_entryscript_runs_upstream_test_sh_instead_of_the_v1_flow() -> None:
    script = create_entryscript({})

    assert "bash /tests/test.sh" in script
    assert "cp /workspace/test.sh /workspace/run_script.sh" in script
    assert "mkdir -p /tests /logs/verifier" in script
    assert "git reset --hard" not in script
    assert "before_repo_set_cmd" not in script
    # v1's run_verification reads these three files back.
    for path in ("patch_apply_status", "output.json", "stdout.log", "stderr.log"):
        assert f"/workspace/{path}" in script


def test_assemble_workspace_files_ships_the_files_test_sh_reads() -> None:
    inputs = VerificationInputsV2(
        instance_id="instance_example",
        base_commit="abc123",
        patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        run_script="run",
        parser_script="parse",
        selected_test_files_to_run="[]",
        fail_to_pass="[]",
        pass_to_pass="[]",
        test_sh="test.sh body",
        test_config="config body",
        test_patch="test patch body",
    )

    files, entryscript = assemble_workspace_files("instance_example", None, inputs.patch, inputs.__dict__)

    assert set(TEST_FILES) <= set(files)
    assert files["test.sh"] == "test.sh body"
    assert files["config.json"] == "config body"
    assert files["test_patch.patch"] == "test patch body"
    assert files["entryscript.sh"] == entryscript


def test_server_uses_the_dataset_image_and_carries_the_v2_files_into_the_inputs() -> None:
    server = make_server()
    body = instance_request()

    assert server._image(body) == "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example"
    assert server._image_info(body)["image"] == "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example"
    inputs = server._verification_inputs(body, "agent patch")
    assert isinstance(inputs, VerificationInputsV2)
    assert (inputs.patch, inputs.test_sh, inputs.test_patch) == ("agent patch", body.test_sh, "test patch")


def test_grade_output_matches_test_sh_required_pass_rule() -> None:
    sample = {"fail_to_pass": '["new_test"]', "pass_to_pass": '["old_test"]'}
    passed = {"tests": [{"name": "new_test", "status": "PASSED"}, {"name": "old_test", "status": "PASSED"}]}
    failed = {"tests": [{"name": "new_test", "status": "PASSED"}, {"name": "old_test", "status": "FAILED"}]}

    assert grade_output(passed, sample) is True
    assert grade_output(failed, sample) is False


@pytest.mark.parametrize(
    ("files", "expected"),
    [(None, POLICY), ({"/workspace/patch.diff": ""}, None)],
    ids=["agent_sandbox_is_restricted", "verifier_sandbox_keeps_network"],
)
@pytest.mark.asyncio
async def test_agent_network_policy_applies_to_the_agent_sandbox_only(
    monkeypatch: MonkeyPatch, files: dict | None, expected: dict | None
) -> None:
    started = []

    class FakeSandbox:
        def __init__(self, provider_config: object) -> None:
            pass

        async def start(self, spec: object) -> None:
            started.append(spec)

    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", FakeSandbox)
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench_pro.app.resolve_provider_config", lambda name, config: {})
    monkeypatch.setattr("resources_servers.swebench_pro.app.resolve_provider_metadata", lambda name, config: {})
    server = make_server(agent_network_policy=POLICY)

    await server._create_sandbox(instance_request(), files=files)

    options = started[0].provider_options
    assert options.get("network_policy") == expected
    assert options["resource_requests"] == {"cpu": 1}
