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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from resources_servers.swebench_pro.app import SWEBenchProInstanceRequest, SWEBenchProResourcesServerConfig
from resources_servers.swebench_pro.verification import VerificationResult, grade_output
from resources_servers.swebench_pro_v2.app import SWEBenchProV2InstanceRequest, SWEBenchProV2ResourcesServer
from resources_servers.swebench_pro_v2.task_data import TaskData
from resources_servers.swebench_pro_v2.verification import (
    TEST_FILES,
    VerificationInputsV2,
    assemble_workspace_files,
    create_entryscript,
)


POLICY = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "model.example.com"}]}


def instance_request() -> SWEBenchProV2InstanceRequest:
    return SWEBenchProV2InstanceRequest.model_validate(
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


@pytest.mark.parametrize("field", ["docker_image", "test_sh", "test_config"])
def test_missing_or_non_string_v2_fields_are_rejected_with_the_field_name(field: str) -> None:
    row = instance_request().model_dump()
    for bad in (None, 1):
        row[field] = bad
        for model in (SWEBenchProV2InstanceRequest, TaskData):
            with pytest.raises(ValueError, match=field):
                model.model_validate(row)
    del row[field]
    with pytest.raises(ValueError, match=field):
        SWEBenchProV2InstanceRequest.model_validate(row)


def test_task_data_requires_exactly_what_the_runtime_request_requires() -> None:
    required = lambda model: {name for name, info in model.model_fields.items() if info.is_required()}  # noqa: E731

    assert required(TaskData) - {"evaluator_commit", "dataset_revision"} == required(SWEBenchProV2InstanceRequest)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [("/testbed\n", "/testbed"), ("/app\n", "/app"), ("", "/app")],
)
@pytest.mark.asyncio
async def test_repository_dir_follows_the_git_repository(stdout: str, expected: str) -> None:
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(return_value=SimpleNamespace(stdout=stdout))

    assert await make_server()._repository_dir(sandbox) == expected
    assert "/app/.git" in sandbox.exec.call_args.args[0]


@pytest.mark.asyncio
async def test_patch_extraction_and_pristine_snapshot_use_the_repository_dir(tmp_path: Path) -> None:
    commands: list[str] = []

    async def exec_(command: str, **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(return_code=0, stdout="/testbed\n" if "echo /testbed" in command else "", stderr="")

    async def download(remote: str, local: Path) -> None:
        local.write_bytes(b"diff --git a/f b/f\n")

    sandbox = MagicMock(exec=exec_, download=download, stop=AsyncMock())
    server = make_server()
    server._session_id_to_sandbox["s"] = sandbox

    assert await server.pristine_untracked_files(sandbox) == frozenset()
    assert await server._extract_model_patch("s", "abc123") == "diff --git a/f b/f\n"
    repo_commands = [c for c in commands if "git -C" in c]
    assert repo_commands and all("git -C /testbed" in c and "/app" not in c for c in repo_commands)


def test_entryscript_enters_testbed_only_when_it_is_the_repository() -> None:
    script = create_entryscript({})

    assert "if [ ! -e /app/.git ] && [ -e /testbed/.git ]; then cd /testbed; else cd /app; fi" in script


def test_opencode_preset_matches_the_v2_agent_and_verifier_budgets() -> None:
    root = Path(__file__).parents[3]
    preset = yaml.safe_load((root / "benchmarks/swebench/pro_v2/opencode.yaml").read_text())
    server = yaml.safe_load((root / "resources_servers/swebench_pro_v2/configs/swebench_pro_v2.yaml").read_text())

    agent = preset["swebench_pro_v2_opencode_sandboxed_agent"]["responses_api_agents"]["opencode_sandboxed_agent"]
    verifier = server["swebench_pro_v2_resources_server"]["resources_servers"]["swebench_pro_v2"]
    assert agent["sandbox_timeout"] == verifier["evaluation_timeout"] == 3000
