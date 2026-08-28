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
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.swebench_pro.verification import (
    VerificationInputs,
    apply_opensandbox_runtime_parity,
    assemble_workspace_files,
    create_entryscript,
    grade_output,
    parse_string_list,
    run_verification,
    strip_binary_hunks,
)


def make_inputs(**overrides) -> VerificationInputs:
    values = {
        "instance_id": "instance_test",
        "base_commit": "abc123",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        "run_script": "#!/bin/bash\nexit 0\n",
        "parser_script": "import json\n",
        "selected_test_files_to_run": '["a.py", "b.py"]',
        "fail_to_pass": '["test_new"]',
        "pass_to_pass": '["test_old"]',
    }
    values.update(overrides)
    return VerificationInputs(**values)


def exec_result(*, return_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(return_code=return_code, stdout=stdout, stderr=stderr)


def verification_side_effects(
    output: str,
    *,
    test_stdout: str = "test stdout",
    test_stderr: str = "test stderr",
    status_codes: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0),
) -> list[SimpleNamespace]:
    return [
        exec_result(),
        exec_result(return_code=status_codes[-1], stdout="run"),
        exec_result(stdout=test_stdout),
        exec_result(stdout=test_stderr),
        *(exec_result(stdout=f"{code}\n") for code in status_codes),
        exec_result(stdout=output),
    ]


def test_parse_string_list_accepts_json_python_and_lists() -> None:
    assert parse_string_list('["a", "b"]') == ["a", "b"]
    assert parse_string_list("['a', 'b']") == ["a", "b"]
    assert parse_string_list(["a"]) == ["a"]


def test_parse_string_list_rejects_non_string_items() -> None:
    with pytest.raises(ValueError):
        parse_string_list("[1]")


def test_strip_binary_hunks_preserves_text_diffs() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n"
    )
    assert strip_binary_hunks(patch) == "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def test_create_entryscript_matches_upstream_contract() -> None:
    inputs = make_inputs(
        before_repo_set_cmd="ignored setup line\nnpm install",
        base_dockerfile="ENV FOO=bar\n",
    )
    script = create_entryscript(asdict(inputs))
    assert "git reset --hard abc123" in script
    assert "git apply -v /workspace/patch.diff" in script
    assert "npm install" in script
    assert "bash /workspace/run_script.sh a.py,b.py" in script
    assert "export FOO=bar" in script
    assert "go mod download" not in script
    assert "printf '%s\\n' \"$RESET_STATUS\" > /workspace/reset_status" in script
    assert "printf '%s\\n' \"$TEST_STATUS\" > /workspace/test_status" in script
    assert "printf '%s\\n' \"$PARSER_STATUS\" > /workspace/parser_status" in script


def test_create_entryscript_prefetches_go_modules_before_tests() -> None:
    inputs = make_inputs(
        run_script="#!/bin/bash\nset -e\ngo test ./...\n",
        repo_language="Go",
        prefetch_go_modules=True,
    )

    script = create_entryscript(asdict(inputs))

    assert "go mod download" in script
    assert script.index("git apply") < script.index("go mod download") < script.index("run_script.sh")


def test_create_entryscript_does_not_prefetch_non_go_tasks() -> None:
    inputs = make_inputs(repo_language="Python", prefetch_go_modules=True)

    assert "go mod download" not in create_entryscript(asdict(inputs))


def test_create_entryscript_scopes_runtime_environment_parity() -> None:
    flipt = create_entryscript(
        asdict(
            make_inputs(
                instance_id="instance_flipt-io__flipt-690672523398c2b6f6e4562f0bf9868664ab894f",
                runtime_parity_adaptations=True,
            )
        )
    )
    nodebb = create_entryscript(
        asdict(
            make_inputs(
                instance_id=("instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"),
                runtime_parity_adaptations=True,
            )
        )
    )
    unrelated_flipt = create_entryscript(
        asdict(
            make_inputs(
                instance_id="instance_flipt-io__flipt-another-task",
                runtime_parity_adaptations=True,
            )
        )
    )
    unrelated_nodebb = create_entryscript(
        asdict(
            make_inputs(
                instance_id="instance_NodeBB__NodeBB-another-task-vnan",
                runtime_parity_adaptations=True,
            )
        )
    )

    assert 'OTEL_*) unset "$variable"' in flipt
    assert "dns-result-order=ipv4first" in nodebb
    assert "OTEL_" not in unrelated_flipt
    assert "dns-result-order=ipv4first" not in unrelated_nodebb


def test_runtime_parity_waits_for_nodebb_redis_pong() -> None:
    run_script = """#!/bin/bash
set -e
  while ! redis-cli ping; do
    echo "Waiting for Redis to start..."
    sleep 1
  done
"""

    adapted = apply_opensandbox_runtime_parity(
        "instance_NodeBB__NodeBB-eb49a64974ca844bca061744fb3383f5d13b02ad-vnan",
        run_script,
    )

    assert 'until [ "$(redis-cli ping 2>/dev/null)" = "PONG" ]' in adapted
    assert "while ! redis-cli ping" not in adapted
    assert apply_opensandbox_runtime_parity("instance_NodeBB__NodeBB-another-task-vnan", run_script) == run_script


def test_runtime_parity_starts_qutebrowser_xvfb_explicitly() -> None:
    run_script = "#!/bin/bash\nset -e\nexport PYTEST_QT_API=pyqt5\npytest tests/unit\n"

    adapted = apply_opensandbox_runtime_parity(
        "instance_qutebrowser__qutebrowser-f631cd4422744160d9dcf7a0455da532ce973315"
        "-v35616345bb8052ea303186706cec663146f0f184",
        run_script,
    )

    assert "Xvfb :99" in adapted
    assert "pytest -p no:xvfb tests/unit" in adapted
    assert (
        apply_opensandbox_runtime_parity(
            "instance_qutebrowser__qutebrowser-another-task",
            run_script,
        )
        == run_script
    )


def test_assemble_workspace_files_embeds_prepared_assets() -> None:
    inputs = make_inputs(patch="diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n")
    files, entryscript = assemble_workspace_files(inputs.instance_id, None, inputs.patch, asdict(inputs))

    assert files["patch.diff"] == ""
    assert files["run_script.sh"] == inputs.run_script
    assert files["parser.py"] == inputs.parser_script
    assert files["entryscript.sh"] == entryscript


def test_grade_output_requires_all_named_tests() -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    assert grade_output(output, asdict(make_inputs()))
    assert not grade_output(output, asdict(make_inputs(fail_to_pass='["missing"]')))
    assert grade_output(output, asdict(make_inputs(fail_to_pass="[]", pass_to_pass="[]")))


@pytest.mark.asyncio
async def test_run_verification_returns_resolved_result(tmp_path) -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=verification_side_effects(json.dumps(output))))

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert result.completed
    assert result.resolved
    assert result.patch_applied
    assert result.test_exit_code == 0
    assert result.parser_exit_code == 0
    assert result.test_output == "STDOUT:\ntest stdout\n\nSTDERR:\ntest stderr"
    assert json.loads((tmp_path / "output.json").read_text()) == output


@pytest.mark.asyncio
async def test_run_verification_rejects_malformed_parser_output(tmp_path) -> None:
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=verification_side_effects(
                "not-json",
                test_stdout="",
                test_stderr="failed",
                status_codes=(0, 0, 1, 0, 0, 0, 1, 1),
            )
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert not result.completed
    assert not result.resolved
    assert result.test_output == "STDOUT:\n\n\nSTDERR:\nfailed"
    assert "invalid JSON" in result.error


@pytest.mark.asyncio
async def test_run_verification_marks_setup_failure_incomplete(tmp_path) -> None:
    output = {"tests": [{"name": "test_new", "status": "PASSED"}, {"name": "test_old", "status": "PASSED"}]}
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=verification_side_effects(
                json.dumps(output),
                status_codes=(0, 0, 0, 0, 1, 0, 0, 0),
            )
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert not result.completed
    assert not result.resolved
    assert result.test_setup_exit_code == 1
    assert result.error == "Failed verification phase(s): test setup"
