# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verifier tests: the verify() adapter logic, reward correctness (FakeSandbox),
and a real docker-backed end-to-end (env-gated so CI never runs it)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import responses_api_agents.swe_env.harnesses  # noqa: F401  (register harnesses)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus, register_provider
from resources_servers.swe_env.app import _as_list, _item_text, build_task, extract_patch
from resources_servers.swe_env.verify_task import verify_task
from responses_api_agents.swe_env.grading import reward_from_report
from responses_api_agents.swe_env.harness import SweTask


# ----- verify() adapter logic (no HTTP / no pydantic construction needed) -----


def test_as_list():
    """Check that _as_list normalizes None, lists, JSON strings, and bare strings."""
    assert _as_list(None) == []
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list('["x", "y"]') == ["x", "y"]
    assert _as_list("single") == ["single"]


def test_extract_patch_from_metadata_field():
    """Check that extract_patch reads the patch from the named response metadata field."""
    response = SimpleNamespace(metadata={"model_patch": "diff --git a/x b/x\n"}, output=[])
    assert extract_patch(response, {}, "model_patch") == "diff --git a/x b/x\n"


def test_extract_patch_from_fenced_diff():
    """Check that extract_patch recovers a diff from a fenced ```diff block in output."""
    item = SimpleNamespace(content="here:\n```diff\n--- a/x\n+++ b/x\n```\n")
    response = SimpleNamespace(metadata={}, output=[item])
    assert "--- a/x" in extract_patch(response, {}, "model_patch")


def test_task_from_request_maps_metadata():
    """Check that build_task maps request metadata and the response patch onto a SweTask."""
    body = SimpleNamespace(
        responses_create_params=SimpleNamespace(
            metadata={
                "instance_id": "abc",
                "image": "img:tag",
                "base_commit": "deadbeef",
                "test_command": "python -m pytest -rA -q",
                "fail_to_pass": '["t::a"]',
                "pass_to_pass": ["t::b"],
                "benchmark": "swe-bench-ext",
            }
        ),
        response=SimpleNamespace(metadata={"model_patch": "diff\n"}, output=[]),
    )
    task = build_task(body, "model_patch")
    assert task.instance_id == "abc"
    assert task.image == "img:tag"
    assert task.fail_to_pass == ["t::a"]
    assert task.pass_to_pass == ["t::b"]
    assert task.model_patch == "diff\n"


def test_build_task_reads_uppercase_fail_pass_to_pass():
    """Check that build_task falls back to UPPERCASE FAIL_TO_PASS / PASS_TO_PASS keys."""
    # A row carrying only the UPPERCASE FAIL_TO_PASS / PASS_TO_PASS keys must still
    # populate the required test lists; otherwise the subset rule (empty subset <= any
    # set) reports resolved=True for every sample -> reward inflation.
    body = SimpleNamespace(
        responses_create_params=SimpleNamespace(
            metadata={
                "instance_id": "abc",
                "FAIL_TO_PASS": '["tests/test_x.py::a"]',
                "PASS_TO_PASS": ["tests/test_x.py::b"],
            }
        ),
        response=SimpleNamespace(metadata={"model_patch": "diff\n"}, output=[]),
    )
    task = build_task(body, "model_patch")
    assert task.fail_to_pass == ["tests/test_x.py::a"]
    assert task.pass_to_pass == ["tests/test_x.py::b"]


def test_build_task_lowercase_wins_over_uppercase():
    """Check that lowercase fail_to_pass / pass_to_pass win when both casings are set."""
    # When both casings are present, lowercase (the canonical key) takes precedence;
    # the UPPERCASE fallback only fills in when lowercase is absent/empty.
    body = SimpleNamespace(
        responses_create_params=SimpleNamespace(
            metadata={
                "instance_id": "abc",
                "fail_to_pass": ["lower::a"],
                "FAIL_TO_PASS": ["UPPER::a"],
                "pass_to_pass": ["lower::b"],
                "PASS_TO_PASS": ["UPPER::b"],
            }
        ),
        response=SimpleNamespace(metadata={"model_patch": "diff\n"}, output=[]),
    )
    task = build_task(body, "model_patch")
    assert task.fail_to_pass == ["lower::a"]
    assert task.pass_to_pass == ["lower::b"]


def test_build_task_unpacks_nested_instance_dict():
    """Check that build_task surfaces nested instance_dict keys while top-level wins."""
    # build_task must surface keys nested in a stringified ``instance_dict``
    # (fail_to_pass_select / base_dockerfile / etc.), with explicit top-level
    # metadata winning on conflict.
    body = SimpleNamespace(
        responses_create_params=SimpleNamespace(
            metadata={
                "instance_id": "i",
                "benchmark": "nv-internal-1",
                "instance_dict": json.dumps(
                    {
                        "fail_to_pass_select": '["sel::a"]',
                        "base_dockerfile": "ENV FOO=bar",
                        "fail_to_pass": '["nested::a"]',
                    }
                ),
                "fail_to_pass": '["toplevel::a"]',
            }
        ),
        response=SimpleNamespace(metadata={"model_patch": "diff\n"}, output=[]),
    )
    task = build_task(body, "model_patch")
    assert task.metadata["fail_to_pass_select"] == '["sel::a"]'  # nested key surfaced
    assert task.metadata["base_dockerfile"] == "ENV FOO=bar"
    assert task.metadata["fail_to_pass"] == '["toplevel::a"]'  # explicit top-level wins


def test_item_text_handles_list_content():
    """Check that _item_text joins the text of a list-valued content field with newlines."""
    item = SimpleNamespace(content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")])
    assert _item_text(item) == "a\nb"


# ----- reward correctness (FakeSandbox) ---------------------------------------


class _FakeProvider:
    """In-memory sandbox provider that returns canned pytest output for tests."""

    name = "fake-verify"

    def __init__(self, *, test_output="", test_rc=0, **_):
        """Store the canned test output and return code this provider replays.

        Args:
            test_output: Stdout returned for any command containing ``pytest``.
            test_rc: Return code returned for any command containing ``pytest``.
        """
        self._test_output = test_output
        self._test_rc = test_rc

    async def create(self, spec):
        """Create a fake sandbox handle.

        Args:
            spec: Sandbox spec whose workdir is echoed back in the handle.

        Returns:
            SandboxHandle: A handle naming this provider and the spec workdir.
        """
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        """Replay the canned output for pytest commands; return success otherwise.

        Args:
            handle: Sandbox handle (unused).
            command: Command string; pytest commands replay the canned output.
            cwd: Working directory (unused).
            env: Environment variables (unused).
            timeout_s: Execution timeout (unused).
            user: User to run as (unused).

        Returns:
            SandboxExecResult: The canned pytest result, or an empty success result.
        """
        if "pytest" in command:
            return SandboxExecResult(stdout=self._test_output, stderr="", return_code=self._test_rc)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, *a, **k):
        """Accept an upload request and do nothing."""
        return None

    async def download_file(self, *a, **k):
        """Accept a download request and do nothing."""
        return None

    async def status(self, handle):
        """Report the sandbox as running.

        Args:
            handle: Sandbox handle (unused).

        Returns:
            SandboxStatus: Always ``RUNNING``.
        """
        return SandboxStatus.RUNNING

    async def close(self, handle):
        """Close the given sandbox handle and do nothing.

        Args:
            handle: Sandbox handle (unused).
        """
        return None

    async def aclose(self):
        """Close the provider and do nothing."""
        return None


register_provider("fake-verify", _FakeProvider, override=True)


def _task(**kw) -> SweTask:
    """Build a SweTask with sensible defaults, overridable by keyword.

    Args:
        **kw: Field overrides applied on top of the default task fields.

    Returns:
        SweTask: The constructed task.
    """
    base = dict(
        instance_id="i",
        image="img:tag",
        base_commit="HEAD",
        test_command="python -m pytest -rA -q",
        model_patch="diff --git a/x b/x\n",
        # Trailing-status pytest text (``<node_id> PASSED``) is the format the
        # lighthouse parser recognizes; the ``.py`` path normalizes to this F2P id.
        test_framework="pytest",
        fail_to_pass=["tests/test_x.py::a"],
        benchmark="swe-bench-ext",
    )
    base.update(kw)
    return SweTask(**base)


def test_reward_gold_patch_resolves():
    """Check that a gold patch with passing F2P tests grades to reward 1.0."""
    report = asyncio.run(verify_task({"fake-verify": {"test_output": "tests/test_x.py::a PASSED\n"}}, _task()))
    assert reward_from_report(report) == 1.0


def test_reward_noop_patch_unresolved():
    """Check that an empty (no-op) patch grades to reward 0.0."""
    report = asyncio.run(verify_task({"fake-verify": {}}, _task(model_patch="")))
    assert reward_from_report(report) == 0.0


def test_reward_failing_tests_unresolved():
    """Check that a patch whose F2P tests fail grades to reward 0.0."""
    report = asyncio.run(
        verify_task({"fake-verify": {"test_output": "tests/test_x.py::a FAILED\n", "test_rc": 1}}, _task())
    )
    assert reward_from_report(report) == 0.0


# ----- REAL docker-backed end-to-end (env-gated; never runs in CI) ------------

_RUN_DOCKER = os.environ.get("SWE_ENV_DOCKER_ITEST") == "1" and shutil.which("docker") is not None

_DOCKERFILE = """FROM python:3.11
RUN pip install --no-cache-dir pytest
WORKDIR /testbed
RUN git config --global user.email a@b.c && git config --global user.name t \\
 && git init -q \\
 && printf 'def add(a, b):\\n    return a - b\\n' > calc.py \\
 && printf 'from calc import add\\n\\n\\ndef test_add():\\n    assert add(1, 2) == 3\\n' > test_calc.py \\
 && git add -A && git commit -q -m base
"""

_GOLD_PATCH = """--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""

_IMAGE_TAG = "swe-env-itest:local"


@pytest.mark.skipif(not _RUN_DOCKER, reason="set SWE_ENV_DOCKER_ITEST=1 and install docker to run")
def test_docker_real_end_to_end():
    """Build a tiny real git repo image; gold patch resolves, empty patch does not."""
    build = subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, "-f", "-", "."],
        input=_DOCKERFILE.encode(),
        capture_output=True,
    )
    assert build.returncode == 0, build.stderr.decode(errors="replace")[-2000:]

    task = SweTask(
        instance_id="calc-1",
        image=_IMAGE_TAG,
        base_commit="HEAD",
        repo_workdir="/testbed",
        # Emit JUnit XML to stdout so the lighthouse parser (junit path for
        # pytest) can grade it host-side.
        test_command="python -m pytest -q --junit-xml=/dev/stdout",
        test_framework="pytest",
        model_patch=_GOLD_PATCH,
        fail_to_pass=["test_calc.py::test_add"],
        benchmark="swe-bench-ext",
    )

    gold_report = asyncio.run(verify_task({"docker": {}}, task))
    sys.stderr.write(f"\n[itest] gold report: {gold_report}\n")
    assert gold_report.patch_applied is True
    assert gold_report.resolved is True
    assert reward_from_report(gold_report) == 1.0

    import dataclasses

    empty_report = asyncio.run(verify_task({"docker": {}}, dataclasses.replace(task, model_patch="")))
    assert empty_report.resolved is False
    assert reward_from_report(empty_report) == 0.0
