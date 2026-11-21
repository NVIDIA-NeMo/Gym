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
"""verify() scoring against upstream's rules, with the sandbox replaced by canned runner results."""

import asyncio
import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import resources_servers.baxbench.app as app_module
from nemo_gym.failure_kinds import PROVIDER_UNAVAILABLE, VERIFIER_ERROR
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_utils.sandbox_verification import SandboxVerificationError
from resources_servers.baxbench.app import (
    BaxBenchResourcesServer,
    BaxBenchResourcesServerConfig,
    BaxBenchVerifyRequest,
)


_META = {
    "task_id": "Python-Flask/Calculator",
    "env_id": "Python-Flask",
    "image_slug": "python-flask",
    "scenario_id": "Calculator",
    "language": "Python",
    "framework": "Flask",
    "is_multi_file": False,
    "code_filename": "app.py",
    "functional_tests": ["func_a", "func_b"],
    "security_tests": ["sec_a", "sec_b"],
    "upstream_sha": "de885cd93d561682e203a5a0d57c33b08aac6f5e",
}
_ANSWER = "<think>plan</think>Here it is:\n```python\nprint('server')\n```"


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _request(text: str = _ANSWER) -> BaxBenchVerifyRequest:
    return BaxBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "q"}]},
        response=_response(text),
        verifier_metadata=_META,
    )


def _ok(**overrides) -> dict:
    result = {"build_failed": False, "server_started": True, "passed": True, "had_exception": False}
    return result | {"cwes": [], "crash_cwes": []} | overrides


@pytest.fixture
def server() -> BaxBenchResourcesServer:
    server = BaxBenchResourcesServer(
        config=BaxBenchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="baxbench",
            sandbox_provider={"fake": {}},
            image_template="registry.example/baxbench-{env}:de885cd9",
            inconclusive_verification_retries=1,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    server._sandbox_target = ({"fake": {}}, {})
    return server


def _fake_sandbox(monkeypatch, results: dict) -> list[dict]:
    """Route each (kind, test) to a canned result, or raise it when it is an exception."""
    calls = []

    async def fake(**kwargs):
        calls.append(kwargs)
        outcome = results[(kwargs["request"]["test_kind"], kwargs["request"]["test_name"])]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(app_module, "run_verification_sandbox", fake)
    return calls


def _all(result: dict) -> dict:
    return {
        (kind, name): result for kind in ("functional", "security") for name in ("func_a", "func_b", "sec_a", "sec_b")
    }


def test_correct_and_secure_scores_one(server, monkeypatch):
    calls = _fake_sandbox(monkeypatch, _all(_ok()))
    out = asyncio.run(server.verify(_request()))
    assert out.reward == 1.0 and out.functional_pass and out.secure_pass and not out.insecure
    assert len(calls) == 4
    first = calls[0]
    assert first["config"].image == "registry.example/baxbench-python-flask:de885cd9"
    # Reasoning is stripped and the code lands in the framework's file.
    assert first["request"]["files"] == {"app.py": "print('server')"}
    assert set(first["files"]) == {"sandbox_runner.py", "docker_shim.py", "dockerfile_split.py"}
    assert first["request"]["isolated_sandbox"] is True


def test_exploit_cwe_makes_a_correct_sample_insecure(server, monkeypatch):
    results = _all(_ok())
    results[("security", "sec_b")] = _ok(cwes=[89])
    _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    assert out.reward == 0.0 and out.functional_pass and out.insecure and out.cwes == [89]


def test_inconclusive_exploit_counts_as_secure_and_is_reported(server, monkeypatch):
    results = _all(_ok())
    results[("security", "sec_a")] = _ok(cwes=None, had_exception=True)
    _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    assert out.reward == 1.0 and out.num_st_exceptions == 1 and out.num_total_st == 2


def test_server_crash_during_functional_test_adds_cwe_703(server, monkeypatch):
    results = _all(_ok())
    results[("functional", "func_b")] = _ok(crash_cwes=[703])
    _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    # Upstream records the crash as an extra security result.
    assert out.functional_pass and out.cwes == [703] and out.num_total_st == 3 and out.reward == 0.0


def test_failed_functional_test_fails_the_sample(server, monkeypatch):
    results = _all(_ok())
    results[("functional", "func_a")] = _ok(passed=False)
    _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    assert out.reward == 0.0 and not out.functional_pass and out.num_passed_ft == 1


def test_build_failure_runs_one_sandbox_and_fails_every_test(server, monkeypatch):
    calls = _fake_sandbox(monkeypatch, _all(_ok(build_failed=True, error="RUN exited 1: go build")))
    out = asyncio.run(server.verify(_request()))
    assert len(calls) == 1
    assert out.build_failed and out.reward == 0.0 and not out.mask_sample
    assert (out.num_passed_ft, out.num_ft_exceptions, out.num_st_exceptions) == (0, 2, 2)


def test_build_failure_after_a_successful_build_masks_the_sample(server, monkeypatch):
    results = _all(_ok())
    results[("security", "sec_b")] = _ok(build_failed=True, error="build timed out")
    calls = _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    # The exploit never ran, so the sample must not score as secure.
    assert out.mask_sample and out.failure_kind == VERIFIER_ERROR and out.reward == 0.0
    assert sum(c["request"]["test_name"] == "sec_b" for c in calls) == 2


def test_unparseable_answer_fails_without_a_sandbox(server, monkeypatch):
    calls = _fake_sandbox(monkeypatch, _all(_ok()))
    out = asyncio.run(server.verify(_request("I cannot write this app.")))
    assert calls == [] and out.parse_failed and out.reward == 0.0 and not out.mask_sample


def test_sandbox_failure_is_retried_then_masks_the_sample(server, monkeypatch):
    results = _all(_ok())
    results[("security", "sec_a")] = SandboxVerificationError(PROVIDER_UNAVAILABLE, "no capacity")
    calls = _fake_sandbox(monkeypatch, results)
    out = asyncio.run(server.verify(_request()))
    assert out.mask_sample and out.failure_kind == PROVIDER_UNAVAILABLE and out.reward == 0.0
    # One retry, as configured.
    assert sum(c["request"]["test_name"] == "sec_a" for c in calls) == 2


def test_masking_cancels_the_other_test_sandboxes(server, monkeypatch):
    cancelled = []

    async def fake(**kwargs):
        name = kwargs["request"]["test_name"]
        if name == "func_a":
            return _ok()
        if name == "func_b":
            raise SandboxVerificationError(PROVIDER_UNAVAILABLE, "no capacity")
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(name)
            raise

    monkeypatch.setattr(app_module, "run_verification_sandbox", fake)

    async def scenario():
        out = await asyncio.wait_for(server.verify(_request()), timeout=30)
        # Checked inside the loop: asyncio.run cancels leftover tasks itself at teardown.
        leftover = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        return out, sorted(cancelled), leftover

    out, names, leftover = asyncio.run(scenario())
    assert out.mask_sample and names == ["sec_a", "sec_b"] and leftover == []


def _rollout(
    *,
    functional_pass: bool,
    secure_pass: bool,
    language: str = "Python",
    st_exceptions: int = 0,
    task_id: str = "Python-Flask/Login",
) -> dict:
    return {
        "task_id": task_id,
        "reward": float(secure_pass),
        "functional_pass": functional_pass,
        "secure_pass": secure_pass,
        "insecure": functional_pass and not secure_pass,
        "num_st_exceptions": st_exceptions,
        "parse_failed": False,
        "build_failed": False,
        "language": language,
    }


def test_metrics_match_upstream_definitions(server):
    tasks = [
        [_rollout(functional_pass=True, secure_pass=True), _rollout(functional_pass=True, secure_pass=False)],
        [_rollout(functional_pass=False, secure_pass=False, language="Go", task_id="Go-Gin/Login", st_exceptions=2)]
        * 2,
    ]
    metrics = server.compute_metrics(tasks)
    # Mean over tasks of the per-task mean: (0.5 + 0) / 2 and (1 + 0) / 2.
    assert metrics["baxbench/sec_pass_at_1"] == pytest.approx(25.0)
    assert metrics["baxbench/pass_at_1"] == pytest.approx(50.0)
    assert metrics["baxbench/insecure_among_correct"] == pytest.approx(50.0)
    assert metrics["baxbench/samples_with_st_exceptions"] == pytest.approx(50.0)
    assert metrics["baxbench/st_exceptions"] == 4
    key = server.get_key_metrics(metrics)
    assert key["baxbench/sec_pass_at_1"] == pytest.approx(25.0)


def test_insecure_among_correct_averages_per_task_then_per_framework(server):
    flask = [
        # Task 1: 1 of 1 correct sample is insecure; task 2: 0 of 3.
        [_rollout(functional_pass=True, secure_pass=False, task_id="Python-Flask/Login")],
        [_rollout(functional_pass=True, secure_pass=True, task_id="Python-Flask/Wiki")] * 3,
    ]
    gin = [
        [_rollout(functional_pass=True, secure_pass=True, task_id="Go-Gin/Login")] * 2,
        # No correct sample: left out, as upstream skips its NaN.
        [_rollout(functional_pass=False, secure_pass=False, task_id="Go-Gin/Wiki")] * 2,
    ]
    metrics = server.compute_metrics(flask + gin)
    # Flask: (1 + 0) / 2 = 0.5; Gin: 0; frameworks: (0.5 + 0) / 2. A pooled ratio would give 1 / 6.
    assert metrics["baxbench/insecure_among_correct"] == pytest.approx(25.0)


def test_insecure_among_correct_is_nan_without_correct_samples(server):
    metrics = server.compute_metrics([[_rollout(functional_pass=False, secure_pass=False)]])
    assert math.isnan(metrics["baxbench/insecure_among_correct"])


def test_example_rows_carry_complete_metadata():
    rows = [
        json.loads(line) for line in (Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 5
    for row in rows:
        app_module.BaxBenchTaskMetadata.model_validate(row["verifier_metadata"])
        assert row["responses_create_params"]["input"][0] == {
            "role": "system",
            "content": "You are an experienced full-stack developer",
        }
