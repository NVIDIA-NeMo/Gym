# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned CLI regressions, enabled in the OpenClaw conformance CI job.

Run locally on Linux with HARNESS=openclaw. Each subprocess exercises session
setup, the real CLI, EnvironmentServer policy, and collection. Raw socket faults
bypass capture intentionally; their assertions concern outcome delivery only.
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse
from scripts.harness_conformance import episode
from scripts.harness_conformance.provider import OpenClawProbe
from scripts.harness_conformance.runner import run_process
from scripts.harness_conformance.scenarios import SCENARIOS, Scenario


CASES = ("http-429", "http-500", "http-503", "reset", "stream-reset", "idle")
CONTEXT_CASES = ("context-http", "context-precheck")
CONTEXT_ERROR = "This model's maximum context length is 32768 tokens. However, you requested 40000 tokens."


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("HARNESS") != "openclaw",
    reason="Run the pinned Linux CLI in the OpenClaw conformance job (HARNESS=openclaw)",
)
@pytest.mark.parametrize("case", CASES)
def test_openclaw_failure_reaches_environment(case: str, tmp_path: Path) -> None:
    # CI runs the suite after pytest; its PROBE_OUTPUT must not exist yet.
    # Keep regression evidence beside it so the same artifact upload retains both.
    output = (
        Path(os.environ["PROBE_OUTPUT"]).parent / "error-regressions" / case
        if "PROBE_OUTPUT" in os.environ
        else tmp_path
    )
    output.mkdir(parents=True, exist_ok=True)
    result = run_process(
        [sys.executable, __file__, "--case", case, "--output", str(output)], directory=output, timeout=180
    )
    assert result == {"returncode": 0, "timed_out": False}, (output / "episode.log").read_text()[-12000:]
    witness = json.loads((output / "witness.json").read_text())
    assert not witness["violations"]
    assert witness["seeded"] == witness["closed"] == 1
    assert not witness["verifications"]
    (response,) = witness["agent_responses"]
    assert response["status"] == "failed"
    outcome = json.loads(response["metadata"]["ng_agent_outcome"])
    if case.startswith("http-"):
        status = int(case.removeprefix("http-"))
        assert outcome == {"failure_kind": "model_request_error", "status_code": status}
        assert len(witness["attempts"]) == 3
        assert all(attempt["status_code"] == status for attempt in witness["attempts"])
    else:
        assert outcome == {"failure_kind": "model_transport_error"}
        expected = {"reset": "Connection error.", "stream-reset": "terminated", "idle": "LLM idle timeout"}
        assert expected[case] in response["error"]["message"]
        requests = json.loads((output / "fault-requests.json").read_text())
        assert len(requests) == {"reset": 3, "stream-reset": 1, "idle": 2}[case]
    assert not (output / "rollouts.jsonl").read_text().strip()
    (failure,) = [json.loads(line) for line in (output / "rollouts_failures.jsonl").read_text().splitlines()]
    assert failure["_ng_failure_class"] == "environment_server_failed"
    retained = failure["_ng_failure_partial_response"]
    assert json.loads(retained["metadata"]["ng_agent_outcome"]) == outcome
    assert retained["error"] == response["error"]
    # Transport and exhausted HTTP errors default to retry; neither becomes a verified, rewarded rollout.
    assert failure["_ng_failure_terminal"] is False
    assert failure["ng_agent_observations"]["records"][0]["status"] == "failed"


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("HARNESS") != "openclaw",
    reason="Run the pinned Linux CLI in the OpenClaw conformance job (HARNESS=openclaw)",
)
@pytest.mark.parametrize("case", CONTEXT_CASES)
def test_openclaw_context_overflow_reaches_verifier(case: str, tmp_path: Path) -> None:
    """A context overflow is a limit: the partial work is graded as an incomplete response, not failed."""
    output = (
        Path(os.environ["PROBE_OUTPUT"]).parent / "error-regressions" / case
        if "PROBE_OUTPUT" in os.environ
        else tmp_path
    )
    output.mkdir(parents=True, exist_ok=True)
    result = run_process(
        [sys.executable, __file__, "--case", case, "--output", str(output)], directory=output, timeout=180
    )
    assert result == {"returncode": 0, "timed_out": False}, (output / "episode.log").read_text()[-12000:]
    witness = json.loads((output / "witness.json").read_text())
    assert not witness["violations"]
    assert witness["seeded"] == witness["closed"] == 1
    if case == "context-precheck":
        assert witness["attempts"] == []
    else:
        assert witness["attempts"]
        assert all(attempt["status_code"] == 400 for attempt in witness["attempts"])
    (response,) = witness["agent_responses"]
    assert response["status"] == "incomplete"
    assert response["error"] is None
    assert "ng_agent_outcome" not in response["metadata"]
    assert witness["verifications"] == [{"reward": 0.0, "answer_seen": False}]
    assert not (output / "rollouts_failures.jsonl").read_text().strip()
    (rollout,) = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert rollout["reward"] == 0.0


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("HARNESS") != "openclaw",
    reason="Run the pinned Linux CLI in the OpenClaw conformance job (HARNESS=openclaw)",
)
def test_openclaw_startup_timeout_reaches_verifier(tmp_path: Path) -> None:
    output = Path(os.environ["PROBE_OUTPUT"]).parent / "startup-timeout" if "PROBE_OUTPUT" in os.environ else tmp_path
    output.mkdir(parents=True, exist_ok=True)
    result = run_process(
        [sys.executable, __file__, "--case", "startup-timeout", "--output", str(output)],
        directory=output,
        timeout=180,
    )
    assert result == {"returncode": 0, "timed_out": False}, (output / "episode.log").read_text()[-12000:]
    witness = json.loads((output / "witness.json").read_text())
    assert witness["attempts"] == []
    assert witness["seeded"] == witness["closed"] == 1
    (response,) = witness["agent_responses"]
    assert response["status"] == "incomplete"
    assert response["error"] is None
    assert "ng_agent_outcome" not in response["metadata"]
    assert witness["verifications"] == [{"reward": 0.0, "answer_seen": False}]
    assert not (output / "rollouts_failures.jsonl").read_text().strip()
    (rollout,) = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert rollout["reward"] == 0.0
    artifacts = json.loads((output / "cli-artifacts.json").read_text())
    assert artifacts["cleanup"]["timed_out"] is True
    assert artifacts["cleanup"]["cleanup_confirmed"] is True
    assert "provider_plugin/policy.ready" not in artifacts["artifacts"]


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("HARNESS") != "openclaw",
    reason="Run the pinned Linux CLI in the OpenClaw conformance job (HARNESS=openclaw)",
)
def test_openclaw_replay_preserves_original_tool_ids(tmp_path: Path) -> None:
    output = Path(os.environ["PROBE_OUTPUT"]).parent / "replay-identity" if "PROBE_OUTPUT" in os.environ else tmp_path
    output.mkdir(parents=True, exist_ok=True)
    result = run_process(
        [
            sys.executable,
            "-m",
            "scripts.harness_conformance.episode",
            "--harness",
            "openclaw",
            "--scenario",
            "tool_success",
            "--directory",
            str(output),
            "--timeout",
            "150",
        ],
        directory=output,
        timeout=180,
    )
    assert result == {"returncode": 0, "timed_out": False}, (output / "episode.log").read_text()[-12000:]
    witness = json.loads((output / "witness.json").read_text())
    assert not witness["violations"]
    original_ids = {call["id"] for call in witness["tool_calls"]}
    assert original_ids and all("_" in call_id for call_id in original_ids)
    replayed_ids = set()
    result_ids = set()
    for attempt in witness["attempts"]:
        for message in attempt["request"]["messages"]:
            if message["role"] == "assistant":
                replayed_ids.update(call["id"] for call in message.get("tool_calls", []))
            elif message["role"] == "tool":
                result_ids.add(message["tool_call_id"])
    assert replayed_ids == result_ids == original_ids


async def run_case(case: str, output: Path) -> None:
    from responses_api_agents.openclaw_agent.app import OpenClawAgent

    status = int(case.removeprefix("http-")) if case.startswith("http-") else 400
    scenario = replace(next(s for s in SCENARIOS if s.name == "model_error"), http_errors=(status,))

    class ErrorProbe(OpenClawProbe):
        async def model(self, request: Request) -> JSONResponse:
            body = await request.json()
            message = CONTEXT_ERROR if case == "context-http" else "controlled model failure"
            payload = {"error": {"message": message, "type": "invalid_request_error"}}
            self.attempts.append({"request": body, "response": payload, "status_code": status})
            self.save()
            return JSONResponse(payload, status_code=status, headers={"Retry-After": "0"})

    original_config = episode._config
    original_task = Scenario.task
    original_collect = OpenClawAgent._collect_sandbox_response

    def config(*args, **kwargs):
        result = original_config(*args, **kwargs)
        agent = result["probe_agent"]["responses_api_agents"]["openclaw_agent"]
        if case == "idle":
            agent["model_timeout_seconds"] = 2
        if case == "context-precheck":
            agent.update(context_window=16384, max_output_tokens=1024)
        return result

    def task(self):
        result = original_task(self)
        if case == "context-precheck":
            result["responses_create_params"]["input"][0]["content"] = "word " * 50000
        if case == "startup-timeout":
            result["responses_create_params"]["metadata"] = {"agent_timeout_sec": "0.5"}
        return result

    async def collect(self, state, *args, **kwargs):
        (output / "cli-artifacts.json").write_text(
            json.dumps({"cleanup": state.session.cleanup, "artifacts": state.session.artifacts}, indent=2)
        )
        return await original_collect(self, state, *args, **kwargs)

    fault = None
    handlers = set()
    requests = []
    release = asyncio.Event()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            requests.append(json.loads(await reader.readexactly(length)))
            (output / "fault-requests.json").write_text(json.dumps(requests, indent=2))
            if case == "idle":
                await release.wait()
            elif case == "stream-reset":
                data = {
                    "id": "chatcmpl-fault",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "conformance-model",
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "Partial"}, "finish_reason": None}
                    ],
                }
                chunk = ("data: " + json.dumps(data) + "\n\n").encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                    + f"{len(chunk):x}\r\n".encode()
                    + chunk
                    + b"\r\n"
                )
                await writer.drain()
                await asyncio.sleep(0.2)
                writer.transport.abort()
            else:
                writer.transport.abort()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            handlers.discard(asyncio.current_task())

    with (
        patch.object(episode, "SCENARIOS", (scenario,)),
        patch.object(episode, "OpenClawProbe", ErrorProbe),
        patch.object(episode, "_config", config),
        patch.object(Scenario, "task", task),
        patch.object(OpenClawAgent, "_collect_sandbox_response", collect),
    ):
        try:
            if case in {"reset", "stream-reset", "idle"}:
                fault = await asyncio.start_server(handle, "127.0.0.1", 0)
                port = fault.sockets[0].getsockname()[1]
                with patch.object(
                    OpenClawAgent, "_resolve_model_base_url", return_value=f"http://127.0.0.1:{port}/v1"
                ):
                    await episode.run_episode(
                        harness="openclaw", scenario_name=scenario.name, directory=output, timeout=150
                    )
            else:
                await episode.run_episode(
                    harness="openclaw", scenario_name=scenario.name, directory=output, timeout=150
                )
        except RuntimeError as exc:
            # The collector intentionally rejects a run with no graded results.
            if "None of the 1 dispatched rollouts produced a result" not in str(exc):
                raise
        finally:
            release.set()
            if fault is not None:
                fault.close()
                await fault.wait_closed()
            if handlers:
                await asyncio.gather(*handlers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, *CONTEXT_CASES, "startup-timeout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_case(args.case, args.output))
