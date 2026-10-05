# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import importlib
import json
import os
import sys
import tomllib
import types
from pathlib import Path

import pytest

from responses_api_agents.agentic_vbench_agent import harbor_runner
from responses_api_agents.agentic_vbench_agent.harbor_runner import (
    JUDGE_PROTOCOLS,
    TYPING_DIAGNOSTIC,
    container_endpoint,
    stage_judge,
)


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_host_proxy_uses_rootless_gateway(host: str) -> None:
    assert container_endpoint(f"http://{host}:8123/v1/") == "http://10.0.2.2:8123/v1"


def test_remote_model_endpoint_is_preserved() -> None:
    assert container_endpoint("https://model.example/v1") == "https://model.example/v1"


@pytest.mark.parametrize("endpoint", ["file:///tmp/model", "http://user:secret@localhost:8000/v1", "localhost"])
def test_invalid_or_credentialed_endpoint_is_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError):
        container_endpoint(endpoint)


JUDGED_TASK = """version = "1.0"

[task]
name = "agentic-vbench/demo"

[environment]
cpus = 4
allow_internet = true

[environment.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY:-}"
GEMINI_API_KEY    = "${GEMINI_API_KEY:-}"

[[steps]]
name = "solve"

[steps.agent]
timeout_sec = 1800.0

[steps.verifier]
timeout_sec = 1800.0
"""


def make_task(root: Path, task_toml: str) -> Path:
    task = root / "demo"
    tests = task / "steps/solve/tests"
    tests.mkdir(parents=True)
    (task / "task.toml").write_text(task_toml)
    (tests / "judge.py").write_text("ORIGINAL JUDGE\n")
    (tests / "test.sh").write_text(f"set -euo pipefail\n{{\n  {TYPING_DIAGNOSTIC}\n}} | tee diag.txt\n")
    return task


def test_hub_judge_is_verifier_only(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", JUDGED_TASK)
    output = tmp_path / "output"
    output.mkdir()

    staged = stage_judge(task, staging=tmp_path / "staging", output=output, protocol="nvinference-hybrid")

    config = tomllib.loads((staged / "task.toml").read_text())
    assert "env" not in config["environment"]
    assert config["steps"][0]["verifier"]["timeout_sec"] == 1800.0
    assert config["steps"][0]["verifier"]["env"] == {
        "NVINFERENCE_API_KEY": "${NVINFERENCE_API_KEY}",
        **JUDGE_PROTOCOLS["nvinference-hybrid"],
    }
    tests = staged / "steps/solve/tests"
    assert (tests / "avb-original-judge.py").read_text() == "ORIGINAL JUDGE\n"
    assert (tests / "judge.py").read_bytes() == (Path(harbor_runner.__file__).with_name("hub_judge.py")).read_bytes()
    assert TYPING_DIAGNOSTIC + " || true" in (tests / "test.sh").read_text()
    # The pinned source task is untouched.
    assert (task / "steps/solve/tests/judge.py").read_text() == "ORIGINAL JUDGE\n"
    assert tomllib.loads((task / "task.toml").read_text())["environment"]["env"]
    provenance = json.loads((output / "judge_protocol.json").read_text())
    assert provenance["original_sha256"]["judge.py"] == hashlib.sha256(b"ORIGINAL JUDGE\n").hexdigest()
    assert "NVINFERENCE_API_KEY" not in json.dumps(provenance["settings"])


def test_unjudged_task_is_not_staged(tmp_path: Path) -> None:
    task = make_task(
        tmp_path,
        JUDGED_TASK.replace(JUDGED_TASK[JUDGED_TASK.index("[environment.env]") : JUDGED_TASK.index("[[steps]]")], ""),
    )
    assert stage_judge(task, staging=tmp_path / "staging", output=tmp_path, protocol="nvinference-hybrid") == task
    assert not (tmp_path / "staging").exists()


def test_unknown_judge_protocol_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown judge protocol"):
        stage_judge(make_task(tmp_path, JUDGED_TASK), staging=tmp_path / "s", output=tmp_path, protocol="native-guess")


@pytest.fixture
def hub_judge(monkeypatch: pytest.MonkeyPatch):
    google = types.ModuleType("google")
    google.genai = types.ModuleType("google.genai")
    google.genai.Client = object
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", google.genai)
    monkeypatch.delitem(sys.modules, "responses_api_agents.agentic_vbench_agent.hub_judge", raising=False)
    module = importlib.import_module("responses_api_agents.agentic_vbench_agent.hub_judge")
    for key, value in JUDGE_PROTOCOLS["nvinference-hybrid"].items():
        monkeypatch.setenv(key, value)
    calls = []

    def fake_post(path: str, payload: dict, headers: dict | None = None) -> dict:
        calls.append((path, payload, headers))
        if path == "/v1/messages":
            return {"content": [{"type": "text", "text": '{"pass": true, "why": "ok"}'}]}
        return {
            "candidates": [
                {"content": {"parts": [{"text": "hidden", "thought": True}, {"text": '{"pass": false, "why": "x"}'}]}}
            ]
        }

    # Keep the real transport reachable for tests of post() itself.
    monkeypatch.setattr(module, "real_post", module.post, raising=False)
    monkeypatch.setattr(module, "post", fake_post)
    return module, calls


def test_hub_claude_keeps_original_request(hub_judge) -> None:
    module, calls = hub_judge
    content = [{"type": "text", "text": "evidence", "cache_control": {"type": "ephemeral"}}]
    response = module.HubAnthropic().messages.create(
        model="claude-opus-4-7", max_tokens=300, system="grader", messages=[{"role": "user", "content": content}]
    )
    assert response.content[0].text == '{"pass": true, "why": "ok"}'
    path, payload, headers = calls[0]
    assert path == "/v1/messages" and headers == {"anthropic-version": "2023-06-01"}
    assert payload == {
        "model": "us/aws/anthropic/eccn-claude-opus-4-7",
        "max_tokens": 300,
        "system": "grader",
        "messages": [{"role": "user", "content": content}],
    }


def test_hub_gemini_sends_inline_media_with_bounded_thinking(hub_judge, tmp_path: Path) -> None:
    module, calls = hub_judge
    video = tmp_path / "repurpose.mp4"
    video.write_bytes(b"video-bytes")
    client = module.HubGemini()
    uploaded = client.files.upload(file=str(video))
    audio = types.SimpleNamespace(inline_data=types.SimpleNamespace(mime_type="audio/mpeg", data=b"mp3"))
    config = types.SimpleNamespace(temperature=0.0, max_output_tokens=4096)

    response = client.models.generate_content(
        model="gemini-3.1-pro-preview", contents=[uploaded, audio, "judge this"], config=config
    )

    assert response.text == '{"pass": false, "why": "x"}'
    path, payload, _ = calls[0]
    assert path == "/v1beta/models/us/gcp/google/eccn-gemini-3.5-flash:generateContent"
    parts = payload["contents"][0]["parts"]
    assert parts[0] == {"inlineData": {"mimeType": "video/mp4", "data": base64.b64encode(b"video-bytes").decode()}}
    assert parts[1] == {"inlineData": {"mimeType": "audio/mpeg", "data": base64.b64encode(b"mp3").decode()}}
    assert parts[2] == {"text": "judge this"}
    assert payload["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "low"},
        "temperature": 0.0,
        "maxOutputTokens": 4096,
    }


ORIGINAL_JUDGE = Path(os.environ.get("AGENTIC_VBENCH_ROOT", "/nonexistent")) / (
    "tasks/agentic_vbench_repurpose/talk-greta-un-climate-2019/steps/solve/tests/judge.py"
)


@pytest.mark.skipif(not ORIGINAL_JUDGE.is_file(), reason="Pinned Agentic-VBench checkout not available")
def test_pinned_judge_frames_audio_violation_items() -> None:
    # The Hub judge relies on the benchmark's own fix (#107): without it, Gemini Flash
    # inverts negative-weight audio items. Guard against an older pinned revision.
    source = ORIGINAL_JUDGE.read_text()
    audio = source[source.index("def judge_item_gemini_audio") : source.index("def judge_item_gemini_video")]
    assert "prompt += _violation_framing(item)" in audio


def make_job(**kwargs) -> dict:
    return harbor_runner.job_config(
        task_path=Path("/tasks/demo"),
        output=Path("/out"),
        runtime_root=Path("/runtime"),
        model="m",
        endpoint="http://10.0.0.5:3825",
        context_tokens=262144,
        output_tokens=100000,
        backend="remote",
        **kwargs,
    )


def test_max_turns_caps_opencode_build_agent_steps() -> None:
    config = make_job(max_turns=50)["agents"][0]["kwargs"]["opencode_config"]
    assert config["agent"] == {"build": {"steps": 50}}
    assert config["provider"]["openai"]["options"]["baseURL"] == "http://10.0.0.5:3825"


def test_turns_are_uncapped_by_default() -> None:
    assert "agent" not in make_job()["agents"][0]["kwargs"]["opencode_config"]


def test_nonpositive_max_turns_is_rejected() -> None:
    with pytest.raises(ValueError, match="turns"):
        make_job(max_turns=0)


def test_opencode_reserves_the_declared_output_limit() -> None:
    agent = make_job()["agents"][0]
    # A template, because Harbor redacts literal *TOKEN* values when persisting the job config.
    assert agent["env"] == {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "${OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX}"}
    model = agent["kwargs"]["opencode_config"]["provider"]["openai"]["models"]["m"]
    assert model["limit"] == {"context": 262144, "output": 100000}


def test_verifier_timeout_multiplier_only_extends() -> None:
    assert make_job()["verifier_timeout_multiplier"] == 1.0
    assert make_job(verifier_timeout_multiplier=2.0)["verifier_timeout_multiplier"] == 2.0
    with pytest.raises(ValueError, match="shorten"):
        make_job(verifier_timeout_multiplier=0.5)


def test_hub_post_retries_transient_400_but_not_auth_failures(hub_judge, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    module, _ = hub_judge
    monkeypatch.setenv("AVB_JUDGE_BASE_URL", "https://hub.example")
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    codes = iter([400, 400, 200])

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        code = next(codes)
        if code != 200:
            raise urllib.error.HTTPError(request.full_url, code, "bad", {}, io.BytesIO(b"x"))
        return Resp(b'{"ok": true}')

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    module.FAILURES.clear()
    assert module.real_post("/v1/messages", {}) == {"ok": True}
    assert module.FAILURES == []

    def forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", {}, io.BytesIO(b"x"))

    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        module.real_post("/v1/messages", {})
    assert len(module.FAILURES) == 1


def test_hub_post_surfaces_provider_status_message_only(hub_judge, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    module, _ = hub_judge
    monkeypatch.setenv("AVB_JUDGE_BASE_URL", "https://hub.example")
    monkeypatch.setenv("NVINFERENCE_API_KEY", "secret-key")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    body = {
        "error": {
            "code": 400,
            "status": "INVALID_ARGUMENT",
            "message": "Request contains an invalid argument.\n  details: " + "x" * 400,
            "details": [{"request": "data:audio/mpeg;base64,AAAA"}],
        }
    }

    def bad_request(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(module.urllib.request, "urlopen", bad_request)
    module.FAILURES.clear()
    with pytest.raises(RuntimeError, match=r"HTTP 400 \(Request contains an invalid argument\. details: x+\)") as info:
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})
    message = str(info.value)
    assert len(message) < 220  # message truncated to one short line
    assert "\n" not in message
    assert "AAAA" not in message and "secret-key" not in message  # no payloads, no credentials
    assert module.FAILURES == []  # a persistent 4xx is this request's error, not an infrastructure rejection

    def not_json(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b"<html>gateway</html>"))

    monkeypatch.setattr(module.urllib.request, "urlopen", not_json)
    with pytest.raises(RuntimeError, match=r"HTTP 400$"):
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})


def _fake_harbor(tmp_path: Path, failures: int, message: str) -> list[str]:
    """A stand-in for ``harbor run`` that fails ``failures`` times with ``message``, then succeeds."""
    counter = tmp_path / "attempts"
    counter.write_text("0")
    script = tmp_path / "harbor.py"
    script.write_text(
        "import pathlib, sys\n"
        f"counter = pathlib.Path({str(counter)!r})\n"
        "n = int(counter.read_text()) + 1\n"
        "counter.write_text(str(n))\n"
        f"if n <= {failures}:\n"
        f"    sys.stderr.write({message!r} + chr(10)); sys.exit(1)\n"
    )
    return [sys.executable, str(script)]


def test_run_harbor_retries_only_the_docker_probe_failure(tmp_path: Path) -> None:
    import subprocess

    naps: list[float] = []
    jobs = tmp_path / "jobs"
    command = _fake_harbor(tmp_path, failures=2, message=harbor_runner.DOCKER_CHECK_FAILURE + ". Please start Docker")
    harbor_runner.run_harbor(command, env=dict(os.environ), jobs_dir=jobs, sleep=naps.append)
    assert (tmp_path / "attempts").read_text() == "3"
    assert naps == [harbor_runner.HARBOR_START_BACKOFF_SECONDS, 2 * harbor_runner.HARBOR_START_BACKOFF_SECONDS]

    # Any other failure is fatal on the first attempt.
    command = _fake_harbor(tmp_path, failures=1, message="verifier crashed")
    with pytest.raises(subprocess.CalledProcessError):
        harbor_runner.run_harbor(command, env=dict(os.environ), jobs_dir=jobs, sleep=naps.append)
    assert (tmp_path / "attempts").read_text() == "1"

    # Once a trial directory exists the run is never repeated, whatever the message.
    (jobs / "trial" / "task__abc").mkdir(parents=True)
    command = _fake_harbor(tmp_path, failures=1, message=harbor_runner.DOCKER_CHECK_FAILURE)
    with pytest.raises(subprocess.CalledProcessError):
        harbor_runner.run_harbor(command, env=dict(os.environ), jobs_dir=jobs, sleep=naps.append)
    assert (tmp_path / "attempts").read_text() == "1"

    # The probe failure gives up after the configured number of attempts.
    command = _fake_harbor(tmp_path, failures=99, message=harbor_runner.DOCKER_CHECK_FAILURE)
    with pytest.raises(subprocess.CalledProcessError):
        harbor_runner.run_harbor(command, env=dict(os.environ), jobs_dir=tmp_path / "nojobs", sleep=naps.append)
    assert (tmp_path / "attempts").read_text() == str(harbor_runner.HARBOR_START_ATTEMPTS)


def test_judge_key_prefers_credentials_then_hub_then_driver_key() -> None:
    assert harbor_runner.judge_api_key({"NVINFERENCE_API_KEY": "file"}, {"NVINFERENCE_API_KEY": "env"}) == "file"
    assert harbor_runner.judge_api_key({}, {"NVINFERENCE_API_KEY": "env", "INFERENCE_API_KEY": "driver"}) == "env"
    assert harbor_runner.judge_api_key({}, {"INFERENCE_API_KEY": "driver"}) == "driver"
    assert harbor_runner.judge_api_key({}, {}) is None


def test_model_call_timeout_reaches_opencode_provider() -> None:
    options = make_job(model_timeout_ms=3_600_000)["agents"][0]["kwargs"]["opencode_config"]["provider"]["openai"]
    assert options["options"]["timeout"] == 3_600_000
    assert "timeout" not in make_job()["agents"][0]["kwargs"]["opencode_config"]["provider"]["openai"]["options"]
    with pytest.raises(ValueError, match="timeout"):
        make_job(model_timeout_ms=0)


def test_hub_post_persistent_request_error_raises_without_rejecting_verification(
    hub_judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx that persists across retries is this request's fault (e.g. empty audio): the judge
    scores that rubric item as a judge error, as it would with the native SDK, and the
    verification is not rejected. Transport failures still reject it."""
    import io
    import urllib.error

    module, _ = hub_judge
    monkeypatch.setenv("AVB_JUDGE_BASE_URL", "https://hub.example")
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    body = b'{"error": {"code": 400, "message": "Unable to submit request because it has an empty inlineData"}}'
    calls = {"n": 0}

    def empty_media(request, timeout):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(body))

    monkeypatch.setattr(module.urllib.request, "urlopen", empty_media)
    module.FAILURES.clear()
    with pytest.raises(
        RuntimeError, match=r"HTTP 400 \(Unable to submit request because it has an empty inlineData\)"
    ):
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})
    assert calls["n"] == module.ATTEMPTS  # still retried, in case the 400 was transient
    assert module.FAILURES == []  # request-specific: not an infrastructure rejection

    def unavailable(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(module.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})
    assert len(module.FAILURES) == 1  # transport failure: rejected

    def too_many(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "slow down", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(module.urllib.request, "urlopen", too_many)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})
    assert len(module.FAILURES) == 2  # rate limiting is transport, not this request's content
