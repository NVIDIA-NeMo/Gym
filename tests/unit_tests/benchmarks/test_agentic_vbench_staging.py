# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-task adjustments the Agentic-VBench Harbor entry applies: judge routing, judge guard, asset retries."""

import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import tomllib
import types
from pathlib import Path

import pytest

from benchmarks.agentic_vbench import staging
from benchmarks.agentic_vbench.staging import (
    HUB_JUDGE,
    JUDGE_PROTOCOLS,
    MATERIALS_FETCH_GUARD,
    MATERIALS_RETRY,
    MATERIALS_RETRY_PATCHED,
    OUTPUT_ARTIFACT,
    TYPING_DIAGNOSTIC,
    stage_assembly_judge_guard,
    stage_judge,
    stage_materials,
)


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

MATERIALS_TASK = """version = "1.0"
[task]
name = "agentic-vbench/exp-demo"
[environment]
cpus = 4
allow_internet = true
[[steps]]
name = "solve"
[steps.verifier]
timeout_sec = 1800.0
[steps.verifier.env]
MATERIALS_URL = "https://huggingface.co/datasets/demo/resolve/main/materials/exp-demo.zip"
"""

# The pinned repair verifier's fetch, a two-line curl command.
MATERIALS_VERIFIER = (
    f"set -euo pipefail\nmkdir -p /tests /tmp/g\ncurl --fail --silent --show-error --location {MATERIALS_RETRY} \\\n"
    "     \"$MATERIALS_URL\" -o /tmp/g.zip\nunzip -qo /tmp/g.zip 'golden/*' -d /tmp/g\n"
)

ASSEMBLY_JUDGE = """import json


def _normalize(source):
    return str(source)


def _zero(reason):
    return {"reward": 0.0, "reason": reason}


def score(solution_json):
    sol = json.loads(solution_json.read_text())

    segments = sol.get("segments")
    if not isinstance(segments, list):
        return _zero("solution.json: segments not a list")

    pred = [_normalize(seg.get("source", "")) for seg in segments]
    return {"reward": len(pred) / 4}
"""


def make_task(root: Path, task_toml: str) -> Path:
    task = root / "demo"
    tests = task / "steps/solve/tests"
    tests.mkdir(parents=True)
    (task / "task.toml").write_text(task_toml)
    (tests / "judge.py").write_text("ORIGINAL JUDGE\n")
    (tests / "test.sh").write_text(f"set -euo pipefail\n{{\n  {TYPING_DIAGNOSTIC}\n}} | tee diag.txt\n")
    return task


def pin_synthetic_assembly_judge(monkeypatch: pytest.MonkeyPatch, task: Path) -> None:
    hardened = ASSEMBLY_JUDGE.replace(staging._ASSEMBLY_MANIFEST_PARSER, staging._ASSEMBLY_HARDENED_PARSER)
    monkeypatch.setattr(
        staging,
        "_ASSEMBLY_JUDGE_SHA256",
        {
            task.name: (
                hashlib.sha256(ASSEMBLY_JUDGE.encode()).hexdigest(),
                hashlib.sha256(hardened.encode()).hexdigest(),
            )
        },
    )


def test_assembly_judge_hash_manifest_covers_all_pinned_tasks() -> None:
    assert set(staging._ASSEMBLY_JUDGE_SHA256) == {f"agentic-vbench-assembly-task{i}" for i in range(1, 19)}
    assert all(len(source) == len(staged) == 64 for source, staged in staging._ASSEMBLY_JUDGE_SHA256.values())


PINNED_ROOT = Path(os.environ.get("AGENTIC_VBENCH_ROOT", "/nonexistent"))


@pytest.mark.skipif(not (PINNED_ROOT / "tasks").is_dir(), reason="Pinned Agentic-VBench checkout not available")
def test_pinned_assembly_judges_match_and_stage_to_exact_hashes(tmp_path: Path) -> None:
    for task_id, (source_sha256, staged_sha256) in staging._ASSEMBLY_JUDGE_SHA256.items():
        task = PINNED_ROOT / "tasks/agentic_vbench_assembly" / task_id
        assert hashlib.sha256((task / "steps/solve/tests/judge.py").read_bytes()).hexdigest() == source_sha256
        output = tmp_path / "outputs" / task_id
        output.mkdir(parents=True)
        staged = stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=output)
        assert hashlib.sha256((staged / "steps/solve/tests/judge.py").read_bytes()).hexdigest() == staged_sha256


def test_assembly_judge_guard_scores_non_object_manifest_zero_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = make_task(tmp_path / "tasks/agentic_vbench_assembly", JUDGED_TASK)
    source_judge = task / "steps/solve/tests/judge.py"
    source_judge.write_text(ASSEMBLY_JUDGE)
    pin_synthetic_assembly_judge(monkeypatch, task)
    output = tmp_path / "output"
    output.mkdir()

    staged = stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=output)

    original_namespace: dict = {}
    exec(ASSEMBLY_JUDGE, original_namespace)
    namespace: dict = {}
    exec((staged / "steps/solve/tests/judge.py").read_text(), namespace)
    bare_list = tmp_path / "bare-list.json"
    bare_list.write_text('[{"source": "3"}]')
    scalar_segment = tmp_path / "scalar-segment.json"
    scalar_segment.write_text('{"segments": [3]}')
    valid_object = tmp_path / "valid-object.json"
    valid_object.write_text('{"segments": [{"source": "3"}, {"source": "4"}]}')
    assert namespace["score"](bare_list) == {"reward": 0.0, "reason": "solution.json: expected top-level object"}
    assert namespace["score"](scalar_segment) == {"reward": 0.0, "reason": "solution.json: segment 0 is not an object"}
    assert namespace["score"](valid_object) == original_namespace["score"](valid_object) == {"reward": 0.5}
    assert source_judge.read_text() == ASSEMBLY_JUDGE
    provenance = json.loads((output / "assembly_judge_guard.json").read_text())
    assert provenance["original_judge_sha256"] == hashlib.sha256(ASSEMBLY_JUDGE.encode()).hexdigest()
    assert provenance["staged_judge_sha256"] != provenance["original_judge_sha256"]
    assert stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=output) == staged
    assert (staged / "steps/solve/tests/judge.py").read_text().count("expected top-level object") == 1


def test_assembly_judge_guard_rejects_unexpected_judges_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = make_task(tmp_path / "tasks/agentic_vbench_assembly", JUDGED_TASK)
    source = task / "steps/solve/tests/judge.py"
    source_text = source.read_text()
    monkeypatch.setattr(
        staging, "_ASSEMBLY_JUDGE_SHA256", {task.name: (hashlib.sha256(source_text.encode()).hexdigest(), "0" * 64)}
    )
    with pytest.raises(ValueError, match="unexpected manifest parser"):
        stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=tmp_path)
    assert source.read_text() == source_text and not (tmp_path / "staging").exists()

    source.write_text(ASSEMBLY_JUDGE)
    pin_synthetic_assembly_judge(monkeypatch, task)
    source.write_text(ASSEMBLY_JUDGE + "# unexpected edit\n")
    with pytest.raises(ValueError, match="unexpected source digest"):
        stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=tmp_path)
    assert not (tmp_path / "staging").exists()


def test_assembly_judge_guard_leaves_other_families_unstaged(tmp_path: Path) -> None:
    task = make_task(tmp_path / "tasks/agentic_vbench_repair", JUDGED_TASK)
    assert stage_assembly_judge_guard(task, staging=tmp_path / "staging", output=tmp_path) == task
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize("protocol", sorted(JUDGE_PROTOCOLS))
def test_hub_judge_is_verifier_only(tmp_path: Path, protocol: str) -> None:
    task = make_task(tmp_path / "source", JUDGED_TASK)
    output = tmp_path / "output"
    output.mkdir()

    staged = stage_judge(task, staging=tmp_path / "staging", output=output, protocol=protocol)

    config = tomllib.loads((staged / "task.toml").read_text())
    assert "env" not in config["environment"]
    assert config["steps"][0]["verifier"]["timeout_sec"] == 1800.0
    assert config["steps"][0]["artifacts"] == [OUTPUT_ARTIFACT]
    assert config["steps"][0]["verifier"]["env"] == {
        "NVINFERENCE_API_KEY": "${NVINFERENCE_API_KEY}",
        **JUDGE_PROTOCOLS[protocol],
    }
    tests = staged / "steps/solve/tests"
    assert (tests / "avb-original-judge.py").read_text() == "ORIGINAL JUDGE\n"
    assert (tests / "judge.py").read_bytes() == HUB_JUDGE.read_bytes()
    assert TYPING_DIAGNOSTIC + " || true" in (tests / "test.sh").read_text()
    # The pinned source task is untouched.
    assert (task / "steps/solve/tests/judge.py").read_text() == "ORIGINAL JUDGE\n"
    assert tomllib.loads((task / "task.toml").read_text())["environment"]["env"]
    provenance = json.loads((output / "judge_protocol.json").read_text())
    assert provenance["original_sha256"]["judge.py"] == hashlib.sha256(b"ORIGINAL JUDGE\n").hexdigest()
    assert "avb-original-judge.py" not in provenance["original_sha256"]  # only files that exist are digested
    assert set(provenance["staged_sha256"]) == {"task.toml", "test.sh", "judge.py", "avb-original-judge.py"}
    assert "NVINFERENCE_API_KEY" not in json.dumps(provenance["settings"])


def test_unjudged_task_is_not_staged(tmp_path: Path) -> None:
    unjudged = JUDGED_TASK.replace(
        JUDGED_TASK[JUDGED_TASK.index("[environment.env]") : JUDGED_TASK.index("[[steps]]")], ""
    )
    task = make_task(tmp_path, unjudged)
    assert stage_judge(task, staging=tmp_path / "staging", output=tmp_path, protocol="nvinference-upstream") == task
    assert not (tmp_path / "staging").exists()


def test_judged_task_without_a_judge_and_unknown_protocols_are_rejected(tmp_path: Path) -> None:
    task = make_task(tmp_path, JUDGED_TASK)
    with pytest.raises(ValueError, match="Unknown judge protocol"):
        stage_judge(task, staging=tmp_path / "s", output=tmp_path, protocol="native-guess")
    (task / "steps/solve/tests/judge.py").unlink()
    with pytest.raises(ValueError, match="without tests/judge.py"):
        stage_judge(task, staging=tmp_path / "s", output=tmp_path, protocol="nvinference-upstream")


def test_materials_download_retries_longer_and_a_failed_fetch_exits_75(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", MATERIALS_TASK)
    verifier = task / "steps/solve/tests/test.sh"
    verifier.write_text(MATERIALS_VERIFIER)
    output = tmp_path / "records"
    output.mkdir()

    staged = stage_materials(task, staging=tmp_path / "staging", output=output)

    assert staged != task
    patched = (staged / "steps/solve/tests/test.sh").read_text()
    assert MATERIALS_RETRY_PATCHED in patched and MATERIALS_RETRY not in patched
    assert "--max-time" in MATERIALS_RETRY_PATCHED and "--retry-max-time" in MATERIALS_RETRY_PATCHED
    assert patched.count(MATERIALS_FETCH_GUARD) == 1
    assert '"$MATERIALS_URL" -o /tmp/g.zip' + MATERIALS_FETCH_GUARD in patched
    assert patched.endswith("unzip -qo /tmp/g.zip 'golden/*' -d /tmp/g\n")
    # The verifier reads only the agent's output: declared as the step's artifact for separate verification.
    config = tomllib.loads((staged / "task.toml").read_text())
    assert config["steps"][0]["artifacts"] == [OUTPUT_ARTIFACT]
    assert config["steps"][0]["verifier"]["env"]["MATERIALS_URL"].endswith("exp-demo.zip")
    # Source task and non-verifier files are untouched.
    assert verifier.read_text() == MATERIALS_VERIFIER
    assert (task / "task.toml").read_text() == MATERIALS_TASK
    record = json.loads((output / "materials.json").read_text())
    assert record["verifiers"]["steps/solve/tests/test.sh"]["original_sha256"] == _sha256(MATERIALS_VERIFIER)
    assert record["verifiers"]["steps/solve/tests/test.sh"]["staged_sha256"] == _sha256(patched)
    assert record["task_toml_sha256"]["original"] != record["task_toml_sha256"]["staged"]
    # Re-staging the same task replaces the earlier copy instead of failing or doubling the guard.
    assert stage_materials(task, staging=tmp_path / "staging", output=output) == staged
    assert (staged / "steps/solve/tests/test.sh").read_text().count(MATERIALS_FETCH_GUARD) == 1


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_guarded_materials_fetch_exits_75_with_the_reason_when_curl_fails(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", MATERIALS_TASK)
    (task / "steps/solve/tests/test.sh").write_text(MATERIALS_VERIFIER)
    staged = stage_materials(task, staging=tmp_path / "staging", output=tmp_path)
    script = (staged / "steps/solve/tests/test.sh").read_text()
    # Run the patched fetch with `curl` replaced by a stub that fails like a 429 (exit 22).
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text("#!/bin/bash\nexit 22\n")
    (fake_bin / "curl").chmod(0o755)
    script = script.replace("mkdir -p /tests /tmp/g\n", "")
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "MATERIALS_URL": "x"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 75
    assert "verifier unavailable: golden assets download from MATERIALS_URL failed (curl exit 22)" in result.stderr
    assert "unzip" not in result.stderr  # the script stopped at the fetch


def test_tasks_without_materials_download_are_unchanged(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", JUDGED_TASK)
    assert stage_materials(task, staging=tmp_path / "staging", output=tmp_path) == task


@pytest.fixture
def hub_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in ("anthropic", "google", "google.genai"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["google"].genai = sys.modules["google.genai"]
    sys.modules["google.genai"].Client = object
    sys.modules["anthropic"].Anthropic = object
    monkeypatch.setenv("AVB_JUDGE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delitem(sys.modules, "benchmarks.agentic_vbench.hub_judge", raising=False)
    module = importlib.import_module("benchmarks.agentic_vbench.hub_judge")
    for key, value in JUDGE_PROTOCOLS["nvinference-upstream"].items():
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
        "model": "azure/anthropic/claude-opus-4-7",
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
    assert path == "/v1beta/models/gcp/google/gemini-3.1-pro-preview:generateContent"
    parts = payload["contents"][0]["parts"]
    assert parts[0]["inlineData"]["mimeType"] == "video/mp4"
    assert parts[1] == {"inlineData": {"mimeType": "audio/mpeg", "data": "bXAz"}}
    assert parts[2] == {"text": "judge this"}
    assert payload["generationConfig"] == {
        "thinkingConfig": {"thinkingLevel": "low"},
        "temperature": 0.0,
        "maxOutputTokens": 4096,
    }


def _http_error(request, code: int, body: bytes = b"x"):
    import io
    import urllib.error

    return urllib.error.HTTPError(request.full_url, code, "status", {}, io.BytesIO(body))


def test_hub_post_retries_transient_400_but_records_auth_failures(hub_judge, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

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
            raise _http_error(request, code)
        return Resp(b'{"ok": true}')

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    module.FAILURES.clear()
    assert module.real_post("/v1/messages", {}) == {"ok": True}
    assert module.FAILURES == {}

    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(request, 403))
    )
    with pytest.raises(RuntimeError, match="HTTP 403"):
        module.real_post("/v1/messages", {})
    assert len(module.FAILURES) == 1


def test_hub_post_surfaces_provider_status_message_only(hub_judge, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(_http_error(request, 400, json.dumps(body).encode())),
    )
    module.FAILURES.clear()
    with pytest.raises(RuntimeError, match=r"HTTP 400 \(Request contains an invalid argument\. details: x+\)") as info:
        module.real_post("/v1beta/models/m:generateContent", {"contents": []})
    message = str(info.value)
    assert len(message) < 220 and "\n" not in message
    assert "AAAA" not in message and "secret-key" not in message  # no payloads, no credentials
    assert module.FAILURES == {}  # a persistent 4xx is this request's error, not an infrastructure rejection


def _socket_server(handler):
    """A local TCP server that answers each connection with ``handler(conn)`` on a daemon thread."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                try:
                    handler(conn)
                except OSError:
                    pass

    threading.Thread(target=serve, daemon=True).start()
    return server


def _drop_before_status_line(conn) -> None:
    conn.recv(65536)


def _truncate_body(conn) -> None:
    conn.recv(65536)
    conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 400\r\n\r\n{"content": [')


def _stall(conn) -> None:
    conn.recv(65536)
    # Not time.sleep: the tests replace the module's time.sleep, which is the global one, with a no-op.
    threading.Event().wait(1.0)


@pytest.mark.parametrize(
    "handler,expected",
    [
        (_drop_before_status_line, "RemoteDisconnected"),
        (_truncate_body, "IncompleteRead"),
        (_stall, "TimeoutError"),
    ],
    ids=["dropped-before-status-line", "truncated-body", "stalled"],
)
def test_hub_post_records_transport_failures_as_unavailable(
    hub_judge, monkeypatch: pytest.MonkeyPatch, handler, expected: str
) -> None:
    module, _ = hub_judge
    server = _socket_server(handler)
    try:
        monkeypatch.setenv("AVB_JUDGE_BASE_URL", f"http://127.0.0.1:{server.getsockname()[1]}")
        monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
        monkeypatch.setattr(module.time, "sleep", lambda _s: None)
        monkeypatch.setattr(module, "CALL_TIMEOUT_SECONDS", 0.2)
        module.FAILURES.clear()
        with pytest.raises(RuntimeError, match=expected):
            module.real_post("/v1/messages", {"model": "m"})
        assert len(module.FAILURES) == 1 and expected in next(iter(module.FAILURES.values()))
    finally:
        server.close()


def test_hub_post_records_5xx_and_size_rejections_but_forgets_a_failure_a_retry_scored(
    hub_judge, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    module, _ = hub_judge
    monkeypatch.setenv("AVB_JUDGE_BASE_URL", "https://hub.example")
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    module.FAILURES.clear()
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda request, timeout: (_ for _ in ()).throw(_http_error(request, 503))
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        module.real_post("/v1/messages", {"item": 1})
    assert len(module.FAILURES) == 1
    # A request the Hub refuses for its size is the Hub's limit, not a judge error of the item.
    big = json.dumps({"error": {"message": "Request payload size exceeds the limit"}}).encode()
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(_http_error(request, 413, big)),
    )
    with pytest.raises(RuntimeError, match="HTTP 413"):
        module.real_post("/v1beta/models/m:generateContent", {"item": 2})
    assert len(module.FAILURES) == 2

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # The judge's own item retry repeats the same request and scores it: that failure no longer counts.
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda request, timeout: Resp(b'{"ok": true}'))
    assert module.real_post("/v1/messages", {"item": 1}) == {"ok": True}
    assert len(module.FAILURES) == 1 and "HTTP 413" in next(iter(module.FAILURES.values()))


def test_hub_post_stops_at_the_overall_deadline(hub_judge, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = hub_judge
    monkeypatch.setenv("AVB_JUDGE_BASE_URL", "https://hub.example")
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    calls = []
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda request, timeout: calls.append(request))
    monkeypatch.setattr(module, "_STARTED", time.monotonic() - module.DEADLINE_SECONDS - 1)
    module.FAILURES.clear()
    with pytest.raises(RuntimeError, match="deadline"):
        module.real_post("/v1/messages", {})
    assert calls == [] and len(module.FAILURES) == 1


def _stub_original_judge(module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, results: bool) -> Path:
    base = tmp_path / "base"
    (base / "results").mkdir(parents=True)
    if results:
        (base / "results" / "harbor_trial_p1.json").write_text(json.dumps({"pass": True, "score": 1.0}))
    original = tmp_path / "avb-original-judge.py"
    original.write_text(f"BASE = {str(base)!r}\n")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "judge.py"))
    monkeypatch.setattr(module.sys, "argv", ["judge.py", "run", "source.mp4", "harbor_trial"])
    return base


def test_hub_judge_merges_provenance_into_every_result_when_the_hub_served_every_call(
    hub_judge, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    module, calls = hub_judge
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    base = _stub_original_judge(module, monkeypatch, tmp_path, results=True)
    module.FAILURES.clear()

    module.main()

    # One preflight request per route went out before the judge ran.
    assert [path for path, _, _ in calls] == [
        "/v1/messages",
        f"/v1beta/models/{JUDGE_PROTOCOLS['nvinference-upstream']['AVB_JUDGE_GEMINI_MODEL']}:generateContent",
    ]
    result = json.loads((base / "results" / "harbor_trial_p1.json").read_text())
    assert result["pass"] is True and result["judge_provider"] == "nvidia-inference-hub"
    assert result["judge_claude_model"] == JUDGE_PROTOCOLS["nvinference-upstream"]["AVB_JUDGE_CLAUDE_MODEL"]
    assert json.loads((tmp_path / "artifacts" / "harbor_trial_p1.json").read_text()) == result
    assert "verifier unavailable" not in capsys.readouterr().err


def test_hub_judge_exits_75_on_missing_configuration_unusable_routes_and_outages(
    hub_judge, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    module, _ = hub_judge
    _stub_original_judge(module, monkeypatch, tmp_path, results=False)
    monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as info:
        module.main()
    assert info.value.code == 75
    assert "verifier unavailable: missing verifier configuration: NVINFERENCE_API_KEY" in capsys.readouterr().err

    # A route the key cannot use fails its preflight request: the whole verification is unavailable, before the
    # judge scores a single item 0.
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")

    def refuse_gemini(path: str, payload: dict, headers: dict | None = None) -> dict:
        if "generateContent" in path:
            raise RuntimeError("Hub judge HTTPError HTTP 404 (model not found)")
        return {"content": []}

    monkeypatch.setattr(module, "post", refuse_gemini)
    with pytest.raises(SystemExit) as info:
        module.main()
    assert info.value.code == 75
    err = capsys.readouterr().err
    assert "Gemini judge route" in err and "HTTP 404" in err

    # The judge ran (stubbed), but a Hub call failed on the way and no later attempt scored it: unavailable.
    monkeypatch.setattr(module, "post", lambda path, payload, headers=None: {"content": [], "candidates": []})
    module.FAILURES.clear()
    module.FAILURES["abc"] = "/v1/messages: Hub judge HTTPError HTTP 403 (not available for this API key)"
    with pytest.raises(SystemExit) as info:
        module.main()
    assert info.value.code == 75
    assert "1 Hub judge failures" in capsys.readouterr().err
