# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-task adjustments the Agentic-VBench Harbor entry applies: judge routing, judge guard, asset retries."""

import hashlib
import importlib
import json
import os
import subprocess
import sys
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
    assert stage_judge(task, staging=tmp_path / "staging", output=tmp_path, protocol="nvinference-native") == task
    assert not (tmp_path / "staging").exists()


def test_judged_task_without_a_judge_and_unknown_protocols_are_rejected(tmp_path: Path) -> None:
    task = make_task(tmp_path, JUDGED_TASK)
    with pytest.raises(ValueError, match="Unknown judge protocol"):
        stage_judge(task, staging=tmp_path / "s", output=tmp_path, protocol="native-guess")
    (task / "steps/solve/tests/judge.py").unlink()
    with pytest.raises(ValueError, match="without tests/judge.py"):
        stage_judge(task, staging=tmp_path / "s", output=tmp_path, protocol="nvinference-native")


def test_materials_download_retries_longer_and_reports_a_failed_fetch_as_inconclusive(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", MATERIALS_TASK)
    verifier = task / "steps/solve/tests/test.sh"
    verifier.write_text(MATERIALS_VERIFIER)

    staged = stage_materials(task, staging=tmp_path / "staging")

    assert staged != task
    patched = (staged / "steps/solve/tests/test.sh").read_text()
    assert MATERIALS_RETRY_PATCHED in patched and MATERIALS_RETRY not in patched
    assert patched.count(MATERIALS_FETCH_GUARD) == 1
    assert '"$MATERIALS_URL" -o /tmp/g.zip' + MATERIALS_FETCH_GUARD in patched
    assert patched.endswith("unzip -qo /tmp/g.zip 'golden/*' -d /tmp/g\n")
    # Source task and non-verifier files are untouched.
    assert verifier.read_text() == MATERIALS_VERIFIER
    assert (staged / "task.toml").read_text() == MATERIALS_TASK
    # Re-staging the same task replaces the earlier copy instead of failing or doubling the guard.
    assert stage_materials(task, staging=tmp_path / "staging") == staged
    assert (staged / "steps/solve/tests/test.sh").read_text().count(MATERIALS_FETCH_GUARD) == 1


def test_guarded_materials_fetch_writes_the_inconclusive_note_when_curl_fails(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", MATERIALS_TASK)
    (task / "steps/solve/tests/test.sh").write_text(MATERIALS_VERIFIER)
    staged = stage_materials(task, staging=tmp_path / "staging")
    script = (staged / "steps/solve/tests/test.sh").read_text()
    # Run the patched fetch with `curl` replaced by a stub that fails like a 429 (exit 22), writing under tmp_path.
    verifier_dir = tmp_path / "logs/verifier"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text("#!/bin/bash\nexit 22\n")
    (fake_bin / "curl").chmod(0o755)
    script = script.replace("/logs/verifier", str(verifier_dir)).replace("mkdir -p /tests /tmp/g\n", "")
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}", "MATERIALS_URL": "x"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 22
    assert json.loads((verifier_dir / "inconclusive.json").read_text()) == {
        "reason": "golden assets download from MATERIALS_URL failed (curl exit 22)"
    }
    assert "unzip" not in result.stderr  # the script stopped at the fetch


def test_tasks_without_materials_download_are_unchanged(tmp_path: Path) -> None:
    task = make_task(tmp_path / "source", JUDGED_TASK)
    assert stage_materials(task, staging=tmp_path / "staging") == task


@pytest.fixture
def hub_judge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for name in ("anthropic", "google", "google.genai"):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["google"].genai = sys.modules["google.genai"]
    sys.modules["google.genai"].Client = object
    sys.modules["anthropic"].Anthropic = object
    monkeypatch.setenv("AVB_VERIFIER_DIR", str(tmp_path / "verifier"))
    monkeypatch.delitem(sys.modules, "benchmarks.agentic_vbench.hub_judge", raising=False)
    module = importlib.import_module("benchmarks.agentic_vbench.hub_judge")
    for key, value in JUDGE_PROTOCOLS["nvinference-native"].items():
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
    assert module.FAILURES == []

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
    assert module.FAILURES == []  # a persistent 4xx is this request's error, not an infrastructure rejection


def test_hub_judge_reports_outages_and_missing_configuration_as_inconclusive(
    hub_judge, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, _ = hub_judge
    note = tmp_path / "verifier" / "inconclusive.json"
    monkeypatch.delenv("NVINFERENCE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing verifier configuration: NVINFERENCE_API_KEY"):
        module.main()
    assert json.loads(note.read_text()) == {"reason": "missing verifier configuration: NVINFERENCE_API_KEY"}
    note.unlink()

    # The original judge ran (stubbed), but a Hub route failed on the way: no score, an inconclusive note.
    monkeypatch.setenv("NVINFERENCE_API_KEY", "k")
    original = tmp_path / "avb-original-judge.py"
    original.write_text("BASE = '.'\n")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "judge.py"))
    monkeypatch.setattr(module.sys, "argv", ["judge.py", "run", "source.mp4", "harbor_trial"])
    module.FAILURES[:] = ["/v1/messages: Hub judge HTTPError HTTP 403 (not available for this API key)"]
    with pytest.raises(RuntimeError, match="Rejecting verifier result after 1 Hub failures"):
        module.main()
    assert "HTTP 403" in json.loads(note.read_text())["reason"]
