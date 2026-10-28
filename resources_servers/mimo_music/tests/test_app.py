# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import shutil
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from resources_servers.mimo_music.app import (
    MimoMusicResourcesServer,
    MimoMusicResourcesServerConfig,
    MimoMusicVerifyRequest,
    assistant_text,
)
from resources_servers.mimo_music.score_worker import score_abc
from resources_servers.mimo_music.scorer.pipeline import compute_score, extract_abc


ABC = "X:1\nT:Smoke melody\nM:4/4\nL:1/8\nQ:1/4=100\nK:C\nC2 E2 G2 E2 | F2 A2 G4 | E2 D2 C2 D2 | E4 C4 |"
HAS_RENDERER = shutil.which("abc2midi") is not None
renderer = pytest.mark.skipif(not HAS_RENDERER, reason="abc2midi is not installed")


def request(text: str) -> MimoMusicVerifyRequest:
    return MimoMusicVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "Compose a melody."}]},
            "verifier_metadata": {"task_id": "test"},
            "response": {
                "id": "response-test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "output": [
                    {
                        "id": "message-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            },
        }
    )


def server(**kwargs) -> MimoMusicResourcesServer:
    config = MimoMusicResourcesServerConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="music", **kwargs)
    with patch(
        "resources_servers.mimo_music.app.ensure_abc2midi", return_value=shutil.which("abc2midi") or "abc2midi"
    ):
        return MimoMusicResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    "text,expected",
    [
        (ABC, ABC),
        (f"<think>{ABC}</think>Answer", "Answer"),
        (f"<thinking>{ABC}</thinking>Answer", "Answer"),
        (f"{ABC}</think>Answer", "Answer"),
        (f"<think>{ABC}", ""),
        ("", ""),
    ],
)
def test_final_text(text, expected):
    assert assistant_text(request(text).response) == expected


def test_reasoning_channel_not_scored():
    payload = request("No music").model_dump()
    payload["response"]["output"].insert(0, {"id": "r", "type": "reasoning", "summary": []})
    body = MimoMusicVerifyRequest.model_validate(payload)
    assert assistant_text(body.response) == "No music"
    body.response.output = []
    assert assistant_text(body.response) == ""


@pytest.mark.parametrize("text", ["", "I cannot compose music.", f"<think>{ABC}</think>No final tune"])
@pytest.mark.asyncio
async def test_no_abc(text):
    result = await server().verify(request(text))
    assert result.reward == 0.0
    assert not result.mask_sample
    assert not result.abc_extracted
    assert result.verifier_metadata == {"task_id": "test"}


@pytest.mark.asyncio
async def test_reverify_discards_old_reward_and_diagnostics():
    payload = request("no music").model_dump()
    payload.update(
        reward=1.0,
        abc_extracted=True,
        scorer_details={"total": 100},
        mask_sample=True,
        failure_kind="mimo_music:scorer_error",
        failure_reason="old",
    )
    result = await server().verify(MimoMusicVerifyRequest.model_validate(payload))
    assert result.reward == 0
    assert result.mask_sample is False
    assert result.failure_kind is None and result.failure_reason is None
    assert result.abc_extracted is False
    assert result.scorer_details == {"skip": "empty_abc"}


@renderer
@pytest.mark.asyncio
async def test_real_score_and_repeatability():
    music_server = server()
    text = f"```abc\n{ABC}\n```"
    expected = compute_score("music", text)
    assert 0 < expected < 1
    results = await asyncio.gather(*(music_server.verify(request(text)) for _ in range(6)))
    for result in results:
        assert result.reward == expected
        assert not result.mask_sample
        assert result.abc_extracted
        assert result.scorer_details["reject"] == 0
        assert result.scorer_details["total"] / 100 == expected


@renderer
@pytest.mark.asyncio
async def test_recorded_model_rollouts_reverify():
    music_server = server()
    path = Path(__file__).resolve().parents[1] / "data" / "example_rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 5
    assert any(row["reward"] > 0 for row in rows)
    for row in rows:
        result = await music_server.verify(MimoMusicVerifyRequest.model_validate(row))
        assert result.reward == row["reward"]
        assert not result.mask_sample


@renderer
@pytest.mark.parametrize("abc", ["X:1\nK:C\n???", ABC.replace("K:C\n", "K:C\n\n"), "X:1\nK:C\nz8 |"])
@pytest.mark.asyncio
async def test_real_invalid_or_empty_music(abc):
    result = await server().verify(request(abc))
    assert result.reward == 0
    assert not result.mask_sample


@pytest.mark.asyncio
async def test_missing_renderer_masked():
    music_server = server()
    music_server._abc2midi = "/nonexistent-music-renderer"
    result = await music_server.verify(request(ABC))
    assert result.reward == 0
    assert result.mask_sample
    assert result.failure_kind == "mimo_music:scorer_error"
    assert "Abc2MidiMissing" in result.failure_reason


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("already_exited", [False, True])
@pytest.mark.asyncio
async def test_process_group_cleanup(cancel, already_exited):
    proc = MagicMock(pid=123)
    proc.communicate = AsyncMock(side_effect=asyncio.CancelledError() if cancel else TimeoutError())
    proc.wait = AsyncMock()
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)), patch("os.killpg") as kill:
        if already_exited:
            kill.side_effect = ProcessLookupError()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await server().verify(request(ABC))
        else:
            result = await server().verify(request(ABC))
            assert result.mask_sample
            assert result.failure_kind == "mimo_music:scorer_timeout"
        kill.assert_called_once_with(123, signal.SIGKILL)
        proc.wait.assert_awaited_once()


@pytest.mark.parametrize(
    "output", [b"not json", b'{"reward": 2, "scorer_details": {}}', b'{"reward": NaN, "scorer_details": {}}']
)
@pytest.mark.asyncio
async def test_bad_worker_output_masked(output):
    proc = MagicMock(returncode=0, communicate=AsyncMock(return_value=(output, b"")))
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        result = await server().verify(request(ABC))
    assert result.reward == 0
    assert result.mask_sample
    assert result.failure_kind == "mimo_music:scorer_error"


@pytest.mark.asyncio
async def test_concurrency_bound():
    active = peak = 0

    async def communicate(data):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return json.dumps({"reward": 0.5, "scorer_details": {}}).encode(), b""

    proc = MagicMock(returncode=0, communicate=communicate)
    music_server = server(num_processes=2)
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
        results = await asyncio.gather(*(music_server.verify(request(ABC)) for _ in range(12)))
    assert peak == 2
    assert all(r.reward == 0.5 and not r.mask_sample for r in results)


def test_http_verify_and_schema():
    with TestClient(server().setup_webserver()) as client:
        response = client.post("/verify", json=request("no music").model_dump())
        assert response.status_code == 200
        assert response.json()["reward"] == 0
        assert response.json()["mask_sample"] is False
        assert client.post("/verify", json={}).status_code == 422


@pytest.mark.asyncio
async def test_real_timeout_reaps_renderer(tmp_path):
    marker = tmp_path / "pid"
    executable = tmp_path / "sleep-renderer"
    executable.write_text(
        f"#!{sys.executable}\nimport os, time\n"
        f"open({str(marker)!r}, 'w').write(str(os.getpid()) + '\\n' + os.environ['TMPDIR'])\ntime.sleep(60)\n"
    )
    executable.chmod(0o755)
    music_server = server(score_timeout_seconds=1.0)
    music_server._abc2midi = str(executable)
    result = await music_server.verify(request(ABC))
    assert result.mask_sample
    assert result.failure_kind == "mimo_music:scorer_timeout"
    pid_text, scratch_dir = marker.read_text().splitlines()
    pid = int(pid_text)
    # An orphan may briefly be a zombie awaiting init; it must not be running.
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
    assert not Path(scratch_dir).exists()


@pytest.mark.parametrize(
    "kwargs", [{"num_processes": 0}, {"score_timeout_seconds": 0}, {"score_timeout_seconds": float("nan")}]
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        server(**kwargs)


def test_extraction_preserves_upstream_behavior():
    assert extract_abc(f"```abc\n{ABC}\n```\nX:2\nK:G\nG4") == ABC
    assert extract_abc("first X:1\nK:C\nC4\nlast X:2\nK:G\nG4") == "X:2\nK:G\nG4"
    assert extract_abc("no music") is None


@pytest.mark.parametrize("details", [{"skip": "abc2midi:TimeoutExpired"}, {"scorer_skip": "analyze:MemoryError"}])
def test_worker_infrastructure_error(details):
    with patch("resources_servers.mimo_music.score_worker.do", return_value=details), pytest.raises(RuntimeError):
        score_abc(ABC)


@pytest.mark.parametrize(
    "details,expected", [({"reject": 1, "total": 80}, 0), ({"skip": "no_midi"}, 0), ({"total": 63.1}, 0.631)]
)
def test_worker_native_reward(details, expected):
    with patch("resources_servers.mimo_music.score_worker.do", return_value=details):
        assert score_abc(ABC)["reward"] == expected
