# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent session lifecycle: the task sandbox is the local sandbox provider in a temporary directory."""

import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import PROVIDER_IO_TIMEOUT, PROVIDER_REQUEST_FAILED, SESSION_LOST, VERIFIER_ERROR, EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, sandbox_failure
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.cvdp import app as app_module
from resources_servers.cvdp.app import CVDPResourcesServer, CVDPResourcesServerConfig


SPEC = "# The codec\n"
ENCODER = "module encoder; endmodule\n"
TESTBENCH = "module tb; endmodule\n"
TOP = "module top; encoder u(); endmodule\n"


def _metadata() -> dict:
    return {
        "task_id": "cvdp_agentic_codec_0001",
        "categories": ["cid005", "medium"],
        "target_files": ["rtl/top.sv"],
        "harness_files": {
            "docker-compose.yml": "services: {}\n",
            "src/test_top.py": "hidden test",
            "src/.env": "SIM=icarus\n",
        },
        "context_files": {
            "docs/specification.md": SPEC,
            "rtl/encoder.sv": ENCODER,
            "verif/tb.sv": TESTBENCH,
            # Never staged: harness files, and paths outside the workspace.
            "src/test_top.py": "hidden test",
            "docker-compose.yml": "services: {}\n",
            "../escape.sv": "outside",
        },
    }


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {"sandbox": {"local": {}}})
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "code"


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    """Every sandbox stopped during the test, in order."""
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _server(root: Path, **extra) -> CVDPResourcesServer:
    config = CVDPResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cvdp_agentic_resources_server",
        num_processes=1,
        agent_sandbox_provider="sandbox",
        agent_sandbox_config={"image": "unused-by-the-local-provider"},
        agent_workdir=str(root),
        **extra,
    )
    return CVDPResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def graded(monkeypatch) -> list[dict[str, str]]:
    """The files each grading run received; every run passes."""
    graded: list[dict[str, str]] = []

    async def run(self, *, rtl_files, harness_files, task_id, context_files=None):
        graded.append(dict(rtl_files))
        return 0, "", []

    monkeypatch.setattr(app_module.TestbenchRunner, "run", run)
    return graded


@pytest.fixture
def server(root, graded) -> CVDPResourcesServer:
    return _server(root)


def _seed(*, session_id: str = "session", rollout_id: str = "0-0", task_data=None) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout_id),
        task_id=TaskId(taskset="cvdp:example_agentic", task_id="cvdp_agentic_codec_0001"),
        task_data={"verifier_metadata": _metadata()} if task_data is None else task_data,
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify_body(final_message: str = "I wrote rtl/top.sv.") -> dict:
    return {
        "verifier_metadata": _metadata(),
        "responses_create_params": {"input": [{"role": "user", "content": "Write rtl/top.sv"}]},
        "response": {
            "id": "resp_1",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": final_message, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    }


def _verify(client: TestClient, body: dict | None = None) -> dict:
    response = client.post("/verify", json=body or _verify_body())
    assert response.status_code == 200, response.text
    return response.json()


def _close(client: TestClient, seed: ResourcesSeedSessionRequest, episode_id: EpisodeId | None = None):
    close = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=episode_id or seed.episode_id
    )
    return client.post("/close_session", json=close.model_dump(mode="json"))


def _solve(root: Path) -> None:
    """What an agent leaves behind: the target, an edited testbench, scratch files and a simulator build."""
    (root / "rtl" / "top.sv").write_text(TOP)
    (root / "verif" / "tb.sv").write_text(TESTBENCH + "// more checks\n")
    (root / "verif" / "sim.out").write_text("compiled")
    (root / "notes.txt").write_text("scratch")


def test_seed_lends_a_sandbox_with_the_context_files_but_not_the_harness(server, root):
    client = TestClient(server.setup_webserver())

    seeded = _post_seed(client, _seed())

    assert seeded.status_code == 200, seeded.text
    access = seeded.json()["sandbox_access"]
    assert access["workdir"] == str(root)
    assert access["connection"]["provider_config_ref"] == "sandbox"
    staged = {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*") if p.is_file()}
    assert staged == {"docs/specification.md": SPEC, "rtl/encoder.sv": ENCODER, "verif/tb.sv": TESTBENCH}
    assert not (root.parent / "escape.sv").exists()


def test_verify_grades_the_files_the_agent_changed_and_stops_the_sandbox(server, root, stopped, graded, caplog):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert "Masking" not in caplog.text, "a graded verify logs no lost session"
    # Unchanged context files, files outside the globs and non-RTL files are not graded.
    assert graded == [{"rtl/top.sv": TOP, "verif/tb.sv": TESTBENCH + "// more checks\n"}]
    assert verified["extracted_rtl"] == graded[0]
    assert (verified["task_id"], verified["category"]) == ("cvdp_agentic_codec_0001", "cid005")
    assert len(stopped) == 1, "verify stops the sandbox once the files are out"


def test_a_target_outside_the_harvested_directories_is_graded(root, graded):
    server = _server(root, harvest_globs=["rtl/**/*.sv"])
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    (root / "verif" / "tb.sv").write_text("edited, but not harvested")

    _verify(client)

    assert graded == [{"rtl/top.sv": TOP}]


def test_an_agent_that_wrote_nothing_is_graded_on_its_final_message(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())

    verified = _verify(client, _verify_body("```systemverilog\nmodule top; endmodule\n```"))

    assert verified["reward"] == 1.0
    assert graded == [{"rtl/top.sv": "module top; endmodule\n"}], "as the legacy agentic path did"
    assert len(stopped) == 1


def test_an_empty_final_message_scores_zero_even_with_files(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)

    verified = _verify(client, _verify_body(""))

    assert (verified["reward"], verified["mask_sample"]) == (0.0, False)
    assert graded == [], "the empty-answer rule is unchanged from the legacy paths"


def test_a_repeated_verify_grades_the_harvested_files_again(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _verify(client)

    again = _verify(client)

    assert again["reward"] == 1.0
    assert len(stopped) == 1
    assert graded[0] == graded[1]


def _exec_result(monkeypatch, return_code: int, stderr: str = "", *, run: bool = True, error_type=None) -> None:
    """Make the harvest command report ``return_code``; ``run`` still runs it, so its archive exists."""
    real_exec = AsyncSandbox.exec

    async def exec_(self, command, **kwargs):
        result = await real_exec(self, command, **kwargs) if run else SandboxExecResult("", "", 0)
        return dataclasses.replace(result, return_code=return_code, stderr=stderr, error_type=error_type)

    monkeypatch.setattr(AsyncSandbox, "exec", exec_)


def test_a_failed_download_is_masked_and_a_resent_verify_harvests_again(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    real_download = AsyncSandbox.download
    monkeypatch.setattr(AsyncSandbox, "download", AsyncMock(side_effect=OSError("download failed")))

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (0.0, True)
    assert verified["failure_kind"] == PROVIDER_REQUEST_FAILED
    assert "download failed" in verified["failure_reason"]
    assert verified["task_id"] == "cvdp_agentic_codec_0001"
    assert stopped == [], "the sandbox is kept, so the files can still be harvested"
    assert graded == []

    monkeypatch.setattr(AsyncSandbox, "download", real_download)
    again = _verify(client)

    assert (again["reward"], again["mask_sample"]) == (1.0, False)
    assert len(stopped) == 1


def test_a_typed_provider_failure_keeps_its_failure_kind(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    timeout = sandbox_failure(operation="exec", failure_kind=PROVIDER_IO_TIMEOUT, reason="exec timed out")
    monkeypatch.setattr(AsyncSandbox, "exec", AsyncMock(side_effect=timeout))

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, PROVIDER_IO_TIMEOUT)
    assert stopped == []


def test_a_command_the_provider_did_not_run_is_masked(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _exec_result(monkeypatch, 125, run=False, error_type="user_switch_unavailable")

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, PROVIDER_REQUEST_FAILED)
    assert stopped == []


def test_a_harvest_command_that_is_not_tar_s_outcome_is_masked(server, root, stopped, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _exec_result(monkeypatch, 127, "tar: command not found", run=False)

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, VERIFIER_ERROR)
    assert "command not found" in verified["failure_reason"]
    assert stopped == []


def test_files_that_changed_while_tar_read_them_still_grade(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _exec_result(monkeypatch, 1, "tar: rtl/top.sv: file changed as we read it")

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert graded[0]["rtl/top.sv"] == TOP
    assert len(stopped) == 1


def test_a_fatal_tar_error_grades_the_final_message(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _exec_result(monkeypatch, 2, "tar: rtl/top.sv: Cannot open: Permission denied", run=False)

    verified = _verify(client)

    assert verified["mask_sample"] is False
    # No files: with one target file, the whole final message is graded as that file.
    assert graded == [{"rtl/top.sv": "I wrote rtl/top.sv."}]
    assert len(stopped) == 1


def test_close_stops_an_unverified_sandbox_and_fences_late_seeds(server, root, stopped):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)

    assert _close(client, seed).status_code == 200

    assert len(stopped) == 1
    assert server._sessions == {}
    late = _post_seed(client, seed)
    assert late.status_code == 409
    assert _close(client, seed).status_code == 200, "a repeated close of the same episode is idempotent"
    assert _close(client, seed, episode_id=EpisodeId(rollout_id="0-1")).status_code == 409


def test_a_verify_after_close_is_masked_as_a_lost_session(server, root, stopped, graded, caplog):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _solve(root)
    assert _close(client, seed).status_code == 200

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert graded == []
    assert "session_lost" in caplog.text


def test_a_verify_without_a_session_is_masked_but_rtl_files_still_grade(server, graded):
    client = TestClient(server.setup_webserver())

    verified = _verify(client)

    assert (verified["mask_sample"], verified["failure_kind"]) == (True, SESSION_LOST)
    legacy = _verify(client, _verify_body() | {"rtl_files": {"rtl/top.sv": TOP}})
    assert (legacy["reward"], legacy["mask_sample"]) == (1.0, False)
    assert graded == [{"rtl/top.sv": TOP}], "cvdp_agent's agentic flavor sends the files it harvested itself"


def test_close_after_verify_does_not_stop_the_sandbox_twice(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _verify(client)

    assert _close(client, seed).status_code == 200

    assert len(stopped) == 1


def test_a_stop_that_fails_after_the_harvest_still_grades_and_close_stops_the_sandbox(
    server, root, graded, monkeypatch
):
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _solve(root)
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=OSError("stop failed")))

    assert _verify(client)["reward"] == 1.0
    assert server._sessions["session"].sandbox is not None
    assert _post_seed(client, seed).status_code == 409, "a harvested sandbox is not lent again"

    monkeypatch.setattr(AsyncSandbox, "stop", real_stop)
    assert _close(client, seed).status_code == 200
    assert server._sessions == {}


def test_seeding_again_is_idempotent_for_its_episode_and_refused_for_another(server, root, stopped):
    client = TestClient(server.setup_webserver())
    seed = _seed()

    first = _post_seed(client, seed)
    again = _post_seed(client, seed)
    other = _post_seed(client, _seed(rollout_id="1-0"))

    assert (first.status_code, again.status_code, other.status_code) == (200, 200, 409)
    assert again.json()["sandbox_access"] == first.json()["sandbox_access"]
    assert stopped == []


@pytest.mark.parametrize(
    "task_data",
    [
        {},
        {"verifier_metadata": {"categories": ["cid005", "medium"]}},
        {"verifier_metadata": _metadata() | {"categories": ["medium"]}},
        {"verifier_metadata": _metadata() | {"categories": ["005", "medium"]}},
    ],
)
def test_a_row_that_no_retry_can_fix_is_refused_as_final_without_a_sandbox(server, root, task_data, monkeypatch):
    start = AsyncMock()
    monkeypatch.setattr(AsyncSandbox, "start", start)
    client = TestClient(server.setup_webserver())

    refused = _post_seed(client, _seed(task_data=task_data))

    assert refused.status_code == 422, refused.text
    start.assert_not_called()
    assert server._sessions == {}


def test_a_failed_start_stops_the_sandbox_and_keeps_nothing(server, root, stopped, monkeypatch):
    monkeypatch.setattr(AsyncSandbox, "start", AsyncMock(side_effect=OSError("no image")))
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    assert _post_seed(client, _seed()).status_code == 500

    assert len(stopped) == 1
    assert server._sessions == {}


def test_shutdown_stops_the_sandboxes_of_open_sessions(server, root, stopped):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed())

    assert len(stopped) == 1
    assert server._sessions == {}


def test_sessions_require_a_single_worker(root):
    with pytest.raises(ValueError, match="num_workers=1"):
        _server(root, num_workers=2)


def test_close_follows_the_session_contract(server, root, graded):
    check_resources_session_contract(
        server.setup_webserver(),
        _seed(session_id="contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._sessions,
        verify_body=_verify_body(),
    )


def test_without_a_sandbox_provider_seeds_are_stateless_and_verify_grades_the_request(root, graded):
    config = CVDPResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="cvdp", num_processes=1)
    server = CVDPResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    typed = _post_seed(client, _seed())
    legacy = client.post("/seed_session", json={"verifier_metadata": _metadata(), "responses_create_params": {}})
    verified = _verify(client, _verify_body("```systemverilog\nmodule top; endmodule\n```"))

    assert typed.status_code == 200 and typed.json().get("sandbox_access") is None
    assert legacy.status_code == 200, legacy.text
    assert verified["reward"] == 1.0
    assert graded == [{"rtl/top.sv": "module top; endmodule\n"}]
    assert server._sessions == {}
