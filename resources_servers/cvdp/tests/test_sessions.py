# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent session lifecycle: the task sandbox is the local sandbox provider in a temporary directory."""

import asyncio
import dataclasses
import io
import os
import shutil
import sys
import tarfile
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


class _Runs(list):
    """The files each grading run received, and in ``context_files`` the context files it staged under them."""

    def __init__(self) -> None:
        super().__init__()
        self.context_files: list[dict[str, str]] = []


@pytest.fixture
def graded(monkeypatch) -> _Runs:
    """The files each grading run received; every run passes."""
    graded = _Runs()

    async def run(self, *, rtl_files, harness_files, task_id, context_files=None):
        graded.append(dict(rtl_files))
        graded.context_files.append(dict(context_files or {}))
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
    # As on cvdp_agent's agentic flavor: every changed text file under the project roots, but not unchanged context
    # files or files outside the roots (notes.txt).
    assert graded == [{"rtl/top.sv": TOP, "verif/tb.sv": TESTBENCH + "// more checks\n", "verif/sim.out": "compiled"}]
    assert verified["extracted_rtl"] == graded[0]
    assert (verified["task_id"], verified["category"]) == ("cvdp_agentic_codec_0001", "cid005")
    assert len(stopped) == 1, "verify stops the sandbox once the files are out"


def test_a_target_outside_the_project_roots_is_graded(root, graded):
    server = _server(root)
    metadata = _metadata() | {"target_files": ["top.sv"]}
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(task_data={"verifier_metadata": metadata}))
    (root / "top.sv").write_text(TOP)
    (root / "notes.txt").write_text("outside every root, and not a target")

    _verify(client, _verify_body() | {"verifier_metadata": metadata})

    assert graded == [{"top.sv": TOP}]


def test_an_agent_that_wrote_nothing_is_graded_on_its_response_text(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())

    verified = _verify(client, _verify_body("```systemverilog\nmodule top; endmodule\n```"))

    assert verified["reward"] == 1.0
    assert graded == [{"rtl/top.sv": "module top; endmodule\n"}], "as the legacy agentic path did"
    assert len(stopped) == 1


def test_a_response_with_no_text_still_grades_the_files(server, root, stopped, graded):
    # output_text joins every message in the trajectory, so this is an agent that never wrote any text. As on the
    # legacy path, the files it wrote are graded anyway.
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)

    verified = _verify(client, _verify_body(""))

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert graded == [{"rtl/top.sv": TOP, "verif/tb.sv": TESTBENCH + "// more checks\n", "verif/sim.out": "compiled"}]


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
    assert verified["evaluation_completed"] is False, "a masked harvest is not an evaluation"
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


def test_a_workspace_that_is_gone_is_masked_not_graded_on_the_response_text(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    # The sandbox lost its workspace mount: the environment failed, whatever the response text says.
    shutil.rmtree(root)

    verified = _verify(client, _verify_body(f"```systemverilog\n{TOP}```"))

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, VERIFIER_ERROR)
    assert verified["evaluation_completed"] is False
    assert "no workspace" in verified["failure_reason"]
    assert graded == []
    assert stopped == [], "the sandbox is kept for close to stop"


@pytest.mark.skipif(sys.platform == "darwin", reason="APFS refuses file names that are not UTF-8")
def test_a_file_name_that_is_not_utf8_is_left_out_instead_of_failing_the_verify(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    with open(os.path.join(os.fsencode(root / "rtl"), b"r\xe9sum\xe9.sv"), "w") as handle:
        handle.write("module resume; endmodule\n")

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert graded == [{"rtl/top.sv": TOP, "verif/tb.sv": TESTBENCH + "// more checks\n", "verif/sim.out": "compiled"}]


def test_files_that_changed_while_tar_read_them_still_grade(server, root, stopped, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _exec_result(monkeypatch, 1, "tar: rtl/top.sv: file changed as we read it")

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert graded[0]["rtl/top.sv"] == TOP
    assert len(stopped) == 1


@pytest.mark.parametrize(
    "stderr",
    [
        "tar: /code/.cvdp-harvest-0.tar: Wrote only 4096 of 10240 bytes",
        "tar: /code/.cvdp-harvest-0.tar: Cannot write: No space left on device",
    ],
)
def test_a_failed_archive_write_is_masked_and_keeps_the_sandbox(server, root, stopped, graded, monkeypatch, stderr):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _exec_result(monkeypatch, 2, stderr, run=False)

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, VERIFIER_ERROR)
    assert stderr in verified["failure_reason"]
    assert graded == [], "the archive may be missing files, so it is neither graded nor replaced by the text"
    assert stopped == [], "a resent verify can harvest again"


@pytest.mark.parametrize(("top", "reward"), [(TOP, 1.0), ("module top; endmodule\n", 0.0)])
def test_files_the_agent_made_unreadable_neither_fail_nor_mask_the_harvest(
    server, root, stopped, graded, monkeypatch, top, reward
):
    async def run(self, *, rtl_files, harness_files, task_id, context_files=None):
        graded.append(dict(rtl_files))
        return (0 if rtl_files.get("rtl/top.sv") == TOP else 1), "", []

    monkeypatch.setattr(app_module.TestbenchRunner, "run", run)
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    (root / "rtl" / "top.sv").write_text(top)
    (root / "rtl" / "scratch.bin").write_bytes(b"\0")
    (root / "rtl" / "sub").mkdir()
    (root / "rtl" / "sub" / "extra.sv").write_text("module extra; endmodule\n")
    for path in ("rtl/scratch.bin", "rtl/top.sv", "rtl/sub"):
        (root / path).chmod(0)

    verified = _verify(client)

    # A failed attempt stays a measured 0: leaving files unreadable cannot opt it out of scoring.
    assert (verified["reward"], verified["mask_sample"]) == (reward, False)
    # The harvest makes them readable again, so even the unreadable target is graded.
    assert graded == [
        {
            "rtl/scratch.bin": "\0",
            "rtl/sub/extra.sv": "module extra; endmodule\n",
            "rtl/top.sv": top,
            "verif/sim.out": "compiled",
            "verif/tb.sv": TESTBENCH + "// more checks\n",
        }
    ]
    assert len(stopped) == 1


def test_the_grader_does_not_restore_a_context_file_the_agent_deleted(server, root, stopped, graded):
    # As on cvdp_agent's agentic flavor: the grader rebuilds the workspace from the context files still there plus
    # the harvested ones, so a starting file the agent deleted or renamed stays gone, and a helper file it added
    # beside the target is graded with it.
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    (root / "rtl" / "encoder.sv").unlink()
    (root / "rtl" / "top.sv").write_text(TOP)
    (root / "rtl" / "defs.svh").write_text("`define WIDTH 8\n")

    verified = _verify(client)

    assert (verified["reward"], verified["mask_sample"]) == (1.0, False)
    assert graded == [{"rtl/defs.svh": "`define WIDTH 8\n", "rtl/top.sv": TOP}]
    # Hidden harness files and paths outside the workspace are never harvested, so they are always kept.
    assert set(graded.context_files[0]) == {
        "docs/specification.md",
        "verif/tb.sv",
        "src/test_top.py",
        "docker-compose.yml",
        "../escape.sv",
    }


def test_a_context_root_outside_the_default_globs_is_harvested(root, graded):
    server = _server(root)
    metadata = _metadata()
    metadata["context_files"] = metadata["context_files"] | {"include/defs.vh": "`define WIDTH 4\n"}
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed(task_data={"verifier_metadata": metadata}))
    _solve(root)
    (root / "include" / "defs.vh").write_text("`define WIDTH 8\n")
    (root / "rtl" / "rundir").mkdir()
    (root / "rtl" / "rundir" / "sim.log").write_text("simulator output")

    _verify(client, _verify_body() | {"verifier_metadata": metadata})

    assert graded[0]["include/defs.vh"] == "`define WIDTH 8\n"
    assert "rtl/rundir/sim.log" not in graded[0], "generated directories are never graded"
    assert graded.context_files[0]["include/defs.vh"] == "`define WIDTH 4\n"


def test_a_target_tar_stores_as_a_hard_link_is_graded(root, graded):
    server = _server(root)
    client = TestClient(server.setup_webserver())
    metadata = _metadata() | {"target_files": ["top.sv"]}
    _post_seed(client, _seed(task_data={"verifier_metadata": metadata}))
    _solve(root)
    # Archived after rtl/, so tar stores it as a hard link to rtl/top.sv.
    (root / "top.sv").hardlink_to(root / "rtl" / "top.sv")

    _verify(client, _verify_body() | {"verifier_metadata": metadata})

    assert graded[0]["top.sv"] == graded[0]["rtl/top.sv"] == TOP


def test_the_harvest_keeps_hard_links_only_to_files_it_extracts(tmp_path):
    archive = tmp_path / "workspace.tar"
    with tarfile.open(archive, "w") as tar:
        top = tarfile.TarInfo("rtl/top.sv")
        top.size = len(TOP)
        tar.addfile(top, io.BytesIO(TOP.encode()))
        for name, target in (("top.sv", "rtl/top.sv"), ("out.sv", "../outside.sv"), ("abs.sv", "/etc/hostname")):
            link = tarfile.TarInfo(name)
            link.type, link.linkname = tarfile.LNKTYPE, target
            tar.addfile(link)

    files, present = app_module._collect_harvest(
        archive, tmp_path / "workspace", ["rtl/**/*.sv"], {}, ["top.sv", "out.sv", "abs.sv"]
    )

    assert files == {"rtl/top.sv": TOP, "top.sv": TOP}
    assert present == {"rtl/top.sv", "top.sv"}


def test_a_workspace_with_none_of_the_harvested_paths_grades_the_response_text(server, root, stopped, graded):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    shutil.rmtree(root / "rtl")
    shutil.rmtree(root / "verif")

    verified = _verify(client)

    assert verified["mask_sample"] is False
    # No files: with one target file, the whole response text is graded as that file.
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
    assert verified["evaluation_completed"] is False
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


def test_ownership_outlives_a_verify_only_while_the_sandbox_is_unreleased(server, root, stopped, monkeypatch):
    """Checkpoint and worker ownership end at verify unless the session still holds a sandbox to stop."""
    client = TestClient(server.setup_webserver())
    seed = _seed()
    _post_seed(client, seed)
    _solve(root)
    assert server.owned_session_routing()["session_release_pending"] == server.session_release_pending
    real_exec = AsyncSandbox.exec

    _exec_result(monkeypatch, 2, "tar: /code/.cvdp-harvest-0.tar: Wrote only 4096 of 10240 bytes", run=False)
    assert _verify(client)["mask_sample"] is True
    assert server.session_release_pending("session"), "a failed harvest keeps the sandbox for a resent verify"

    monkeypatch.setattr(AsyncSandbox, "exec", real_exec)
    assert _verify(client)["mask_sample"] is False
    assert not server.session_release_pending("session"), "a stopped sandbox leaves nothing to release"
    assert not server.session_release_pending("unknown")


def test_retirement_stops_a_sandbox_that_a_failed_harvest_kept(server, root, stopped, monkeypatch):
    """A masked verify keeps the sandbox; with the close lost, retirement still stops it."""
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    _exec_result(monkeypatch, 2, "tar: /code/.cvdp-harvest-0.tar: Wrote only 4096 of 10240 bytes", run=False)
    assert _verify(client)["mask_sample"] is True
    assert stopped == [] and server.session_release_pending("session")

    asyncio.run(server.retire_session_state("session"))

    assert len(stopped) == 1
    assert server._sessions == {}
    assert not server.session_release_pending("session")


def test_retirement_stops_a_sandbox_whose_stop_failed_at_verify(server, root, graded, monkeypatch):
    client = TestClient(server.setup_webserver())
    _post_seed(client, _seed())
    _solve(root)
    real_stop = AsyncSandbox.stop
    monkeypatch.setattr(AsyncSandbox, "stop", AsyncMock(side_effect=OSError("stop failed")))
    assert _verify(client)["reward"] == 1.0
    assert server.session_release_pending("session")

    stopped: list[AsyncSandbox] = []

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    asyncio.run(server.retire_session_state("session"))

    assert len(stopped) == 1
    assert server._sessions == {}
    assert not server.session_release_pending("session")


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
        {"verifier_metadata": _metadata() | {"repository_workspace_sha256": "0" * 64}},
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


def test_a_seed_resent_after_failed_setup_and_cleanup_stops_the_old_sandbox_first(server, root, monkeypatch):
    events: list[tuple[str, AsyncSandbox]] = []
    real_start, real_stop = AsyncSandbox.start, AsyncSandbox.stop
    stop_failures = [OSError("stop failed"), OSError("stop failed again")]

    async def start(self, spec):
        events.append(("start", self))
        await real_start(self, spec)

    async def stop(self):
        events.append(("stop", self))
        if stop_failures:
            raise stop_failures.pop(0)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "start", start)
    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(side_effect=[OSError("serialize"), {"sandbox_id": "x"}]))
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    seed = _seed()

    assert _post_seed(client, seed).status_code == 500
    original = server._sessions["session"].sandbox
    assert original is not None and not server._sessions["session"].ready

    # The stop fails again: a retryable error, the original still owned, and nothing new started.
    assert _post_seed(client, seed).status_code == 500
    assert server._sessions["session"].sandbox is original
    assert [kind for kind, _ in events] == ["start", "stop", "stop"]

    resent = _post_seed(client, seed)
    assert resent.status_code == 200, resent.text
    assert events[3:] == [("stop", original), ("start", server._sessions["session"].sandbox)]
    assert server._sessions["session"].ready
    assert _post_seed(client, _seed(rollout_id="1-0")).status_code == 409, "another episode is still refused"


def test_shutdown_stops_the_sandboxes_of_open_sessions(server, root, stopped):
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed())

    assert len(stopped) == 1
    assert server._sessions == {}


def test_shutdown_stops_the_sandboxes_concurrently(server, root, monkeypatch):
    # A stopping server is killed about a second after it is asked to stop, so it must not stop them one by one.
    running, overlap = 0, 0
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        nonlocal running, overlap
        running += 1
        overlap = max(overlap, running)
        await asyncio.sleep(0.05)
        running -= 1
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    with TestClient(server.setup_webserver()) as client:
        _post_seed(client, _seed(session_id="first", rollout_id="0-0"))
        _post_seed(client, _seed(session_id="second", rollout_id="0-1"))
        assert len(server._sessions) == 2

    assert overlap == 2
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
