# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from nemo_gym.server_utils import ServerClient, SESSION_ID_KEY
from resources_servers.cvdp_agentic_heavy.app import (
    CVDPAgenticHeavyResourcesServer,
    CVDPAgenticHeavyConfig,
    CVDPAgenticHeavyVerifyRequest,
    SessionState,
    ToolRequest,
    _apply_substitutions,
    _filter_code_volumes,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SV = "module foo(input clk);\nendmodule"
COMPOSE = "services:\n  direct:\n    image: __OSS_SIM_IMAGE__\n    command: echo done\n"
HARNESS = {"docker-compose.yml": COMPOSE, "src/.env": "SIM=icarus\n"}
CTX = {"rtl/foo.sv": SV, "rtl/bar.sv": "module bar;\nendmodule", "docs/spec.md": "# Spec"}
META = {
    "task_id": "t1",
    "categories": ["cid016", "hard"],
    "difficulty": "hard",
    "context_files": CTX,
    "harness_files": HARNESS,
    "origin": {},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**kw):
    d = dict(
        host="0.0.0.0", port=8080, entrypoint="", name="ah",
        num_processes=1, execution_backend="apptainer", container_timeout=30, tool_timeout=30,
    )
    d.update(kw)
    return CVDPAgenticHeavyConfig(**d)


def _srv(**kw):
    return CVDPAgenticHeavyResourcesServer(
        config=_cfg(**kw), server_client=MagicMock(spec=ServerClient),
    )


def _req(sid="s1"):
    r = MagicMock()
    r.session = {SESSION_ID_KEY: sid}
    return r


def _vbody(sid="s1"):
    bd = {
        "responses_create_params": {"input": [{"role": "user", "content": "fix"}]},
        "response": {
            "id": "resp_test",
            "created_at": 1000000,
            "model": "test-model",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
        "verifier_metadata": META,
    }
    return _req(sid), CVDPAgenticHeavyVerifyRequest.model_validate(bd)


def _setup_sandbox(srv, sid="s1"):
    sb = Path(tempfile.mkdtemp())
    (sb / "rtl").mkdir()
    (sb / "rtl" / "foo.sv").write_text(SV)
    (sb / "rtl" / "bar.sv").write_text("module bar;\nendmodule")
    (sb / "docs").mkdir()
    (sb / "docs" / "spec.md").write_text("# Spec")
    srv._sessions[sid] = SessionState(sandbox=sb, container_id="")
    return sb


# ---------------------------------------------------------------------------
# Tests: _apply_substitutions
# ---------------------------------------------------------------------------


class TestApplySubstitutions:
    def test_oss_sim(self):
        assert "osvb" in _apply_substitutions("__OSS_SIM_IMAGE__", _cfg())

    def test_oss_pnr(self):
        assert "pnr" in _apply_substitutions("__OSS_PNR_IMAGE__", _cfg())

    def test_unknown_placeholder_is_unchanged(self):
        assert "__UNSUPPORTED_IMAGE__" == _apply_substitutions("__UNSUPPORTED_IMAGE__", _cfg())

    def test_noop(self):
        assert _apply_substitutions("hello", _cfg()) == "hello"



# ---------------------------------------------------------------------------
# Tests: backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_default_backend_is_apptainer(self):
        assert _srv()._backend() == "apptainer"

    def test_docker_backend_is_supported_for_compatibility(self):
        assert _srv(execution_backend="docker")._backend() == "docker"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            _srv(execution_backend="podman")._backend()


# ---------------------------------------------------------------------------
# Tests: _filter_code_volumes
# ---------------------------------------------------------------------------


class TestFilterCodeVolumes:
    def test_removes_code_volumes(self):
        c = "services:\n  s:\n    volumes:\n      - /code/x:/y\n      - ./src:/src\n"
        d = yaml.safe_load(_filter_code_volumes(c))
        assert not any("/code" in v for v in d["services"]["s"]["volumes"])

    def test_keeps_non_code(self):
        c = "services:\n  s:\n    volumes:\n      - ./src:/src\n"
        d = yaml.safe_load(_filter_code_volumes(c))
        assert any("/src" in v for v in d["services"]["s"]["volumes"])

    def test_no_volumes(self):
        c = "services:\n  s:\n    command: echo\n"
        assert "echo" in _filter_code_volumes(c)


# ---------------------------------------------------------------------------
# Tests: tool_ls
# ---------------------------------------------------------------------------


class TestToolLs:
    def setup_method(self):
        self.srv = _srv()
        self.sb = _setup_sandbox(self.srv)

    def teardown_method(self):
        shutil.rmtree(self.sb, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_root(self):
        r = await self.srv.tool_ls(_req(), ToolRequest(path="."))
        assert "rtl" in r.output

    @pytest.mark.asyncio
    async def test_subdir(self):
        r = await self.srv.tool_ls(_req(), ToolRequest(path="rtl"))
        assert "foo.sv" in r.output
        assert "bar.sv" in r.output

    @pytest.mark.asyncio
    async def test_missing_dir(self):
        r = await self.srv.tool_ls(_req(), ToolRequest(path="nope"))
        assert "Error" in r.output

    @pytest.mark.asyncio
    async def test_path_escape(self):
        r = await self.srv.tool_ls(_req(), ToolRequest(path="../../etc"))
        assert "Error" in r.output


# ---------------------------------------------------------------------------
# Tests: tool_cat
# ---------------------------------------------------------------------------


class TestToolCat:
    def setup_method(self):
        self.srv = _srv()
        self.sb = _setup_sandbox(self.srv)

    def teardown_method(self):
        shutil.rmtree(self.sb, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_read_file(self):
        r = await self.srv.tool_cat(_req(), ToolRequest(filename="rtl/foo.sv"))
        assert "module foo" in r.output

    @pytest.mark.asyncio
    async def test_missing_file(self):
        r = await self.srv.tool_cat(_req(), ToolRequest(filename="nope.sv"))
        assert "Error" in r.output

    @pytest.mark.asyncio
    async def test_no_filename(self):
        r = await self.srv.tool_cat(_req(), ToolRequest())
        assert "Error" in r.output

    @pytest.mark.asyncio
    async def test_path_escape(self):
        r = await self.srv.tool_cat(_req(), ToolRequest(filename="../../etc/passwd"))
        assert "Error" in r.output


# ---------------------------------------------------------------------------
# Tests: tool_echo
# ---------------------------------------------------------------------------


class TestToolEcho:
    def setup_method(self):
        self.srv = _srv()
        self.sb = _setup_sandbox(self.srv)

    def teardown_method(self):
        shutil.rmtree(self.sb, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_create_file(self):
        r = await self.srv.tool_echo(
            _req(), ToolRequest(content="module x;\nendmodule", filename="rtl/x.sv"),
        )
        assert "Written" in r.output
        assert (self.sb / "rtl" / "x.sv").exists()
        assert "module x" in (self.sb / "rtl" / "x.sv").read_text()

    @pytest.mark.asyncio
    async def test_overwrite_file(self):
        (self.sb / "t.txt").write_text("old")
        await self.srv.tool_echo(_req(), ToolRequest(content="new", filename="t.txt"))
        assert (self.sb / "t.txt").read_text() == "new"

    @pytest.mark.asyncio
    async def test_creates_parent_dirs(self):
        await self.srv.tool_echo(_req(), ToolRequest(content="d", filename="a/b/c.txt"))
        assert (self.sb / "a" / "b" / "c.txt").exists()

    @pytest.mark.asyncio
    async def test_no_filename(self):
        r = await self.srv.tool_echo(_req(), ToolRequest(content="x"))
        assert "Error" in r.output

    @pytest.mark.asyncio
    async def test_path_escape(self):
        r = await self.srv.tool_echo(
            _req(), ToolRequest(content="x", filename="../../bad"),
        )
        assert "Error" in r.output


# ---------------------------------------------------------------------------
# Tests: tool_pwd
# ---------------------------------------------------------------------------


class TestToolPwd:
    def setup_method(self):
        self.srv = _srv()
        self.sb = _setup_sandbox(self.srv)

    def teardown_method(self):
        shutil.rmtree(self.sb, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_returns_code(self):
        r = await self.srv.tool_pwd(_req(), ToolRequest())
        assert r.output == "/code"



# ---------------------------------------------------------------------------
# Tests: simulation tools
# ---------------------------------------------------------------------------


class TestSimulationTools:
    def setup_method(self):
        self.srv = _srv()
        self.sb = _setup_sandbox(self.srv)

    def teardown_method(self):
        shutil.rmtree(self.sb, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_iverilog_uses_argument_vector(self):
        with patch.object(self.srv, "_tool_exec", new_callable=AsyncMock, return_value="ok") as mock_exec:
            r = await self.srv.tool_iverilog(_req(), ToolRequest(args="-g2012 -o sim.vvp rtl/foo.sv"))
        assert r.output == "ok"
        argv = mock_exec.await_args.args[1]
        assert argv == ["iverilog", "-g2012", "-o", "sim.vvp", "rtl/foo.sv"]

    @pytest.mark.asyncio
    async def test_vvp_rejects_shell_redirection(self):
        r = await self.srv.tool_vvp(_req(), ToolRequest(filename="sim.vvp > out.txt"))
        assert "single simulation file path" in r.output


# ---------------------------------------------------------------------------
# Tests: no session error
# ---------------------------------------------------------------------------


class TestNoSession:
    @pytest.mark.asyncio
    async def test_tool_without_session_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await _srv().tool_ls(_req("bad"), ToolRequest(path="."))
        assert exc_info.value.status_code == 400
        assert "No session" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Tests: verify
# ---------------------------------------------------------------------------


class TestVerify:
    def setup_method(self):
        self.srv = _srv()

    @pytest.mark.asyncio
    async def test_no_sandbox_returns_zero(self):
        req, body = _vbody("bad")
        r = await self.srv.verify(req, body)
        assert r.reward == 0.0
        assert r.container_exit_code == 1
        assert r.docker_exit_code == 1

    @pytest.mark.asyncio
    async def test_container_pass(self):
        _setup_sandbox(self.srv)
        req, body = _vbody()
        with patch.object(
            self.srv, "_run_container_harness",
            new_callable=AsyncMock, return_value=(0, "", []),
        ):
            r = await self.srv.verify(req, body)
        assert r.reward == 1.0
        assert r.task_id == "t1"
        assert r.category == "cid016"
        assert r.difficulty == "hard"

    @pytest.mark.asyncio
    async def test_container_fail(self):
        _setup_sandbox(self.srv)
        req, body = _vbody()
        with patch.object(
            self.srv, "_run_container_harness",
            new_callable=AsyncMock, return_value=(1, "FAIL", []),
        ):
            r = await self.srv.verify(req, body)
        assert r.reward == 0.0
        assert r.container_exit_code == 1
        assert r.docker_exit_code == 1

    @pytest.mark.asyncio
    async def test_cleans_up_session(self):
        _setup_sandbox(self.srv)
        req, body = _vbody()
        with patch.object(
            self.srv, "_run_container_harness",
            new_callable=AsyncMock, return_value=(0, "", []),
        ):
            await self.srv.verify(req, body)
        assert "s1" not in self.srv._sessions

    @pytest.mark.asyncio
    async def test_commercial_eda_placeholder_is_rejected(self):
        sb = _setup_sandbox(self.srv)
        compose = """services:
  xrun:
    image: __VERIF_EDA_IMAGE__
    command: true
"""
        try:
            exit_code, stderr, services = await self.srv._run_container_harness(
                sandbox=sb,
                harness_files={"docker-compose.yml": compose},
                task_id="eda_only",
            )
        finally:
            shutil.rmtree(sb, ignore_errors=True)
        assert exit_code == 1
        assert services == []
        assert "Commercial EDA" in stderr


# ---------------------------------------------------------------------------
# Tests: end-to-end (seed -> tools -> verify, harness mocked)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_episode(self):
        srv = _srv()
        sid = "e2e"
        req = _req(sid)

        # Seed: create sandbox with context files
        sb = _setup_sandbox(srv, sid)

        # Agent reads a file
        r = await srv.tool_cat(req, ToolRequest(filename="rtl/foo.sv"))
        assert "module foo" in r.output

        # Agent lists directory
        r = await srv.tool_ls(req, ToolRequest(path="rtl"))
        assert "foo.sv" in r.output
        assert "bar.sv" in r.output

        # Agent writes a fix
        fixed = "module foo_fixed(input clk);\nendmodule"
        r = await srv.tool_echo(req, ToolRequest(content=fixed, filename="rtl/foo.sv"))
        assert "Written" in r.output
        assert "foo_fixed" in (sb / "rtl" / "foo.sv").read_text()

        # Verify (container harness mocked)
        vreq, vbody = _vbody(sid)
        with patch.object(
            srv, "_run_container_harness",
            new_callable=AsyncMock, return_value=(0, "", []),
        ):
            r = await srv.verify(vreq, vbody)

        assert r.reward == 1.0
        assert sid not in srv._sessions


# ---------------------------------------------------------------------------
# Tests: SessionState cleanup
# ---------------------------------------------------------------------------


class TestSessionState:
    @pytest.mark.asyncio
    async def test_cleanup_removes_sandbox(self):
        sb = Path(tempfile.mkdtemp())
        (sb / "t.txt").write_text("x")
        session = SessionState(sandbox=sb, container_id="")
        await session.cleanup()
        assert not sb.exists()
