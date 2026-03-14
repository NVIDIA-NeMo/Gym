# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.cvdp.app import (
    CVDPResourcesServer,
    CVDPResourcesServerConfig,
    _apply_substitutions,
    _build_bind_args,
    _build_command,
    _build_env_args,
    _load_dot_env,
    _parse_compose_service,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_COMPOSE = """
services:
  direct:
    image: __OSS_SIM_IMAGE__
    volumes:
      - ./src/:/src/:ro
    working_dir: /code/rundir
    command: /bin/sh -c "echo done"
"""

MINIMAL_HARNESS_FILES = {
    "docker-compose.yml": MINIMAL_COMPOSE,
    "src/.env": "SIM=icarus\nTOPLEVEL_LANG=verilog\nVERILOG_SOURCES=/code/rtl/foo.sv\nTOPLEVEL=foo\nMODULE=test_foo\n",
}

SAMPLE_RTL = "module foo(input clk);\nendmodule"


def _make_server() -> CVDPResourcesServer:
    config = CVDPResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cvdp",
        num_processes=1,
        container_timeout=30,
    )
    return CVDPResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_body(
    output_text: str,
    target_files: list | None = None,
    harness_files: dict | None = None,
) -> dict:
    if target_files is None:
        target_files = ["rtl/foo.sv"]
    if harness_files is None:
        harness_files = MINIMAL_HARNESS_FILES
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "Design a module"}]},
        "response": {
            "id": "resp_test",
            "created_at": 0.0,
            "model": "test-model",
            "object": "response",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": output_text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
        "verifier_metadata": {
            "task_id": "test_task_001",
            "categories": ["cid003", "medium"],
            "target_files": target_files,
            "harness_files": harness_files,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests: _apply_substitutions
# ---------------------------------------------------------------------------


class TestApplySubstitutions:
    def _make_config(self, oss_image="ghcr.io/hdl/sim/osvb", eda_image=""):
        return CVDPResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="cvdp",
            oss_sim_image=oss_image,
            eda_sim_image=eda_image,
        )

    def test_substitutes_oss_image(self):
        config = self._make_config()
        result = _apply_substitutions("image: __OSS_SIM_IMAGE__", config)
        assert "__OSS_SIM_IMAGE__" not in result
        assert "ghcr.io/hdl/sim/osvb" in result

    def test_substitutes_eda_image_when_set(self):
        config = self._make_config(eda_image="nvcr.io/internal/xcelium:latest")
        result = _apply_substitutions("image: __VERIF_EDA_IMAGE__", config)
        assert "xcelium" in result

    def test_no_error_when_placeholder_absent(self):
        config = self._make_config()
        result = _apply_substitutions("no placeholders here", config)
        assert result == "no placeholders here"


# ---------------------------------------------------------------------------
# Unit tests: _parse_compose_service
# ---------------------------------------------------------------------------


class TestParseComposeService:
    def test_extracts_image(self):
        svc = _parse_compose_service(MINIMAL_COMPOSE.replace("__OSS_SIM_IMAGE__", "ghcr.io/hdl/sim/osvb"), "direct")
        assert svc["image"] == "ghcr.io/hdl/sim/osvb"

    def test_extracts_command(self):
        svc = _parse_compose_service(MINIMAL_COMPOSE, "direct")
        assert svc["command"] == '/bin/sh -c "echo done"'

    def test_extracts_volumes(self):
        svc = _parse_compose_service(MINIMAL_COMPOSE, "direct")
        assert any("/src/" in v for v in svc["volumes"])

    def test_extracts_working_dir(self):
        svc = _parse_compose_service(MINIMAL_COMPOSE, "direct")
        assert svc["working_dir"] == "/code/rundir"

    def test_missing_service_returns_defaults(self):
        svc = _parse_compose_service(MINIMAL_COMPOSE, "nonexistent")
        assert svc["image"] == ""
        assert svc["command"] == ""
        assert svc["volumes"] == []


# ---------------------------------------------------------------------------
# Unit tests: _build_bind_args
# ---------------------------------------------------------------------------


class TestBuildBindArgs:
    def test_includes_code_mounts(self):
        args = _build_bind_args("/tmp/work", [])
        assert "--bind" in args
        assert "/tmp/work/rtl:/code/rtl" in args
        assert "/tmp/work/rundir:/code/rundir" in args
        assert "/tmp/work/src:/code/src" in args

    def test_includes_compose_volumes(self):
        args = _build_bind_args("/tmp/work", ["./src/:/src/:ro"])
        # Compose volume should be resolved relative to workdir
        assert any("/src/:ro" in a for a in args)

    def test_skips_code_volumes_from_compose(self):
        args = _build_bind_args("/tmp/work", ["./rtl:/code/rtl:ro"])
        # The /code/rtl from compose should be skipped (we mount it ourselves)
        code_rtl_compose = [a for a in args if a == "./rtl:/code/rtl:ro"]
        assert len(code_rtl_compose) == 0


# ---------------------------------------------------------------------------
# Unit tests: _build_env_args
# ---------------------------------------------------------------------------


class TestBuildEnvArgs:
    def test_empty_environment_returns_empty(self):
        args = _build_env_args({})
        assert args == []

    def test_dict_environment(self):
        args = _build_env_args({"SIM": "icarus", "TOPLEVEL": "foo"})
        assert "SIM=icarus" in args
        assert "TOPLEVEL=foo" in args

    def test_list_environment(self):
        args = _build_env_args(["SIM=icarus", "TOPLEVEL=foo"])
        assert "SIM=icarus" in args
        assert "TOPLEVEL=foo" in args

    def test_dot_env_vars_included(self):
        dot_env = {"VERILOG_SOURCES": "/code/rtl/foo.sv", "SIM": "icarus"}
        args = _build_env_args({}, dot_env)
        assert "VERILOG_SOURCES=/code/rtl/foo.sv" in args
        assert "SIM=icarus" in args

    def test_compose_env_overrides_dot_env(self):
        dot_env = {"SIM": "icarus"}
        args = _build_env_args({"SIM": "verilator"}, dot_env)
        # dot_env SIM comes first, then compose SIM overrides
        sim_values = [a for a in args if a.startswith("SIM=")]
        assert sim_values[-1] == "SIM=verilator"


class TestLoadDotEnv:
    def test_loads_env_file(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / ".env").write_text("SIM=icarus\nVERILOG_SOURCES=/code/rtl/foo.sv\n")
        env = _load_dot_env(str(tmp_path))
        assert env["SIM"] == "icarus"
        assert env["VERILOG_SOURCES"] == "/code/rtl/foo.sv"

    def test_skips_comments_and_blanks(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / ".env").write_text("# comment\n\nSIM=icarus\n")
        env = _load_dot_env(str(tmp_path))
        assert env == {"SIM": "icarus"}

    def test_missing_file_returns_empty(self, tmp_path):
        env = _load_dot_env(str(tmp_path))
        assert env == {}


# ---------------------------------------------------------------------------
# Unit tests: _build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_string_command(self):
        parts = _build_command(None, '/bin/sh -c "echo done"')
        assert parts == ["/bin/sh", "-c", "echo done"]

    def test_list_command(self):
        parts = _build_command(None, ["/bin/sh", "-c", "echo done"])
        assert parts == ["/bin/sh", "-c", "echo done"]

    def test_entrypoint_plus_command(self):
        parts = _build_command("/bin/sh", '-c "echo done"')
        assert parts[0] == "/bin/sh"
        assert "echo done" in parts

    def test_no_command_or_entrypoint(self):
        parts = _build_command(None, "")
        assert parts == []


# ---------------------------------------------------------------------------
# Integration-level server tests (Apptainer mocked at _run_harness)
# ---------------------------------------------------------------------------


class TestCVDPServerVerify:
    def setup_method(self):
        self.server = _make_server()

    @pytest.mark.asyncio
    async def test_verify_empty_output_returns_zero_reward(self):
        body_dict = _make_body(output_text="")
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0
        assert result.container_exit_code is None

    @pytest.mark.asyncio
    async def test_verify_no_code_returns_zero_reward(self):
        body_dict = _make_body(output_text="I am unable to generate this design.")
        with patch("resources_servers.cvdp.app._parse_model_response", return_value=None):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0
        assert result.extracted_rtl is None
        assert result.parse_failed is True

    @pytest.mark.asyncio
    async def test_verify_harness_pass_returns_one_reward(self):
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        with patch.object(
            self.server,
            "_run_harness",
            new_callable=AsyncMock,
            return_value=(0, "", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert result.container_exit_code == 0
        assert result.extracted_rtl is not None

    @pytest.mark.asyncio
    async def test_verify_harness_fail_returns_zero_reward(self):
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        with patch.object(
            self.server,
            "_run_harness",
            new_callable=AsyncMock,
            return_value=(1, "FAILED: assertion error", [{"service": "direct", "exit_code": 1, "stderr": "FAILED"}]),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0
        assert result.container_exit_code == 1

    @pytest.mark.asyncio
    async def test_verify_harness_timeout_returns_zero_reward(self):
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        with patch.object(
            self.server,
            "_run_harness",
            new_callable=AsyncMock,
            return_value=(-1, "apptainer exec timed out after 30s", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0
        assert result.container_exit_code == -1

    @pytest.mark.asyncio
    async def test_verify_missing_compose_returns_zero_reward(self):
        harness_no_compose = {"src/.env": "SIM=icarus\n"}
        body_dict = _make_body(
            output_text=f"```systemverilog\n{SAMPLE_RTL}\n```",
            harness_files=harness_no_compose,
        )
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_verify_multi_file_json_response(self):
        code_obj = {
            "code": [
                {"rtl/a.sv": "module a;\nendmodule"},
                {"rtl/b.sv": "module b;\nendmodule"},
            ]
        }
        body_dict = _make_body(
            output_text=json.dumps(code_obj),
            target_files=["rtl/a.sv", "rtl/b.sv"],
        )
        with patch.object(
            self.server,
            "_run_harness",
            new_callable=AsyncMock,
            return_value=(0, "", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert "rtl/a.sv" in result.extracted_rtl
        assert "rtl/b.sv" in result.extracted_rtl


# ---------------------------------------------------------------------------
# Apptainer-dependent tests (skipped if apptainer is not installed)
# ---------------------------------------------------------------------------

_SKIP_APPTAINER = pytest.mark.skipif(
    shutil.which("apptainer") is None,
    reason="apptainer not installed",
)


@_SKIP_APPTAINER
class TestApptainerHarness:
    """Smoke tests that require a real Apptainer installation."""

    def setup_method(self):
        self.server = _make_server()

    @pytest.mark.asyncio
    async def test_missing_compose_returns_error_exit_code(self):
        exit_code, stderr, services = await self.server._run_harness(
            rtl_files={"rtl/foo.sv": SAMPLE_RTL},
            harness_files={},  # no compose file
            task_id="test",
        )
        assert exit_code != 0


# ---------------------------------------------------------------------------
# Helpers for constructing request objects
# ---------------------------------------------------------------------------


def _make_request(body_dict: dict):
    """Build a CVDPVerifyRequest from a raw dict."""
    from resources_servers.cvdp.app import CVDPVerifyRequest

    return CVDPVerifyRequest.model_validate(body_dict)
