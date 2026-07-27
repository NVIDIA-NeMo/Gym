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

from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.cvdp.app import (
    CVDPResourcesServer,
    CVDPResourcesServerConfig,
    _parse_model_response,
)
from resources_servers.cvdp.cvdp_lib.subjective import calculate_BLEU, calculate_ROUGE
from resources_servers.cvdp.testbench_runner import (
    _apply_substitutions,
    _build_binds,
    _build_command,
    _build_env,
    _build_runtime_tmp_env,
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
# Unit tests: _parse_model_response
# ---------------------------------------------------------------------------


class TestParseModelResponse:
    def test_extracts_systemverilog_fence(self):
        text = "Here is the code:\n```systemverilog\nmodule foo;\nendmodule\n```"
        result = _parse_model_response(text, ["rtl/foo.sv"])
        assert result is not None
        assert "module foo" in result["rtl/foo.sv"]

    def test_extracts_verilog_fence(self):
        text = "```verilog\nmodule bar;\nendmodule\n```"
        result = _parse_model_response(text, ["rtl/bar.v"])
        assert result is not None
        assert "module bar" in result["rtl/bar.v"]

    def test_plain_text_treated_as_code(self):
        text = "I cannot generate this design."
        result = _parse_model_response(text, ["rtl/foo.sv"])
        assert result is not None
        assert result["rtl/foo.sv"] == text

    def test_returns_none_for_empty_target_files(self):
        result = _parse_model_response("some text", [])
        assert result is None

    def test_extracts_multi_file_json_format(self):
        code_obj = {
            "code": [
                {"rtl/a.sv": "module a;\nendmodule"},
                {"rtl/b.sv": "module b;\nendmodule"},
            ]
        }
        text = json.dumps(code_obj)
        result = _parse_model_response(text, ["rtl/a.sv", "rtl/b.sv"])
        assert result is not None
        assert "module a" in result["rtl/a.sv"]
        assert "module b" in result["rtl/b.sv"]


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
# Unit tests: _load_dot_env
# ---------------------------------------------------------------------------


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
    async def test_verify_plain_text_goes_to_harness(self):
        body_dict = _make_body(output_text="I am unable to generate this design.")
        with patch.object(
            self.server._harness,
            "run",
            new_callable=AsyncMock,
            return_value=(1, "FAILED", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0
        assert result.extracted_rtl is not None
        assert result.parse_failed is False

    @pytest.mark.asyncio
    async def test_verify_harness_pass_returns_one_reward(self):
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        with patch.object(
            self.server._harness,
            "run",
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
            self.server._harness,
            "run",
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
            self.server._harness,
            "run",
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
            self.server._harness,
            "run",
            new_callable=AsyncMock,
            return_value=(0, "", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert "rtl/a.sv" in result.extracted_rtl
        assert "rtl/b.sv" in result.extracted_rtl


# ---------------------------------------------------------------------------
# Unit tests: subjective scoring (BLEU / ROUGE)
# ---------------------------------------------------------------------------


class TestSubjectiveScoring:
    def test_bleu_identical_strings(self):
        score = calculate_BLEU("the cat sat on the mat", "the cat sat on the mat", 2)
        assert score == pytest.approx(1.0)

    def test_bleu_completely_different(self):
        score = calculate_BLEU("alpha beta gamma", "one two three four five six", 2)
        assert score == pytest.approx(0.0)

    def test_rouge_identical_strings(self):
        score = calculate_ROUGE("the cat sat on the mat", "the cat sat on the mat", 2)
        assert score == pytest.approx(1.0)

    def test_rouge_completely_different(self):
        score = calculate_ROUGE("alpha beta gamma", "one two three four five six", 2)
        assert score == pytest.approx(0.0)

    def test_bleu_partial_overlap(self):
        score = calculate_BLEU("the cat sat on the mat", "the cat slept on a mat", 2)
        assert 0.0 < score < 1.0

    def test_rouge_partial_overlap(self):
        score = calculate_ROUGE("the cat sat on the mat", "the cat slept on a mat", 2)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Integration: subjective category verify flow
# ---------------------------------------------------------------------------


def _make_subjective_body(
    output_text: str,
    category: str = "cid010",
    difficulty: str = "easy",
    subjective_reference: str = "This is the reference answer.",
) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "Explain the barrel shifter"}]},
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
            "task_id": "test_subjective_001",
            "categories": [category, difficulty],
            "target_files": [],
            "harness_files": {},
            "subjective_reference": subjective_reference,
        },
    }


class TestCVDPServerVerifySubjective:
    def setup_method(self):
        self.server = _make_server()

    @pytest.mark.asyncio
    async def test_identical_answer_gets_high_reward(self):
        ref = "Testing circular shifts with shift_bits equal to DATA_WIDTH checks correctness."
        body_dict = _make_subjective_body(output_text=ref, subjective_reference=ref)
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == pytest.approx(1.0)
        assert result.bleu_score == pytest.approx(1.0)
        assert result.rouge_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_empty_output_returns_zero(self):
        body_dict = _make_subjective_body(output_text="", subjective_reference="some reference")
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_no_reference_returns_zero(self):
        body_dict = _make_subjective_body(output_text="some answer", subjective_reference="")
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_partial_match_returns_fractional_reward(self):
        ref = "The barrel shifter rotates bits in a circular fashion using multiplexers."
        answer = "The barrel shifter rotates bits using a series of multiplexers for shifting."
        body_dict = _make_subjective_body(output_text=answer, subjective_reference=ref)
        result = await self.server.verify(_make_request(body_dict))
        assert 0.0 < result.reward < 1.0
        assert result.bleu_score is not None
        assert result.rouge_score is not None

    @pytest.mark.asyncio
    async def test_completely_wrong_answer_gets_low_reward(self):
        ref = "The barrel shifter rotates bits in a circular fashion using multiplexers."
        answer = "Quantum computing leverages entanglement for parallel processing."
        body_dict = _make_subjective_body(output_text=answer, subjective_reference=ref)
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward < 0.3

    @pytest.mark.asyncio
    async def test_category_6_uses_subjective_path(self):
        ref = "module foo implements a decoder."
        body_dict = _make_subjective_body(output_text=ref, category="cid006", subjective_reference=ref)
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == pytest.approx(1.0)
        assert result.bleu_score is not None

    @pytest.mark.asyncio
    async def test_category_8_uses_subjective_path(self):
        ref = "The testbench checks reset behavior."
        body_dict = _make_subjective_body(output_text=ref, category="cid008", subjective_reference=ref)
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_code_gen_category_does_not_use_subjective(self):
        """Category 3 (code gen) should go to the harness path, not subjective."""
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        body_dict["verifier_metadata"]["categories"] = ["cid003", "medium"]
        with patch.object(
            self.server._harness,
            "run",
            new_callable=AsyncMock,
            return_value=(0, "", []),
        ):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert result.bleu_score is None  # Not a subjective category


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
        exit_code, stderr, services = await self.server._harness.run(
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


# ---------------------------------------------------------------------------
# Unit tests: agentic rtl_files path (grade files written on disk)
# ---------------------------------------------------------------------------


class TestVerifyConsumesRtlFiles:
    def setup_method(self):
        self.server = _make_server()

    @pytest.mark.asyncio
    async def test_rtl_files_take_precedence_over_text_parse(self):
        # Model chat text says one thing; the agent reports different files on
        # disk. The on-disk files must win.
        body_dict = _make_body(output_text="```systemverilog\nmodule from_text;\nendmodule\n```")
        body_dict["rtl_files"] = {"rtl/foo.sv": "module from_disk;\nendmodule"}
        captured = {}

        async def fake_harness(*, rtl_files, harness_files, task_id, context_files=None):
            captured["rtl_files"] = rtl_files
            return (0, "", [])

        with patch.object(self.server._harness, "run", side_effect=fake_harness):
            result = await self.server.verify(_make_request(body_dict))

        assert result.reward == 1.0
        assert result.extracted_rtl == {"rtl/foo.sv": "module from_disk;\nendmodule"}
        assert captured["rtl_files"] == {"rtl/foo.sv": "module from_disk;\nendmodule"}

    @pytest.mark.asyncio
    async def test_falls_back_to_text_parse_without_rtl_files(self):
        body_dict = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
        # No rtl_files key -> text parsing path.
        with patch.object(self.server._harness, "run", new_callable=AsyncMock, return_value=(0, "", [])):
            result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert "module foo" in result.extracted_rtl["rtl/foo.sv"]


# ---------------------------------------------------------------------------
# Unit tests: provider-facing bind/env helpers
# ---------------------------------------------------------------------------


class TestBuildBinds:
    def test_includes_code_mounts(self):
        binds = _build_binds("/tmp/work", [])
        assert "/tmp/work/rtl:/code/rtl" in binds
        assert "/tmp/work/rundir:/code/rundir" in binds
        assert "/tmp/work/src:/code/src" in binds

    def test_includes_compose_volumes_resolved(self):
        binds = _build_binds("/tmp/work", ["./extra:/data:ro"])
        assert "/tmp/work/extra:/data:ro" in binds

    def test_skips_code_volumes_from_compose(self):
        binds = _build_binds("/tmp/work", ["./rtl:/code/rtl:ro"])
        assert all("/code/rtl:ro" not in b or b == "/tmp/work/rtl:/code/rtl" for b in binds)


class TestBuildEnv:
    def test_dict_environment(self):
        env = _build_env({"SIM": "icarus"})
        assert env["SIM"] == "icarus"

    def test_list_environment(self):
        env = _build_env(["SIM=icarus", "TOPLEVEL=foo"])
        assert env == {"SIM": "icarus", "TOPLEVEL": "foo"}

    def test_compose_overrides_dot_env(self):
        env = _build_env({"SIM": "verilator"}, {"SIM": "icarus", "X": "1"})
        assert env["SIM"] == "verilator"
        assert env["X"] == "1"


class TestBuildRuntimeTmpEnv:
    def test_keys(self):
        env = _build_runtime_tmp_env("/scratch/tmp")
        assert env["TMPDIR"] == "/scratch/tmp"
        assert env["CDS_LOCK"] == "/scratch/tmp/.cdslock"
        assert env["JAVA_TOOL_OPTIONS"] == "-Djava.io.tmpdir=/scratch/tmp"


# ---------------------------------------------------------------------------
# Unit tests: _run_service runs through the Apptainer provider
# ---------------------------------------------------------------------------


class _FakeHandle:
    sandbox_id = "inst-fake"


class _FakeProvider:
    def __init__(self, exec_result, create_error=None):
        self._exec_result = exec_result
        self._create_error = create_error
        self.created = []
        self.execs = []
        self.closed = []

    async def create(self, spec):
        if self._create_error is not None:
            raise self._create_error
        self.created.append(spec)
        return _FakeHandle()

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s})
        return self._exec_result

    async def close(self, handle):
        self.closed.append(handle)


_COMPOSE_WITH_CMD = """
services:
  direct:
    image: ghcr.io/hdl/sim/osvb
    volumes:
      - ./extra:/data:ro
    working_dir: /code/rundir
    command: /bin/sh -c "echo hi"
"""

_COMPOSE_NO_CMD = """
services:
  direct:
    image: ghcr.io/hdl/sim/osvb
    working_dir: /code/rundir
"""


class TestRunServiceProvider:
    def setup_method(self):
        self.server = _make_server()

    async def _run(self, tmp_path, fake, compose):
        with (
            patch.object(self.server._harness, "_ensure_sif", new_callable=AsyncMock, return_value="/cache/img.sif"),
            patch.object(self.server._harness, "_get_provider", new_callable=AsyncMock, return_value=fake),
        ):
            return await self.server._harness._run_service(str(tmp_path), "direct", compose)

    @pytest.mark.asyncio
    async def test_create_exec_close_and_payload(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="hi\n", stderr="", return_code=0))
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)

        assert exit_code == 0
        assert output == "hi\n"
        # create got the cached SIF and the workspace + compose binds.
        spec = fake.created[0]
        assert spec.image == "/cache/img.sif"
        binds = spec.provider_options["binds"]
        assert any(b.endswith(":/code/rtl") for b in binds)
        assert any(b.endswith(":/data:ro") for b in binds)
        # exec wraps the command with HOME export, sets cwd + timeout.
        call = fake.execs[0]
        assert "export HOME=/code/rundir" in call["command"]
        assert "echo hi" in call["command"]
        assert call["cwd"] == "/code/rundir"
        assert call["timeout_s"] == 30
        # instance is always torn down.
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_no_command_uses_runscript(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0))
        await self._run(tmp_path, fake, _COMPOSE_NO_CMD)
        assert "/.singularity.d/runscript" in fake.execs[0]["command"]

    @pytest.mark.asyncio
    async def test_timeout_maps_to_negative_one(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"))
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == -1
        assert "timed out after 30s" in output
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_create_failure_returns_error(self, tmp_path):
        fake = _FakeProvider(
            SandboxExecResult(stdout="", stderr="", return_code=0),
            create_error=SandboxCreateError("boom"),
        )
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == 1
        assert "instance start failed" in output
        assert fake.execs == []
        assert fake.closed == []

    @pytest.mark.asyncio
    async def test_tmp_bind_path_added(self, tmp_path):
        self.server.config.container_tmp_bind_path = "/container/tmp"
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0))
        await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        spec = fake.created[0]
        assert any(b.endswith(":/container/tmp") for b in spec.provider_options["binds"])
        assert fake.execs[0]["env"]["TMPDIR"] == "/container/tmp"
