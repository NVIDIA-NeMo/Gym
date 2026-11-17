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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.responses import JSONResponse

from nemo_gym.client import ServerClient
from nemo_gym.eval.rollout_reverification import _is_judge_failure
from nemo_gym.rollouts.rollout_recovery import has_saved_generation
from nemo_gym.sandbox import SandboxCreateError, SandboxExecResult
from nemo_gym.verification import JudgeError, SandboxVerificationError, judge_failsafe
from resources_servers.cvdp.app import (
    CVDPResourcesServer,
    CVDPResourcesServerConfig,
    _parse_model_response,
)
from resources_servers.cvdp.cvdp_lib.subjective import calculate_BLEU, calculate_ROUGE
from resources_servers.cvdp.testbench_runner import (
    TestbenchRunner as CVDPTestbenchRunner,
)
from resources_servers.cvdp.testbench_runner import (
    _apply_substitutions,
    _build_command,
    _build_env,
    _build_runtime_tmp_env,
    _compose_workspace_links,
    _load_dot_env,
    _pack_workspace,
    _parse_compose_service,
    _resolve_service_image,
    _service_build_key,
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
    async def test_verify_empty_output_grades_harvested_rtl_files(self):
        body_dict = _make_body(output_text="")
        body_dict["rtl_files"] = {"rtl/foo.sv": SAMPLE_RTL}
        with patch.object(
            self.server._harness,
            "run",
            new_callable=AsyncMock,
            return_value=(0, "", []),
        ) as run:
            result = await self.server.verify(_make_request(body_dict))
        run.assert_awaited_once()
        assert result.reward == 1.0
        assert result.extracted_rtl == {"rtl/foo.sv": SAMPLE_RTL}

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
    async def test_verify_missing_compose_is_masked_not_zero(self):
        harness_no_compose = {"src/.env": "SIM=icarus\n"}
        body_dict = _make_body(
            output_text=f"```systemverilog\n{SAMPLE_RTL}\n```",
            harness_files=harness_no_compose,
        )
        with pytest.raises(JudgeError, match="No docker-compose.yml"):
            await self.server.verify(_make_request(body_dict))

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


class TestHarnessConfigDefects:
    """A defect in the task's harness files fails before any sandbox starts, so it needs no Apptainer."""

    def setup_method(self):
        self.server = _make_server()

    @pytest.mark.asyncio
    async def test_missing_compose_is_a_verifier_error(self):
        with pytest.raises(SandboxVerificationError) as caught:
            await self.server._harness.run(
                rtl_files={"rtl/foo.sv": SAMPLE_RTL},
                harness_files={},  # no compose file
                task_id="test",
            )
        assert caught.value.failure_kind == "verifier_error"


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
    async def test_repository_changes_are_graded_and_retained_without_chat_text(self):
        body_dict = _make_body(output_text="")
        body_dict["verifier_metadata"]["repository_workspace_sha256"] = "a" * 64
        changes = {"archive": "saved-delta", "changed_count": 1, "deleted": ["old.v"]}
        body_dict["repository_changes"] = changes
        self.server._harness.run = AsyncMock(return_value=(0, "passed", []))
        result = await self.server.verify(_make_request(body_dict))
        assert result.reward == 1.0
        assert not result.parse_failed
        assert result.model_dump()["repository_changes"] == changes
        self.server._harness.run.assert_awaited_once_with(
            rtl_files={},
            harness_files=body_dict["verifier_metadata"]["harness_files"],
            task_id=body_dict["verifier_metadata"]["task_id"],
            context_files={},
            repository_workspace_sha256="a" * 64,
            repository_changes=changes,
        )

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
# Unit tests: provider-neutral Compose/env helpers
# ---------------------------------------------------------------------------


class TestComposeWorkspaceLinks:
    def test_translates_relative_sources(self):
        assert _compose_workspace_links(["./src:/src:ro", "./extra:/data"]) == [
            ("/code/src", "/src"),
            ("/code/extra", "/data"),
        ]

    def test_skips_already_materialized_code_path(self):
        assert _compose_workspace_links(["./rtl:/code/rtl:ro"]) == []

    def test_rejects_host_absolute_source(self):
        with pytest.raises(ValueError, match="host-absolute"):
            _compose_workspace_links(["/tmp/src:/src"])


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


class TestPreparedImages:
    def test_explicit_image_override_uses_local_sif(self):
        server = _make_server()
        server.config.image_overrides = {"hdlc/sim:osvb": "/images/cvdp-arm.sif"}
        compose = {"services": {"direct": {"image": "docker://hdlc/sim:osvb"}}}
        image, key = _resolve_service_image(compose, "direct", {}, server.config, {})
        assert image == "/images/cvdp-arm.sif"
        assert key is None

    def test_direct_image_needs_no_manifest(self):
        server = _make_server()
        compose = yaml.safe_load(_COMPOSE_WITH_CMD)
        image, key = _resolve_service_image(compose, "direct", {}, server.config, {})
        assert image == "ghcr.io/hdl/sim/osvb"
        assert key is None

    def test_build_service_resolves_by_recipe_key(self):
        server = _make_server()
        compose = {"services": {"synth": {"build": {"dockerfile": "src/Dockerfile.synth"}}}}
        harness = {"src/Dockerfile.synth": "FROM __OSS_PNR_IMAGE__\nRUN true\n"}
        key = _service_build_key(compose, "synth", harness, server.config)
        image, resolved_key = _resolve_service_image(
            compose,
            "synth",
            harness,
            server.config,
            {key: "registry.example/cvdp:synth"},
        )
        assert image == "registry.example/cvdp:synth"
        assert resolved_key == key

    def test_named_provider_config_and_metadata_are_resolved(self):
        server = _make_server()
        server.config.sandbox_provider = "sandbox"
        runner = CVDPTestbenchRunner(
            server.config,
            {
                "sandbox": {
                    "opensandbox": {"connection": {"domain": "sandbox.example"}},
                    "default_metadata": {"sandbox-api": "opensandbox-sdk"},
                }
            },
        )
        assert "opensandbox" in runner._sandbox_provider
        assert runner._sandbox_metadata == {"sandbox-api": "opensandbox-sdk"}


# ---------------------------------------------------------------------------
# Unit tests: _run_service runs through AsyncSandbox
# ---------------------------------------------------------------------------


class _FakeHandle:
    sandbox_id = "inst-fake"


class _FakeProvider:
    def __init__(
        self, exec_result, create_error=None, name="fake", snapshot_result=None, setup_result=None, upload_error=None
    ):
        self.name = name
        self._exec_result = exec_result
        self._create_error = create_error
        self._snapshot_result = snapshot_result
        self._setup_result = setup_result
        self._upload_error = upload_error
        self.created = []
        self.execs = []
        self.closed = []
        self.uploads = []
        self.downloads = []
        self._uploaded_archive = None

    async def create(self, spec):
        if self._create_error is not None:
            raise self._create_error
        self.created.append(spec)
        return _FakeHandle()

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.execs.append({"command": command, "cwd": cwd, "env": env, "timeout_s": timeout_s})
        if "cvdp-workspace.tar.gz" in command and "tar -xzf" in command:
            return self._setup_result or SandboxExecResult(stdout="", stderr="", return_code=0)
        if "cvdp-workspace-out.tar.gz" in command and "tar -czf" in command:
            return self._snapshot_result or SandboxExecResult(stdout="", stderr="", return_code=0)
        return self._exec_result

    async def upload_file(self, handle, source_path, target_path):
        if self._upload_error is not None:
            raise self._upload_error
        self.uploads.append((Path(source_path), target_path))
        self._uploaded_archive = Path(source_path)

    async def download_file(self, handle, source_path, target_path):
        self.downloads.append((source_path, Path(target_path)))
        shutil.copyfile(self._uploaded_archive, target_path)

    async def close(self, handle):
        self.closed.append(handle)

    async def aclose(self):
        pass


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

    @pytest.mark.asyncio
    async def test_transfer_archives_inside_workspace_are_excluded_from_snapshot(self, tmp_path):
        self.server.config.container_transfer_dir = "/code/.cvdp-transfer"
        fake = _FakeProvider(SandboxExecResult(stdout="hi\n", stderr="", return_code=0))
        exit_code, _ = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == 0
        snapshot = next(call["command"] for call in fake.execs if "tar -czf" in call["command"])
        assert "--exclude=./.cvdp-transfer" in snapshot
        assert "--exclude=./.apptainer-workdir" in snapshot
        assert fake.downloads[0][0] == "/code/.cvdp-transfer/cvdp-workspace-out.tar.gz"

    async def _run(self, tmp_path, fake, compose, harness_files=None):
        (tmp_path / "rtl").mkdir(exist_ok=True)
        (tmp_path / "src").mkdir(exist_ok=True)
        archive = tmp_path.parent / f"{tmp_path.name}.tar.gz"
        _pack_workspace(tmp_path, archive)
        try:
            self.server._harness._sandbox_provider = fake
            return await self.server._harness._run_service(
                archive,
                "direct",
                compose,
                harness_files if harness_files is not None else {"src/.env": "SIM=icarus\n"},
            )
        finally:
            archive.unlink(missing_ok=True)
            archive.with_name(f"{archive.name}.next").unlink(missing_ok=True)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("env_file", ["./src/nested/.env", ["src/base.env", "./src/nested/.env"]])
    async def test_declared_nested_env_files_reach_verifier(self, tmp_path, env_file):
        fake = _FakeProvider(SandboxExecResult(stdout="hi\n", stderr="", return_code=0))
        compose = yaml.safe_load(_COMPOSE_WITH_CMD)
        compose["services"]["direct"].update(env_file=env_file, environment={"SIM": "icarus"})
        await self._run(
            tmp_path,
            fake,
            yaml.safe_dump(compose),
            {
                "src/base.env": "VERILOG_SOURCES=wrong.sv\nBASE_ONLY=kept\n",
                "src/nested/.env": "VERILOG_SOURCES=/code/rtl/top.sv\nSIM=verilator\n",
                "src/.env": "VERILOG_SOURCES=should_not_be_loaded.sv\n",
            },
        )
        env = next(item["env"] for item in fake.execs if "echo hi" in item["command"])
        assert env["VERILOG_SOURCES"] == "/code/rtl/top.sv"
        assert env["SIM"] == "icarus"
        if isinstance(env_file, list):
            assert env["BASE_ONLY"] == "kept"

    @pytest.mark.asyncio
    async def test_missing_declared_env_file_is_inconclusive(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0))
        compose = yaml.safe_load(_COMPOSE_WITH_CMD)
        compose["services"]["direct"]["env_file"] = "src/missing.env"
        with pytest.raises(SandboxVerificationError, match="Missing env_file"):
            await self._run(tmp_path, fake, yaml.safe_dump(compose), {})
        assert fake.created == []

    @pytest.mark.asyncio
    async def test_create_exec_close_and_payload(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="hi\n", stderr="", return_code=0))
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)

        assert exit_code == 0
        assert output == "hi\n"
        spec = fake.created[0]
        assert spec.image == "ghcr.io/hdl/sim/osvb"
        assert spec.provider_options == {}
        assert fake.uploads[0][1] == "/sandbox/cvdp-workspace.tar.gz"
        setup = fake.execs[0]
        assert "ln -s /code/extra /data" in setup["command"]
        assert setup["cwd"] == "/"
        call = next(item for item in fake.execs if "echo hi" in item["command"])
        assert "echo hi" in call["command"]
        assert call["command"].startswith("env HOME=/code/rundir PYTHONNOUSERSITE=1 ")
        assert call["cwd"] == "/code/rundir"
        assert call["env"]["SIM"] == "icarus"
        assert "HOME" not in call["env"]
        assert call["timeout_s"] == 30
        assert fake.downloads[0][0] == "/sandbox/cvdp-workspace-out.tar.gz"
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_no_command_is_verifier_error_not_a_zero(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0))
        with pytest.raises(SandboxVerificationError, match="no explicit command") as caught:
            await self._run(tmp_path, fake, _COMPOSE_NO_CMD)
        assert caught.value.failure_kind == "verifier_error"
        assert fake.created == []

    @pytest.mark.asyncio
    async def test_timeout_maps_to_negative_one(self, tmp_path):
        fake = _FakeProvider(
            SandboxExecResult(stdout="waiting for valid", stderr="timed out\n", return_code=125, error_type="timeout"),
            snapshot_result=SandboxExecResult(stdout="", stderr="file changed as we read it", return_code=1),
        )
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == -1
        assert "timed out after 30s" in output
        assert "waiting for valid" in output
        assert not any("tar -czf" in call["command"] for call in fake.execs)
        assert fake.downloads == []
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("return_code", [-9, -15])
    async def test_terminated_process_result_is_preserved_and_closed(self, tmp_path, return_code):
        fake = _FakeProvider(SandboxExecResult(stdout=None, stderr=None, return_code=return_code))
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == return_code
        assert output == ""
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_create_failure_is_provider_unavailable(self, tmp_path):
        fake = _FakeProvider(
            SandboxExecResult(stdout="", stderr="", return_code=0),
            create_error=SandboxCreateError("boom"),
        )
        with pytest.raises(SandboxVerificationError, match="sandbox create failed.*boom") as caught:
            await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert caught.value.failure_kind == "provider_unavailable"
        assert fake.execs == []
        assert fake.closed == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fake_kwargs,reason",
        [
            ({"snapshot_result": SandboxExecResult(stdout="", stderr="disk full", return_code=1)}, "snapshot failed"),
            ({"setup_result": SandboxExecResult(stdout="", stderr="tar: corrupt", return_code=2)}, "setup failed"),
            ({"upload_error": OSError("transfer dir not mounted")}, "operation failed"),
        ],
        ids=["snapshot", "workspace_setup", "upload"],
    )
    async def test_workspace_transfer_failure_is_verifier_error(self, tmp_path, fake_kwargs, reason):
        fake = _FakeProvider(SandboxExecResult(stdout="hi\n", stderr="", return_code=0), **fake_kwargs)
        with pytest.raises(SandboxVerificationError, match=reason) as caught:
            await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert caught.value.failure_kind == "verifier_error"
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_exec_without_exit_code_is_verifier_error(self, tmp_path):
        fake = _FakeProvider(
            SandboxExecResult(stdout=None, stderr="instance vanished", return_code=None, error_type="provider_error")
        )
        with pytest.raises(SandboxVerificationError, match="no exit code.*provider_error") as caught:
            await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert caught.value.failure_kind == "verifier_error"
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_failing_testbench_keeps_its_exit_code(self, tmp_path):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="FAILED: assertion error", return_code=1))
        exit_code, output = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert (exit_code, output) == (1, "FAILED: assertion error")
        assert len(fake.closed) == 1

    @pytest.mark.asyncio
    async def test_tmp_bind_path_added(self, tmp_path):
        self.server.config.container_tmp_bind_path = "/container/tmp"
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0))
        await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert "ln -s /code/rundir/tmp /container/tmp" in fake.execs[0]["command"]
        command = next(item for item in fake.execs if "echo hi" in item["command"])
        assert command["env"]["TMPDIR"] == "/container/tmp"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name", ["apptainer", "opensandbox"])
    async def test_provider_identity_does_not_change_spec(self, tmp_path, provider_name):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0), name=provider_name)
        exit_code, _ = await self._run(tmp_path, fake, _COMPOSE_WITH_CMD)
        assert exit_code == 0
        assert fake.created[0].provider_options == {}


# ---------------------------------------------------------------------------
# /verify through judge_failsafe: infrastructure failures vs testbench verdicts
# ---------------------------------------------------------------------------

_RTL_ANSWER = f"```systemverilog\n{SAMPLE_RTL}\n```"


class TestVerifyInfrastructureFailure:
    """The harness runs for real through AsyncSandbox; only the provider is a test double."""

    async def _verify(self, fake, body_dict) -> dict:
        server = _make_server()
        server._harness._sandbox_provider = fake
        # The wrapper SimpleResourcesServer registers on /verify.
        result = await judge_failsafe(server.verify)(body=_make_request(body_dict))
        return json.loads(result.body) if isinstance(result, JSONResponse) else result.model_dump(mode="json")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fake_kwargs,kind",
        [
            ({"create_error": SandboxCreateError("no free instance")}, "provider_unavailable"),
            ({"setup_result": SandboxExecResult(stdout="", stderr="tar: corrupt", return_code=2)}, "verifier_error"),
            ({"upload_error": OSError("transfer dir not mounted")}, "verifier_error"),
        ],
        ids=["create", "workspace_setup", "upload"],
    )
    async def test_model_answer_is_kept_for_reverify(self, fake_kwargs, kind):
        fake = _FakeProvider(SandboxExecResult(stdout="", stderr="", return_code=0), **fake_kwargs)
        row = await self._verify(fake, _make_body(output_text=_RTL_ANSWER))
        assert row["_ng_failure_class"] == row["failure_kind"] == "judge_failed"
        assert f"CVDP harness infrastructure failure ({kind})" in row["failure_reason"]
        assert row["mask_sample"] is True and row.get("container_exit_code") is None
        assert row["response"]["output"][0]["content"][0]["text"] == _RTL_ANSWER
        # gym eval reverify --judge-failed-only selects this row and simulates its saved RTL again; no regeneration.
        assert _is_judge_failure(row) and has_saved_generation(row)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exec_result,reward",
        [
            (SandboxExecResult(stdout="1 passed", stderr="", return_code=0), 1.0),
            (SandboxExecResult(stdout="", stderr="FAILED: assertion error", return_code=1), 0.0),
            (SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout"), 0.0),
        ],
        ids=["passing_rtl", "failing_rtl", "hanging_rtl"],
    )
    async def test_testbench_verdict_is_scored(self, exec_result, reward):
        row = await self._verify(_FakeProvider(exec_result), _make_body(output_text=_RTL_ANSWER))
        assert row["reward"] == reward
        assert row["mask_sample"] is False and "_ng_failure_class" not in row
        assert row["extracted_rtl"]["rtl/foo.sv"].strip() == SAMPLE_RTL
        assert not _is_judge_failure(row)

    @pytest.mark.asyncio
    async def test_agent_written_rtl_is_masked_not_scored(self):
        """Reverification cannot replay rtl_files, so the sample is masked with the files kept."""
        body = _make_body(output_text="done") | {"rtl_files": {"rtl/foo.sv": SAMPLE_RTL}}
        fake = _FakeProvider(
            SandboxExecResult(stdout="", stderr="", return_code=0), create_error=SandboxCreateError("no free instance")
        )
        row = await self._verify(fake, body)
        assert row["mask_sample"] is True and row["reward"] == 0.0
        assert row["failure_kind"] == "provider_unavailable" and "no free instance" in row["failure_reason"]
        assert row["extracted_rtl"] == {"rtl/foo.sv": SAMPLE_RTL}
        assert "_ng_failure_class" not in row
