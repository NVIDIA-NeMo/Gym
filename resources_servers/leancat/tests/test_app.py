# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.lean_proof.status import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPILE_ERROR,
    STATUS_COMPLETED,
    STATUS_EMPTY_GENERATION,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    STATUS_TIMEOUT,
)
from resources_servers.leancat.app import (
    LeanCatResourcesServer,
    LeanCatResourcesServerConfig,
    LeanCatVerifyRequest,
)
from resources_servers.leancat.prepare import (
    PROMPT_CONFIG_PATH,
    REPO_ROOT,
    UPSTREAM_PROMPT_CONFIG_PATH,
)


DATA_DIR = Path(__file__).absolute().parent.parent / "data"

REFERENCE = """import Mathlib

open CategoryTheory

variable {C : Type*} [Category.{v} C]

theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by
  sorry"""

SOLVED = """import Mathlib

open CategoryTheory

variable {C : Type*} [Category.{v} C]

theorem id_comm (α β : (𝟭 C) ⟶ (𝟭 C)) : α ≫ β = β ≫ α := by
  ext X
  exact (α.naturality (β.app X)).symm"""


def _load_rows(filename: str) -> list:
    return [json.loads(line) for line in (DATA_DIR / filename).read_text(encoding="utf-8").splitlines() if line]


PROMPT_CONFIG_FPATH = REPO_ROOT / PROMPT_CONFIG_PATH
UPSTREAM_PROMPT_CONFIG_FPATH = REPO_ROOT / UPSTREAM_PROMPT_CONFIG_PATH

# Data-backed tests read the committed 5-row `data/example.jsonl`; the 100 problems are
# gitignored, so a test reading them would skip on every CI checkout and report green.


@pytest.fixture
def config() -> LeanCatResourcesServerConfig:
    return LeanCatResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="leancat",
        compilation_timeout=300.0,
    )


@pytest.fixture
def server(config) -> LeanCatResourcesServer:
    return LeanCatResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestLeanCatApp:
    def _create_response(self, text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
        message = NeMoGymResponseOutputMessage(
            id=msg_id,
            role="assistant",
            type="message",
            content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
        )
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            output=[message],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def _create_request(
        self,
        text: str,
        formal_statement: str = REFERENCE,
        level: str = "Easy",
    ) -> LeanCatVerifyRequest:
        return LeanCatVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_response(text),
            formal_statement=formal_statement,
            problem_id="0001",
            level=level,
            tag=["Basic"],
            domain=["Category"],
        )

    def _stub_sandbox(self, server: LeanCatResourcesServer, axiom_check: str = "ok", **result) -> AsyncMock:
        """Stub the compile step, so no sandbox is created and no Lean runs.

        Lean would run the appended axiom check and print its verdict under the per-compile
        token; ``axiom_check`` is that verdict: ``ok``, ``bad`` or ``missing`` (never ran).
        """
        base = {"stdout": "", "stderr": "", "return_code": 0} | result

        async def run(code: str, timeout_s: float | None = None) -> SandboxExecResult:
            token = re.search(r"AXIOMS_[0-9a-f]{32}", code)
            verdict = {"ok": "_OK", "bad": "_BAD: #[t uses cheat]", "missing": None}[axiom_check]
            if token is None or verdict is None:
                return SandboxExecResult(**base)
            return SandboxExecResult(**(base | {"stdout": f"{base['stdout']}\n{token.group(0)}{verdict}\n"}))

        mock = AsyncMock(side_effect=run)
        server._run_lean = mock
        return mock

    @pytest.mark.asyncio
    async def test_verify_successful_proof(self, server):
        self._stub_sandbox(server)
        result = await server.verify(self._create_request(f"Here you go.\n```lean4\n{SOLVED}\n```", level="High"))
        assert result.reward == 1.0
        assert result.proof_status == STATUS_COMPLETED
        assert result.statement_preserved
        assert result.failure_reason is None
        assert result.predicted_proof == SOLVED
        # compute_subset_metrics groups on `level`, so the row fields have to survive verify.
        assert result.level == "High"
        assert result.problem_id == "0001"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sandbox_output,expected_status",
        [
            (
                {"stderr": "/lean4/my_project/x.lean:7:2: error: unknown tactic", "return_code": 1},
                STATUS_COMPILE_ERROR,
            ),
            # `lake env lean` exits 0 on a sorry-carrying build; the exit code alone would pass it.
            ({"stdout": "warning: declaration uses 'sorry'"}, STATUS_COMPILE_ERROR),
            # A non-zero exit with no error_type is how an ordinary wrong proof looks, and
            # must not be reported as infrastructure trouble.
            ({"return_code": 1}, STATUS_COMPILE_ERROR),
            ({"return_code": -1, "error_type": "TimeoutError"}, STATUS_TIMEOUT),
            # `timeout` stopping Lean at the documented budget: exit 124, no error_type. Without
            # a mapping this reads as an ordinary non-zero exit, i.e. a rejected proof.
            ({"return_code": 124}, STATUS_TIMEOUT),
            # Lean ignored TERM and was killed after the grace (also what an OOM kill looks like).
            ({"return_code": 137}, STATUS_TIMEOUT),
            ({"return_code": -1, "error_type": "SandboxConnectionError"}, STATUS_SANDBOX_ERROR),
        ],
        ids=[
            "compile-error",
            "zero-exit-with-sorry",
            "non-zero-lake-exit",
            "timeout",
            "lean-killed-at-budget",
            "lean-killed-after-grace",
            "sandbox-down",
        ],
    )
    async def test_verify_maps_sandbox_outcomes_to_statuses(self, server, sandbox_output, expected_status):
        self._stub_sandbox(server, **sandbox_output)
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == expected_status

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text,expected_status",
        [
            ("", STATUS_EMPTY_GENERATION),
            (f"```lean4\n{REFERENCE}\n```", STATUS_BANNED_TOKENS),
            (f"```lean4\n{SOLVED.replace('α ≫ β = β ≫ α', 'True')}\n```", STATUS_STATEMENT_MODIFIED),
        ],
        ids=["empty", "still-has-sorry", "weakened-statement"],
    )
    async def test_verify_rejects_on_text_without_compiling(self, server, text, expected_status):
        """A five-minute Mathlib compile must not be spent on a submission already lost."""
        mock = self._stub_sandbox(server)
        result = await server.verify(self._create_request(text))
        assert result.reward == 0.0
        assert result.proof_status == expected_status
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("axiom_check", ["bad", "missing"])
    async def test_verify_rejects_a_file_that_fails_the_axiom_check(self, server, axiom_check):
        """A clean exit is not enough: the file must also pass the axiom check Lean ran on it."""
        self._stub_sandbox(server, axiom_check=axiom_check)
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_BANNED_TOKENS

    @pytest.mark.asyncio
    async def test_verify_ignores_a_success_line_the_submission_printed_itself(self, server):
        """The token is random per compile, so a printed `_OK` line proves nothing."""
        self._stub_sandbox(server, axiom_check="missing", stdout="AXIOMS_" + "0" * 32 + "_OK")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_BANNED_TOKENS

    @pytest.mark.asyncio
    async def test_verify_skips_the_axiom_check_when_shortcuts_are_not_banned(self, server):
        server.config.ban_proof_shortcuts = False
        mock = self._stub_sandbox(server, axiom_check="missing")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 1.0
        assert mock.await_args.args[0] == SOLVED

    @pytest.mark.asyncio
    async def test_verify_statement_guard_can_be_disabled(self, server):
        server.config.require_statement_preserved = False
        self._stub_sandbox(server)
        cheat = SOLVED.replace("α ≫ β = β ≫ α", "True")
        result = await server.verify(self._create_request(f"```lean4\n{cheat}\n```"))
        assert result.reward == 1.0
        # The guard still reports what it saw, so the run stays diagnosable.
        assert result.statement_preserved is False

    @pytest.mark.asyncio
    async def test_verify_compiles_the_whole_submitted_file(self, server):
        # The model owns the whole file: nothing is prepended.
        captured = {}

        async def capture_code(code: str, timeout_s: float | None = None):
            captured["code"] = code
            return SandboxExecResult(stdout="", stderr="", return_code=0)

        server._run_lean = capture_code
        await server.verify(self._create_request(f"Reasoning...\n```lean4\n{SOLVED}\n```"))
        assert captured["code"].startswith(SOLVED)


class TestMetrics:
    def _task(self, rewards, level):
        return [{"reward": r, "level": level, "statement_preserved": True} for r in rewards]

    def test_pass_at_k_is_broken_out_by_difficulty(self, server):
        tasks = [
            self._task([1.0, 0.0], "Easy"),
            self._task([0.0, 0.0], "Easy"),
            self._task([0.0, 0.0], "High"),
        ]
        metrics = server.compute_metrics(tasks)

        assert metrics["pass@1/accuracy"] == pytest.approx(100.0 * (0.5 + 0.0 + 0.0) / 3)
        assert metrics["pass@2/accuracy"] == pytest.approx(100.0 / 3)
        assert metrics["Easy/pass@2/accuracy"] == pytest.approx(50.0)
        assert metrics["High/pass@2/accuracy"] == pytest.approx(0.0)

    def test_statement_preserved_is_tracked_as_a_second_score(self, server):
        tasks = [[{"reward": 0.0, "level": "Easy", "statement_preserved": False}]]
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1/statement_preserved"] == pytest.approx(0.0)

    def test_key_metrics_pick_the_highest_k(self, server):
        agent_metrics = {
            "mean/output_tokens": 12345.0,
            "pass@1/accuracy": 8.25,
            "pass@4/accuracy": 12.0,
            "pass@1[avg-of-4]/accuracy": 8.25,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["pass@4/accuracy"] == 12.0
        assert key["pass@1[avg-of-4]/accuracy"] == 8.25
        assert key["mean/output_tokens"] == 12345.0
        assert "pass@1/accuracy" not in key


class TestDataset:
    def test_rows_load_as_verify_requests(self):
        rows = _load_rows("example.jsonl")
        assert len(rows) == 5

        for row in rows:
            # Rows are flat -- no responses_create_params, no verifier_metadata -- so they
            # splat straight into the request the way the agent hands them to verify().
            request = LeanCatVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                response=TestLeanCatApp()._create_response(""),
                **row,
            )
            assert request.formal_statement
            assert request.level in {"Easy", "Medium", "High"}
            # Every task is a hole to fill; a row with no `sorry` would be unsolvable by
            # construction, since filling nothing still leaves the file unchanged.
            assert "sorry" in request.formal_statement


class TestPrompt:
    """Both templates ship: `paper.yaml` (the default, Appendix D.1) and `upstream-repo.yaml`."""

    @pytest.mark.parametrize("fpath", [PROMPT_CONFIG_FPATH, UPSTREAM_PROMPT_CONFIG_FPATH])
    def test_is_a_valid_gym_prompt_config(self, fpath):
        config = load_prompt_config(str(fpath))
        assert "{formal_statement}" in config.user
        assert config.system is None, "upstream posts a single user message; a system prompt would deviate"

    def test_filling_it_reproduces_the_prompt_the_benchmark_runs(self):
        """A shipped row plus the shipped template must equal one user message."""
        config = load_prompt_config(str(PROMPT_CONFIG_FPATH))
        for row in _load_rows("example.jsonl"):
            messages = apply_prompt_to_row(row, config)["responses_create_params"]["input"]
            assert [m["role"] for m in messages] == ["user"]
            assert messages[0]["content"] == config.user.format(formal_statement=row["formal_statement"])
            assert row["formal_statement"] in messages[0]["content"]


class TestToolchainProbe:
    """The probe itself is tested in lean_proof; this is only the wiring."""

    @pytest.mark.asyncio
    async def test_mismatch_against_the_rows_pin_is_logged(self, server, caplog):
        server._run_lean = AsyncMock(return_value=SandboxExecResult(stdout='"4.12.0"', stderr="", return_code=0))
        with caplog.at_level("ERROR"):
            await server._check_toolchain_once("leanprover/lean4:v4.19.0")
        assert "MATHLIB MISMATCH" in caplog.text
        assert "4.12.0" in caplog.text and "4.19.0" in caplog.text


class TestVerifierMetadataLifting:
    """Gym posts rows with `verifier_metadata` nested; a top-level field wins on conflict."""

    @pytest.mark.parametrize(
        "fields,expected_statement,expected_level",
        [
            (
                {"verifier_metadata": {"formal_statement": REFERENCE, "level": "Easy"}},
                REFERENCE,
                "Easy",
            ),
            ({"formal_statement": REFERENCE, "level": "High"}, REFERENCE, "High"),
            (
                {
                    "formal_statement": REFERENCE,
                    "level": "High",
                    "verifier_metadata": {"formal_statement": "other", "level": "Easy"},
                },
                REFERENCE,
                "High",
            ),
        ],
        ids=["nested-is-lifted", "top-level", "top-level-wins"],
    )
    def test_row_fields_reach_the_request(self, fields, expected_statement, expected_level):
        request = LeanCatVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=TestLeanCatApp()._create_response(""),
            **fields,
        )
        assert request.formal_statement == expected_statement
        assert request.level == expected_level
