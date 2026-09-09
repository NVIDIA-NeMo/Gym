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
from nemo_gym.server_utils import ServerClient
from resources_servers.leancat.app import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPILE_ERROR,
    STATUS_COMPILED,
    STATUS_NO_CODE,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    STATUS_TIMEOUT,
    LeanCatResourcesServer,
    LeanCatResourcesServerConfig,
    LeanCatVerifyRequest,
)
from resources_servers.leancat.lean_verifier import check_statement_preserved


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


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="test_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                role="assistant",
                type="message",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _request(text: str, formal_statement: str = REFERENCE, level: str = "Easy") -> LeanCatVerifyRequest:
    return LeanCatVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response(text),
        formal_statement=formal_statement,
        problem_id="0001",
        level=level,
        tag=["Basic"],
        domain=["Category"],
    )


@pytest.fixture
def server() -> LeanCatResourcesServer:
    config = LeanCatResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="leancat",
        sandbox_host="127.0.0.1",
        sandbox_port=6000,
        compilation_timeout=300.0,
    )
    return LeanCatResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _stub_sandbox(server: LeanCatResourcesServer, **output) -> AsyncMock:
    result = {"process_status": "completed", "stdout": "", "stderr": ""} | output
    mock = AsyncMock(return_value=result)
    server._sandbox_client.execute_lean4 = mock
    return mock


class TestVerify:
    @pytest.mark.asyncio
    async def test_clean_compile_scores_one(self, server):
        _stub_sandbox(server)
        result = await server.verify(_request(f"Here you go.\n```lean4\n{SOLVED}\n```"))
        assert result.reward == 1.0
        assert result.status == STATUS_COMPILED
        assert result.statement_preserved
        assert result.failure_reason is None
        assert result.submitted_code == SOLVED

    @pytest.mark.asyncio
    async def test_row_metadata_survives_onto_the_response(self, server):
        # compute_subset_metrics groups on `level`, so it has to make it through verify.
        _stub_sandbox(server)
        result = await server.verify(_request(f"```lean4\n{SOLVED}\n```", level="High"))
        assert result.level == "High"
        assert result.problem_id == "0001"

    @pytest.mark.asyncio
    async def test_compiler_errors_score_zero(self, server):
        _stub_sandbox(server, stderr="/lean4/my_project/x.lean:7:2: error: unknown tactic")
        result = await server.verify(_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_COMPILE_ERROR
        assert result.compiler_output.stderr.endswith("unknown tactic")

    @pytest.mark.asyncio
    async def test_zero_exit_with_sorry_warning_scores_zero(self, server):
        # `lake env lean` exits 0 on a sorry-carrying build; the status alone would pass it.
        _stub_sandbox(server, stdout="warning: declaration uses 'sorry'")
        result = await server.verify(_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_COMPILE_ERROR

    @pytest.mark.asyncio
    async def test_timeout_is_reported_distinctly(self, server):
        _stub_sandbox(server, process_status="timeout")
        result = await server.verify(_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_TIMEOUT

    @pytest.mark.asyncio
    async def test_sandbox_failure_is_reported_distinctly(self, server):
        _stub_sandbox(server, process_status="error", stderr="connection refused")
        result = await server.verify(_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_SANDBOX_ERROR

    @pytest.mark.asyncio
    async def test_empty_response_short_circuits(self, server):
        mock = _stub_sandbox(server)
        result = await server.verify(_request(""))
        assert result.reward == 0.0
        assert result.status == STATUS_NO_CODE
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sorry_is_rejected_without_compiling(self, server):
        mock = _stub_sandbox(server)
        result = await server.verify(_request(f"```lean4\n{REFERENCE}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_BANNED_TOKENS
        assert "sorry" in result.failure_reason
        # A five-minute Mathlib compile must not be spent on a submission already lost.
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_weakened_statement_is_rejected_without_compiling(self, server):
        mock = _stub_sandbox(server)
        cheat = SOLVED.replace("α ≫ β = β ≫ α", "True")
        result = await server.verify(_request(f"```lean4\n{cheat}\n```"))
        assert result.reward == 0.0
        assert result.status == STATUS_STATEMENT_MODIFIED
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_statement_guard_can_be_disabled(self, server):
        server.config.require_statement_preserved = False
        _stub_sandbox(server)
        cheat = SOLVED.replace("α ≫ β = β ≫ α", "True")
        result = await server.verify(_request(f"```lean4\n{cheat}\n```"))
        assert result.reward == 1.0
        # The guard still reports what it saw, so the run stays diagnosable.
        assert result.statement_preserved is False


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

    def test_no_tasks(self, server):
        assert server.compute_metrics([]) == {}

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


def _load_rows(filename: str) -> list:
    return [json.loads(line) for line in (DATA_DIR / filename).read_text(encoding="utf-8").splitlines() if line]


# train.jsonl is gitignored and regenerated by prepare_leancat.py, so the full-dataset
# tests skip rather than fail on a fresh checkout. example.jsonl is committed and always runs.
needs_full_dataset = pytest.mark.skipif(
    not (DATA_DIR / "train.jsonl").exists(),
    reason="run prepare_leancat.py to generate data/train.jsonl",
)


class TestDataset:
    @pytest.mark.parametrize(
        "filename,expected_rows",
        [pytest.param("train.jsonl", 100, marks=needs_full_dataset), ("example.jsonl", 5)],
    )
    def test_rows_load_as_verify_requests(self, filename, expected_rows):
        rows = _load_rows(filename)
        assert len(rows) == expected_rows

        for row in rows:
            request = LeanCatVerifyRequest(
                responses_create_params=row["responses_create_params"],
                response=_response(""),
                **row["verifier_metadata"],
            )
            assert request.formal_statement
            assert request.level in {"Easy", "Medium", "High"}
            # Every task is a hole to fill; a row with no `sorry` would be unsolvable by
            # construction, since filling nothing still leaves the file unchanged.
            assert "sorry" in request.formal_statement

    @needs_full_dataset
    def test_guard_accepts_every_reference_statement_with_its_holes_filled(self):
        """The statement guard must not reject an honest answer on any of the 100 tasks.

        A false positive here is invisible in a real run -- it just looks like the model
        failed -- so it is checked against every shipped row rather than a sample. The
        substitution stands in for the minimal honest submission: the reference file with
        each hole replaced by a tactic and nothing else touched.
        """
        rows = _load_rows("train.jsonl")
        rejected = []
        for row in rows:
            statement = row["verifier_metadata"]["formal_statement"]
            filled = re.sub(r"\bsorry\b", "aesop_cat", statement)
            preserved, reason = check_statement_preserved(statement, filled)
            if not preserved:
                rejected.append((row["verifier_metadata"]["problem_id"], reason))
        assert rejected == []

    def test_prompt_contains_the_statement(self):
        for row in _load_rows("example.jsonl"):
            prompt = row["responses_create_params"]["input"][0]["content"]
            assert row["verifier_metadata"]["formal_statement"] in prompt
