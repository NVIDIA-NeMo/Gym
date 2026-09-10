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
from nemo_gym.server_utils import ServerClient
from resources_servers.leancat.app import (
    STATUS_BANNED_TOKENS,
    STATUS_COMPILE_ERROR,
    STATUS_COMPLETED,
    STATUS_EMPTY_GENERATION,
    STATUS_SANDBOX_ERROR,
    STATUS_STATEMENT_MODIFIED,
    STATUS_TIMEOUT,
    LeanCatResourcesServer,
    LeanCatResourcesServerConfig,
    LeanCatVerifyRequest,
)
from resources_servers.leancat.prepare import (
    PROMPT_CONFIG_PATH,
    REPO_ROOT,
    UPSTREAM_PROMPT_CONFIG_PATH,
    UPSTREAM_PROMPT_URL,
)
from resources_servers.leancat.proof_utils import check_statement_preserved


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

# Every data-backed test reads the committed 5-row `data/example.jsonl`, as the rest of the
# repo's server tests do. LeanCat's 100 problems are gitignored and regenerated from a pinned
# upstream commit, so a test that read them would skip on every CI checkout and report green
# without having run.


class TestLeanCatApp:
    @pytest.fixture
    def config(self) -> LeanCatResourcesServerConfig:
        return LeanCatResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="leancat",
            sandbox_host="127.0.0.1",
            sandbox_port=6000,
            compilation_timeout=300.0,
        )

    @pytest.fixture
    def server(self, config) -> LeanCatResourcesServer:
        return LeanCatResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_response(self, text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=msg_id,
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=text,
                            annotations=[],
                        )
                    ],
                )
            ],
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

    def _stub_sandbox(self, server: LeanCatResourcesServer, **output) -> AsyncMock:
        result = {"process_status": "completed", "stdout": "", "stderr": ""} | output
        mock = AsyncMock(return_value=result)
        server._sandbox_client.execute_lean4 = mock
        return mock

    @pytest.mark.asyncio
    async def test_verify_successful_proof(self, server):
        self._stub_sandbox(server)
        result = await server.verify(self._create_request(f"Here you go.\n```lean4\n{SOLVED}\n```"))
        assert result.reward == 1.0
        assert result.proof_status == STATUS_COMPLETED
        assert result.statement_preserved
        assert result.failure_reason is None
        assert result.predicted_proof == SOLVED

    @pytest.mark.asyncio
    async def test_verify_row_metadata_survives_onto_the_response(self, server):
        # compute_subset_metrics groups on `level`, so it has to make it through verify.
        self._stub_sandbox(server)
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```", level="High"))
        assert result.level == "High"
        assert result.problem_id == "0001"

    @pytest.mark.asyncio
    async def test_verify_failed_proof(self, server):
        self._stub_sandbox(server, stderr="/lean4/my_project/x.lean:7:2: error: unknown tactic")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_COMPILE_ERROR
        assert result.compiler_output.stderr.endswith("unknown tactic")

    @pytest.mark.asyncio
    async def test_verify_zero_exit_with_sorry_warning(self, server):
        # `lake env lean` exits 0 on a sorry-carrying build; the status alone would pass it.
        self._stub_sandbox(server, stdout="warning: declaration uses 'sorry'")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_COMPILE_ERROR

    @pytest.mark.asyncio
    async def test_verify_timeout(self, server):
        self._stub_sandbox(server, process_status="timeout")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_TIMEOUT

    @pytest.mark.asyncio
    async def test_verify_lean_rejected_is_a_compile_error_not_a_sandbox_error(self, server):
        # The NeMo-Skills sandbox says "failed" for any non-zero lake exit; that is the
        # normal way a wrong proof looks, and must not be reported as infrastructure trouble.
        self._stub_sandbox(server, process_status="failed")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_COMPILE_ERROR

    @pytest.mark.asyncio
    async def test_verify_sandbox_failure(self, server):
        self._stub_sandbox(server, process_status="error", stderr="connection refused")
        result = await server.verify(self._create_request(f"```lean4\n{SOLVED}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_SANDBOX_ERROR

    @pytest.mark.asyncio
    async def test_verify_empty_generation(self, server):
        mock = self._stub_sandbox(server)
        result = await server.verify(self._create_request(""))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_EMPTY_GENERATION
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_verify_rejects_sorry_without_compiling(self, server):
        mock = self._stub_sandbox(server)
        result = await server.verify(self._create_request(f"```lean4\n{REFERENCE}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_BANNED_TOKENS
        assert "sorry" in result.failure_reason
        # A five-minute Mathlib compile must not be spent on a submission already lost.
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_verify_rejects_weakened_statement_without_compiling(self, server):
        mock = self._stub_sandbox(server)
        cheat = SOLVED.replace("α ≫ β = β ≫ α", "True")
        result = await server.verify(self._create_request(f"```lean4\n{cheat}\n```"))
        assert result.reward == 0.0
        assert result.proof_status == STATUS_STATEMENT_MODIFIED
        mock.assert_not_awaited()

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
        # The model owns the file here, unlike math_formal_lean: nothing is prepended.
        captured = {}

        async def capture_code(code: str, timeout: float):
            captured["code"] = code
            return {"process_status": "completed", "stdout": "", "stderr": ""}

        server._sandbox_client.execute_lean4 = capture_code
        await server.verify(self._create_request(f"Reasoning...\n```lean4\n{SOLVED}\n```"))
        assert captured["code"] == SOLVED


class TestMetrics:
    @pytest.fixture
    def server(self) -> LeanCatResourcesServer:
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

    def test_guard_accepts_a_reference_statement_with_its_holes_filled(self):
        """The statement guard must not reject an honest answer.

        A false positive here is invisible in a real run -- it just looks like the model
        failed -- so it is worth pinning. The substitution stands in for the minimal honest
        submission: the reference file with each hole replaced by a tactic and nothing else
        touched. Checked over the committed example rows, which is what CI can see; the
        upstream statements themselves are pinned by commit, so the 5 are stable inputs and
        not a random sample.
        """
        rejected = []
        for row in _load_rows("example.jsonl"):
            statement = row["formal_statement"]
            filled = re.sub(r"\bsorry\b", "aesop_cat", statement)
            preserved, reason = check_statement_preserved(statement, filled)
            if not preserved:
                rejected.append((row["problem_id"], reason))
        assert rejected == []

    def test_rows_carry_no_prebuilt_input(self):
        """Rows must stay prompt-free, or `prompt_config` refuses to run.

        `nemo_gym.prompt.validate_prompt_compatibility` rejects a row that carries
        `responses_create_params.input` alongside a `prompt_config`, and the benchmark config
        sets one -- so a prepare script that started baking prompts in would break the run,
        not merely duplicate work.
        """
        for row in _load_rows("example.jsonl"):
            assert not row.get("responses_create_params", {}).get("input")


class TestPrompt:
    """Two templates ship, both applied at run time via ``prompt_config``.

    ``paper.yaml`` is the benchmark default: the template printed in the paper's Appendix D.1,
    which is what the published numbers correspond to. ``upstream-repo.yaml`` is what upstream's
    own ``scripts/passk.py`` reads, kept runnable with ``--prompt-config`` so the difference
    between the two stays measurable rather than being a claim in the README.

    The paper's is a reconstruction of a typeset listing, so it cannot be pinned against a
    source; upstream's can be, and is -- ``test_upstream_template_still_matches_upstream``
    refetches the pinned file.
    """

    @pytest.mark.parametrize("fpath", [PROMPT_CONFIG_FPATH, UPSTREAM_PROMPT_CONFIG_FPATH])
    def test_is_a_valid_gym_prompt_config(self, fpath):
        config = load_prompt_config(str(fpath))
        assert "{formal_statement}" in config.user
        assert config.system is None, "upstream posts a single user message; a system prompt would deviate"

    def test_the_two_templates_really_differ(self):
        paper = load_prompt_config(str(PROMPT_CONFIG_FPATH)).user
        upstream = load_prompt_config(str(UPSTREAM_PROMPT_CONFIG_FPATH)).user
        assert paper != upstream
        # The substantive divergence, and the whole reason both are shipped: upstream permits
        # auxiliary declarations and names the banned tokens; the paper asks for step-by-step
        # reasoning instead. Running upstream's took Easy from a published 20.0% to 1/10.
        assert "step by step" in paper and "step by step" not in upstream
        assert "auxiliary definitions" in upstream and "auxiliary definitions" not in paper

    def test_upstream_template_still_matches_upstream(self):
        """Refetch the pinned `prompts/static_passk.md` and prove our transcription is exact.

        Stronger than committing a copy of the file: this checks against the pinned source
        itself, so a transcription slip cannot hide behind a copy that drifted with it. Skips
        offline -- it is a fidelity check, not a correctness one, and CI without egress should
        not fail on it.
        """
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(UPSTREAM_PROMPT_URL, timeout=30) as response:
                upstream = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"no network to reach {UPSTREAM_PROMPT_URL}: {exc}")

        # `.strip()` matches eval_common.load_prompt, which upstream applies to every template.
        assert load_prompt_config(str(UPSTREAM_PROMPT_CONFIG_FPATH)).user.strip() == upstream.strip()

    def test_filling_it_reproduces_the_prompt_the_benchmark_runs(self):
        """End-to-end: a shipped row plus the shipped template equals one user message.

        Locks the pair together, so a change to either the template or the row shape surfaces
        as a test failure rather than as a benchmark number that quietly stops being
        comparable to the paper's. The half that needs the network -- that formal_statement is
        the ``CAT_statement/*.lean`` bytes verbatim, trailing newline included -- is enforced
        in prepare.py, which fails if the .lean file and the JSONL record disagree.
        """
        config = load_prompt_config(str(PROMPT_CONFIG_FPATH))
        for row in _load_rows("example.jsonl"):
            messages = apply_prompt_to_row(row, config)["responses_create_params"]["input"]
            assert [m["role"] for m in messages] == ["user"]
            assert messages[0]["content"] == config.user.format(formal_statement=row["formal_statement"])
            assert row["formal_statement"] in messages[0]["content"]


class TestVerifierMetadataLifting:
    """Gym posts rows with `verifier_metadata` still nested; the request must accept that."""

    def test_nested_verifier_metadata_is_lifted(self):
        request = LeanCatVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=TestLeanCatApp()._create_response(""),
            verifier_metadata={
                "formal_statement": REFERENCE,
                "problem_id": "0044",
                "level": "Easy",
                "tag": ["Limit"],
                "domain": ["Category"],
            },
        )
        assert request.formal_statement == REFERENCE
        assert request.problem_id == "0044"
        assert request.level == "Easy"

    def test_top_level_fields_still_work(self):
        request = LeanCatVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=TestLeanCatApp()._create_response(""),
            formal_statement=REFERENCE,
            level="High",
        )
        assert request.formal_statement == REFERENCE
        assert request.level == "High"

    def test_top_level_wins_over_metadata(self):
        request = LeanCatVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=TestLeanCatApp()._create_response(""),
            formal_statement=REFERENCE,
            level="High",
            verifier_metadata={"formal_statement": "other", "level": "Easy"},
        )
        assert request.formal_statement == REFERENCE
        assert request.level == "High"
