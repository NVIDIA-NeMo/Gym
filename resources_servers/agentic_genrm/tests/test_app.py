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
from pathlib import Path
from unittest.mock import MagicMock

from pytest import fixture

from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_genrm.app import (
    AgenticGenRMResourcesServer,
    AgenticGenRMResourcesServerConfig,
    AgenticGenRMSeedSessionRequest,
    AgenticGenRMVerifyRequest,
    extract_code_blocks,
    split_genrm_prompt,
)
from resources_servers.generative_reward_model.app import (
    GenerativeRewardModelResourcesServer,
    GenerativeRewardModelVerifyRequest,
)


EXAMPLE_ROWS = [
    json.loads(line) for line in (Path(__file__).parents[1] / "data" / "example.jsonl").read_text().splitlines()
]
GT_OVERALL = {"score_1": 4.0, "score_2": 2.0, "ranking": 2.0}
VERDICT = json.dumps(
    {
        "rubric_evaluations": [
            {
                "rubric_id": 1,
                "response_1_analysis": "a",
                "response_2_analysis": "b",
                "score_1": 4,
                "score_2": 2,
                "ranking": 2,
            }
        ],
        "overall": {"response_1_analysis": "a", "response_2_analysis": "b", "score_1": 4, "score_2": 2, "ranking": 2},
    }
)
GT_RUBRICS = [{"rubric_id": 1, "score_1": 4.0, "score_2": 2.0, "ranking": 2.0}]
PROMPT = (
    "You are an expert evaluation judge.\n\n"
    "#### Conversation Context ####\nUser: Sort a list in Python.\n\n"
    "#### Responses to be Scored ####\n"
    "[The Start of Response 1]\nUse sorted:\n```python\nprint(sorted([3, 1, 2]))\n```\n[The End of Response 1]\n\n"
    "[The Start of Response 2]\nCall .sort():\n```\nxs.sort()\n```\n```bash\n\n```\n[The End of Response 2]\n\n"
    "#### Evaluation Plan ####\nCriteria: correctness\n"
)


def _message(text: str, idx: int = 0) -> dict:
    return {
        "type": "message",
        "id": f"msg_{idx}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(name: str, idx: int) -> list:
    return [
        {"type": "function_call", "call_id": f"call_{idx}", "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": f"call_{idx}", "output": "ok"},
    ]


def _verify_request(output: list, status=None) -> dict:
    response = {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }
    if status is not None:
        response |= {"status": status, "incomplete_details": {"reason": "max_output_tokens"}}
    return {
        "id": "row-1",
        "responses_create_params": {"input": [{"role": "user", "content": PROMPT}]},
        "response": response,
        "ground_truth_overall": GT_OVERALL,
        "ground_truth_rubric_scores": GT_RUBRICS,
    }


class TestAgenticGenRM:
    @fixture
    def config(self) -> AgenticGenRMResourcesServerConfig:
        return AgenticGenRMResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="app.py", name="agentic_genrm")

    @fixture
    def server(self, config) -> AgenticGenRMResourcesServer:
        return AgenticGenRMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_split_genrm_prompt_on_example_rows(self) -> None:
        for row in EXAMPLE_ROWS:
            prompt = row["responses_create_params"]["input"][-1]["content"]
            context, response_1, response_2 = split_genrm_prompt(prompt)
            assert context and response_1 and response_2
            assert context in prompt and response_1 in prompt and response_2 in prompt
            for part in (context, response_1, response_2):
                assert "[The Start of Response" not in part and "[The End of Response" not in part
            assert "#### Evaluation Plan ####" not in response_2

    def test_split_genrm_prompt_rejects_other_prompts(self) -> None:
        assert split_genrm_prompt("Just a question") is None
        assert split_genrm_prompt(PROMPT.replace("[The End of Response 2]", "")) is None

    def test_extract_code_blocks(self) -> None:
        _, response_1, response_2 = split_genrm_prompt(PROMPT)
        assert extract_code_blocks(response_1) == [("py", "print(sorted([3, 1, 2]))\n")]
        # Untagged blocks become .txt; empty blocks are skipped.
        assert extract_code_blocks(response_2) == [("txt", "xs.sort()\n")]

    async def test_seed_session_writes_code_blocks_as_files(self, server) -> None:
        body = AgenticGenRMSeedSessionRequest.model_validate(_verify_request([]))
        files = (await server.seed_session(MagicMock(), body)).sandbox_files
        assert files == {
            "/workspace/genrm/response_1_code/block_01.py": "print(sorted([3, 1, 2]))\n",
            "/workspace/genrm/response_2_code/block_01.txt": "xs.sort()\n",
        }

    async def test_seed_session_without_files(self, config) -> None:
        body = AgenticGenRMSeedSessionRequest.model_validate(_verify_request([]))
        config.sandbox_files_dir = None
        server = AgenticGenRMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert (await server.seed_session(MagicMock(), body)).sandbox_files == {}

        config.sandbox_files_dir = "/workspace/genrm/"
        server = AgenticGenRMResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert sorted((await server.seed_session(MagicMock(), body)).sandbox_files) == [
            "/workspace/genrm/response_1_code/block_01.py",
            "/workspace/genrm/response_2_code/block_01.txt",
        ]
        # Content parts and unrecognized prompts.
        parts = AgenticGenRMSeedSessionRequest.model_validate(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": PROMPT}]}]
                }
            }
        )
        assert len((await server.seed_session(MagicMock(), parts)).sandbox_files) == 2
        other = AgenticGenRMSeedSessionRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}}
        )
        assert (await server.seed_session(MagicMock(), other)).sandbox_files == {}
        assert (await server.seed_session(MagicMock(), AgenticGenRMSeedSessionRequest())).sandbox_files == {}

    async def test_verify_scores_only_the_final_message(self, server, config) -> None:
        output = [
            _message(f"Draft verdict before checking: {VERDICT}", 0),
            *_call("bash", 1),
            _message("The code runs.", 2),
            *_call("genrm_search_search", 3),
            *_call("genrm_search_browse", 4),
            *_call("bash", 5),
            *_call("read", 6),
            _message(VERDICT, 7),
        ]
        body = _verify_request(output)
        result = await server.verify(AgenticGenRMVerifyRequest.model_validate(body))
        assert result.reward == 0.0
        assert result.format_correct == 1.0
        assert result.final_message_found == 1.0
        assert result.response_incomplete == 0.0
        assert (result.num_tool_calls, result.num_bash_calls, result.num_search_calls, result.num_browse_calls) == (
            5.0,
            2.0,
            1.0,
            1.0,
        )
        assert (result.num_other_tool_calls, result.used_tools, result.used_web) == (1.0, 1.0, 1.0)
        # The whole trajectory comes back, not the scored message alone.
        assert len(result.response.output) == len(output)

        # The single-turn server joins every message: the draft plus the final verdict is two dicts.
        single_turn = GenerativeRewardModelResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        baseline = await single_turn.verify(GenerativeRewardModelVerifyRequest.model_validate(body))
        assert baseline.reward == config.parse_failure_penalty

    async def test_verify_fails_without_a_final_message(self, server, config) -> None:
        output = [_message(VERDICT, 0), *_call("bash", 1)]
        result = await server.verify(AgenticGenRMVerifyRequest.model_validate(_verify_request(output)))
        assert result.reward == config.parse_failure_penalty
        assert result.format_correct == 0.0
        assert result.final_message_found == 0.0

        result = await server.verify(AgenticGenRMVerifyRequest.model_validate(_verify_request([])))
        assert result.reward == config.parse_failure_penalty
        assert (result.final_message_found, result.used_tools, result.used_web) == (0.0, 0.0, 0.0)

    async def test_verify_fails_an_incomplete_response(self, server, config) -> None:
        body = _verify_request([_message(VERDICT, 0)], status="incomplete")
        result = await server.verify(AgenticGenRMVerifyRequest.model_validate(body))
        assert result.reward == config.parse_failure_penalty
        assert result.response_incomplete == 1.0
        assert result.final_message_found == 1.0

    async def test_verify_scores_errors_like_the_single_turn_server(self, server) -> None:
        wrong = json.loads(VERDICT)
        wrong["overall"] |= {"score_1": 2, "score_2": 4, "ranking": 5}
        body = _verify_request([*_call("bash", 0), _message(json.dumps(wrong), 1)])
        result = await server.verify(AgenticGenRMVerifyRequest.model_validate(body))
        # Overall: |4-2| + |2-4| + 2 * |2-5| = 10; the rubric is perfect.
        assert result.reward == -10.0
        assert result.overall_ranking_binary_acc == 0.0
