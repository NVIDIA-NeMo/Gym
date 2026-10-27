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
"""Tests for the DWRL pointwise GenRM resources server.

How DWRL pointwise works, which these tests pin down:
- verify() groups the rollouts of one prompt into a cohort of `num_rollouts_per_prompt`.
- Every rollout is judged on its own (see `_run_single_comparison`): the judge first writes an evaluation of the
  response, then a second 1-token call reads the probability of `score_token` ("yes") from the logprobs. That
  probability is the BT score, and it is the rollout's reward.
- The cohort's rewards are computed when the last rollout has been judged, whatever order they finish in.
"""

import asyncio
import logging
import math
import random
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pydantic import ValidationError
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.dwrl_genrm_compare_pointwise import app as app_module
from resources_servers.dwrl_genrm_compare_pointwise.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
    GenRMCompareResponse,
    GenRMCompareVerifyRequest,
    _input_to_conversation_history,
    convert_to_chat_completion_format,
    flatten_to_single_turn,
)
from resources_servers.dwrl_genrm_compare_pointwise.cohort import CohortRegistry, MemberJudgmentFailed
from resources_servers.dwrl_genrm_compare_pointwise.utils import extract_output_text, get_prompt_key_from_input


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dwrl_genrm_compare_pointwise.yaml"

DWRL_PROMPT = "CONTEXT:\n{context}\nRESPONSE:\n{response}\nPLAN:\n{principle}"
BT_PROMPT = "Is the response good? Answer yes or no."


def _make_config(**overrides) -> GenRMCompareConfig:
    kwargs = dict(
        # Required fields from the base server config
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        # GenRMCompareConfig fields
        name="dwrl_pointwise_compare_resources_server",
        genrm_model_server=ModelServerRef(type="responses_api_models", name="dwrl_model"),
        genrm_responses_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[], max_tokens=1024, temperature=0.7, top_p=0.9
        ),
        # DWRL specific
        score_token="yes",
        bt_prompt=BT_PROMPT,
        dwrl_prompt=DWRL_PROMPT,
    )
    kwargs.update(overrides)
    return GenRMCompareConfig(**kwargs)


def _make_server(config=None, server_client=None) -> GenRMCompareResourcesServer:
    # model_construct bypasses validation: the server client is replaced by a mock in the tests that need it.
    return GenRMCompareResourcesServer.model_construct(
        config=config or _make_config(), server_client=server_client or MagicMock()
    )


def _make_response(text: str, response_id: str = "resp") -> NeMoGymResponse:
    output = []
    if text:
        output.append(
            NeMoGymResponseOutputMessage(
                id=f"msg_{response_id}",
                role="assistant",
                status="completed",
                type="message",
                content=[NeMoGymResponseOutputText(text=text, type="output_text", annotations=[])],
            )
        )
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="dummy_model",
        tools=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        output=output,
        object="response",
    )


def _make_verify_request(
    answer: str, prompt: str = "What is 2+2?", principle=None, **identity
) -> GenRMCompareVerifyRequest:
    """`identity` is any of task_index, rollout_index and prompt_id."""
    return GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=prompt, type="message")]
        ),
        response=_make_response(answer),
        principle=principle,
        **identity,
    )


def _response_obj(text: str) -> dict:
    return _make_response(text).model_dump()


class TestGenRMCompareConfig:
    def test_defaults(self):
        config = _make_config()

        assert config.num_rollouts_per_prompt == 1
        assert config.num_judges_per_comparison == 1
        assert config.use_principle is False
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.default_score == 3.0
        assert config.default_ranking == 0.5
        assert config.genrm_parse_retries == 3
        assert config.debug_logging is False

    @pytest.mark.parametrize("missing", ["score_token", "bt_prompt", "dwrl_prompt"])
    def test_dwrl_specific_fields_are_required(self, missing):
        kwargs = dict(
            host="localhost",
            port=8000,
            entrypoint="app.py",
            domain="rlhf",
            genrm_model_server=ModelServerRef(type="responses_api_models", name="dwrl_model"),
            genrm_responses_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
            score_token="yes",
            bt_prompt=BT_PROMPT,
            dwrl_prompt=DWRL_PROMPT,
        )
        del kwargs[missing]
        with pytest.raises(ValidationError):
            GenRMCompareConfig(**kwargs)

    def test_at_least_one_judge_is_required(self):
        with pytest.raises(ValidationError):
            _make_config(num_judges_per_comparison=0)

    def test_cohorts_cannot_be_spread_over_several_worker_processes(self):
        # The cohorts live in the memory of one process, so rollouts of the same prompt must reach the same process.
        with pytest.raises(ValidationError, match="num_workers must be 1"):
            _make_config(num_rollouts_per_prompt=8, num_workers=4)

    @pytest.mark.parametrize("num_workers", [None, 1])
    def test_single_worker_process_is_accepted(self, num_workers):
        assert _make_config(num_rollouts_per_prompt=8, num_workers=num_workers).num_rollouts_per_prompt == 8

    def test_several_workers_are_fine_without_cohorts(self):
        # With one rollout per prompt verify() does not group anything.
        assert _make_config(num_rollouts_per_prompt=1, num_workers=4).num_workers == 4

    def test_packaged_config_matches_the_config_class(self):
        """configs/dwrl_genrm_compare_pointwise.yaml must stay in sync with GenRMCompareConfig."""
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        server_config = raw["dwrl_pointwise_compare_resources_server"]["resources_servers"][
            "dwrl_genrm_compare_pointwise"
        ]

        # `???` marks the values that are supplied when the run is launched.
        assert server_config["num_rollouts_per_prompt"] == "???"
        server_config["num_rollouts_per_prompt"] = 8

        config = GenRMCompareConfig(host="localhost", port=8000, name="dwrl", **server_config)

        assert config.genrm_model_server.name == "dwrl_model"
        assert config.num_rollouts_per_prompt == 8
        assert config.score_token == "yes"
        assert config.aggregator_method == "simple_tiebreaker"
        # The prompts must contain the placeholders that _run_single_comparison fills in.
        for placeholder in ("{context}", "{response}", "{principle}"):
            assert placeholder in config.dwrl_prompt
        # Literal braces in the JSON example are escaped so that str.format works.
        assert config.dwrl_prompt.strip().format(context="c", response="r", principle="p")


class TestRequestAndResponseModels:
    def test_compare_request(self):
        request = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "What is 2+2?"}],
            response_objs=[_response_obj("4"), _response_obj("Four")],
            principle="Be concise",
        )

        assert len(request.conversation_history) == 1
        assert len(request.response_objs) == 2
        assert request.principle == "Be concise"

    def test_compare_response(self):
        response = GenRMCompareResponse(rewards=[0.7, 0.2], metrics={"mean_individual_score": 0.45})

        assert response.rewards == approx([0.7, 0.2])
        assert response.comparison_results is None

    def test_verify_request_principle_is_optional(self):
        assert _make_verify_request("4").principle is None
        assert _make_verify_request("4", principle="Be concise").principle == "Be concise"

    @staticmethod
    def _verify_request_payload(**extra):
        return {
            "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="hello", type="message")]
            ),
            "response": _make_response("hi"),
            **extra,
        }

    def test_verify_request_identity_fields_default_to_none(self):
        request = _make_verify_request("4")

        assert request.task_index is None
        assert request.rollout_index is None
        assert request.prompt_id is None

    def test_verify_request_reads_the_gym_index_keys(self):
        # Gym's rollout collection stamps these keys on every row, and the agent forwards them to /verify.
        request = GenRMCompareVerifyRequest.model_validate(
            self._verify_request_payload(**{TASK_INDEX_KEY_NAME: 7, ROLLOUT_INDEX_KEY_NAME: 2})
        )

        assert request.task_index == 7
        assert request.rollout_index == 2

    def test_verify_request_index_fields_use_the_gym_key_names_as_aliases(self):
        fields = GenRMCompareVerifyRequest.model_fields

        assert fields["task_index"].alias == TASK_INDEX_KEY_NAME
        assert fields["rollout_index"].alias == ROLLOUT_INDEX_KEY_NAME

    def test_verify_request_can_also_be_built_from_the_field_names(self):
        request = GenRMCompareVerifyRequest(
            **self._verify_request_payload(task_index=7, rollout_index=2, prompt_id="prompt-123")
        )

        assert (request.task_index, request.rollout_index, request.prompt_id) == (7, 2, "prompt-123")

    def test_verify_request_keeps_unknown_fields(self):
        request = GenRMCompareVerifyRequest.model_validate(self._verify_request_payload(some_other_field="kept"))

        assert request.model_extra["some_other_field"] == "kept"

    def test_verify_request_rejects_a_non_integer_task_index(self):
        with pytest.raises(ValidationError):
            GenRMCompareVerifyRequest.model_validate(self._verify_request_payload(**{TASK_INDEX_KEY_NAME: "seven"}))


class TestGetVerifyCohortKey:
    """Rollouts that belong together must get the same key, and rollouts that do not must get different keys."""

    PROMPT = "What is 2+2?"

    def _key(self, prompt=PROMPT, principle=None, **identity):
        request = _make_verify_request("answer", prompt, principle, **identity)
        return _make_server()._get_verify_cohort_key(request, request.responses_create_params.input, principle)

    def _prompt_hash(self, prompt=PROMPT, principle=None):
        messages = [NeMoGymEasyInputMessage(role="user", content=prompt, type="message")]
        return get_prompt_key_from_input(messages, principle)

    def test_without_identity_the_key_is_the_prompt_hash(self):
        assert self._key() == self._prompt_hash()

    def test_task_index_is_preferred(self):
        assert self._key(task_index=7) == f"task_idx::7::{self._prompt_hash()}"

    def test_task_index_zero_is_a_valid_task(self):
        assert self._key(task_index=0) == f"task_idx::0::{self._prompt_hash()}"

    def test_prompt_id_is_used_when_there_is_no_task_index(self):
        assert self._key(prompt_id="prompt-123") == f"prompt_id::prompt-123::{self._prompt_hash()}"

    def test_task_index_wins_over_prompt_id(self):
        assert self._key(task_index=7, prompt_id="prompt-123") == f"task_idx::7::{self._prompt_hash()}"

    def test_principle_is_part_of_the_key(self):
        assert self._key(principle="Be concise", task_index=7) == (
            f"task_idx::7::{self._prompt_hash(principle='Be concise')}"
        )
        assert self._key(principle="Be concise", task_index=7) != self._key(principle="Be thorough", task_index=7)

    def test_same_prompt_in_different_tasks_gets_different_keys(self):
        assert self._key(task_index=1) != self._key(task_index=2)

    def test_same_prompt_with_different_prompt_ids_gets_different_keys(self):
        assert self._key(prompt_id="a") != self._key(prompt_id="b")

    def test_same_task_with_different_prompt_text_gets_different_keys(self):
        assert self._key("first prompt", task_index=7) != self._key("second prompt", task_index=7)

    def test_all_rollouts_of_a_task_share_one_key(self):
        keys = {self._key(task_index=7, rollout_index=i) for i in range(8)}

        assert len(keys) == 1

    def test_key_is_deterministic(self):
        assert self._key(task_index=7) == self._key(task_index=7)


class TestInputToConversationHistory:
    def test_dict_messages(self):
        assert _input_to_conversation_history([{"role": "user", "content": "Hello"}]) == [
            {"role": "user", "content": "Hello"}
        ]

    def test_pydantic_messages(self):
        messages = [
            NeMoGymEasyInputMessage(role="system", content="Be nice", type="message"),
            NeMoGymEasyInputMessage(role="user", content="Hello", type="message"),
        ]
        assert _input_to_conversation_history(messages) == [
            {"role": "system", "content": "Be nice"},
            {"role": "user", "content": "Hello"},
        ]

    def test_role_defaults_to_user_and_content_to_empty(self):
        assert _input_to_conversation_history([{}]) == [{"role": "user", "content": ""}]

    def test_only_output_text_parts_are_joined(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello "},
                    {"type": "input_text", "text": "ignored"},
                    {"type": "output_text", "text": "world"},
                ],
            }
        ]
        assert _input_to_conversation_history(messages) == [{"role": "assistant", "content": "Hello world"}]

    def test_none_and_empty_input(self):
        assert _input_to_conversation_history(None) == []
        assert _input_to_conversation_history([]) == []


class TestMessageHelpers:
    @staticmethod
    def _messages(*pairs):
        return [NeMoGymEasyInputMessage(role=role, content=content, type="message") for role, content in pairs]

    def test_flatten_labels_each_role(self):
        messages = self._messages(("system", " Be nice "), ("user", "Hi"), ("assistant", "Hello!"), ("user", "Bye"))
        assert flatten_to_single_turn(messages) == "System: Be nice\n\nUser: Hi\n\nAssistant: Hello!\n\nUser: Bye"

    def test_flatten_removes_thinking_from_assistant_turns(self):
        messages = self._messages(
            ("user", "Hi"),
            ("assistant", "<think>secret plan</think>Hello!"),
            ("assistant", "unterminated opener</think>Hi again"),
        )
        flattened = flatten_to_single_turn(messages)

        assert "secret plan" not in flattened
        assert "unterminated opener" not in flattened
        assert flattened == "User: Hi\n\nAssistant: Hello!\n\nAssistant: Hi again"

    def test_flatten_keeps_thinking_tags_in_user_turns(self):
        flattened = flatten_to_single_turn(self._messages(("user", "What does <think> mean?")))
        assert flattened == "User: What does <think> mean?"

    def test_convert_to_chat_completion_format(self):
        converted = convert_to_chat_completion_format(
            self._messages(("system", "s"), ("user", "u"), ("assistant", "a"), ("developer", "ignored"))
        )
        # Messages with roles the judge does not use are dropped.
        assert [m["role"] for m in converted] == ["system", "user", "assistant"]
        assert [m["content"] for m in converted] == ["s", "u", "a"]


def _http_response(payload) -> AsyncMock:
    response = AsyncMock()
    response.json = AsyncMock(return_value=payload)
    return response


def _evaluation_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _score_payload(*top_logprobs) -> dict:
    return {
        "choices": [
            {"logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in top_logprobs]}]}}
        ]
    }


JUDGE_JSON = '```json\n{"response_analysis": "Fine.", "score_1": 4}\n```'


class TestRunSingleComparison:
    """`_run_single_comparison` talks to the GenRM twice per rollout: evaluate, then read the yes-probability."""

    @staticmethod
    def _make(evaluation=JUDGE_JSON, score_logprobs=(("yes", math.log(0.75)), ("no", math.log(0.25))), **config):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                _http_response(_evaluation_payload(evaluation)),
                _http_response(_score_payload(*score_logprobs)),
            ]
        )
        return _make_server(_make_config(**config), client), client

    @staticmethod
    def _run(server, response_text="4", conversation=None, principle=None):
        conversation = conversation or [{"role": "user", "content": "What is 2+2?"}]
        return asyncio.run(
            server._run_single_comparison(conversation, _response_obj(response_text), principle=principle)
        )

    @staticmethod
    def _call(client, index):
        return client.post.call_args_list[index].kwargs

    def test_returns_parsed_score_and_bt_score(self):
        server, _ = self._make()
        score_1, bt_score = self._run(server)

        assert score_1 == approx(4.0)
        assert bt_score == approx(0.75)

    def test_makes_two_calls_to_the_genrm_server(self):
        server, client = self._make()
        self._run(server)

        assert client.post.call_count == 2
        for index in (0, 1):
            assert self._call(client, index)["server_name"] == "dwrl_model"
            assert self._call(client, index)["url_path"] == "/v1/chat/completions"

    def test_evaluation_request_contains_context_response_and_principle(self):
        server, client = self._make()
        self._run(
            server,
            response_text="The answer is 4.",
            conversation=[{"role": "user", "content": "What is 2+2?"}],
            principle="Be accurate.",
        )

        body = self._call(client, 0)["json"]
        assert len(body.messages) == 1
        assert body.messages[0]["role"] == "user"
        assert body.messages[0]["content"] == (
            "CONTEXT:\nUser: What is 2+2?\nRESPONSE:\nThe answer is 4.\nPLAN:\nBe accurate."
        )

    def test_evaluation_request_keeps_the_configured_sampling_params(self):
        server, client = self._make()
        self._run(server)

        body = self._call(client, 0)["json"]
        assert body.max_tokens == 1024
        assert body.temperature == approx(0.7)
        assert body.top_p == approx(0.9)
        assert body.logprobs is None

    def test_configured_params_are_not_modified_by_a_call(self):
        server, _ = self._make()
        self._run(server)

        params = server.config.genrm_responses_create_params
        assert params.messages == []
        assert params.max_tokens == 1024
        assert params.logprobs is None
        assert params.metadata is None

    def test_default_principle_is_used_when_none_is_given(self):
        server, client = self._make(default_principle="  The default plan.  ")
        self._run(server, principle=None)

        assert self._call(client, 0)["json"].messages[0]["content"].endswith("PLAN:\nThe default plan.")

    def test_empty_response_is_replaced_by_a_placeholder(self):
        server, client = self._make()
        self._run(server, response_text="")

        assert "RESPONSE:\nNone\nPLAN:" in self._call(client, 0)["json"].messages[0]["content"]

    def test_only_the_final_answer_of_the_rollout_is_judged(self):
        server, client = self._make()
        response_obj = {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hidden chain of thought"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Final answer."}]},
            ]
        }
        asyncio.run(server._run_single_comparison([{"role": "user", "content": "q"}], response_obj))

        content = self._call(client, 0)["json"].messages[0]["content"]
        assert "Final answer." in content
        assert "hidden chain of thought" not in content

    def test_score_request_continues_the_conversation_and_asks_for_logprobs(self):
        server, client = self._make(evaluation="My evaluation text.")
        self._run(server)

        body = self._call(client, 1)["json"]
        assert [m["role"] for m in body.messages] == ["user", "assistant", "user"]
        assert body.messages[1]["content"] == "My evaluation text."
        assert body.messages[2]["content"] == BT_PROMPT
        assert body.max_tokens == 1
        assert body.logprobs is True
        assert body.top_logprobs == 20
        assert body.temperature == approx(1.0)
        assert body.top_p == approx(1.0)

    def test_score_request_turns_thinking_off(self):
        server, client = self._make()
        self._run(server)

        metadata = self._call(client, 1)["json"].metadata
        assert '"enable_thinking": false' in metadata["chat_template_kwargs"]
        assert '"enable_thinking": false' in metadata["extra_body"]

    def test_bt_score_is_the_probability_of_the_score_token(self):
        server, _ = self._make(score_logprobs=(("no", math.log(0.9)), ("yes", math.log(0.1))))
        _, bt_score = self._run(server)

        assert bt_score == approx(0.1)

    def test_score_token_is_matched_ignoring_case_and_whitespace(self):
        server, _ = self._make(score_logprobs=(("no", math.log(0.6)), (" Yes", math.log(0.4))))
        _, bt_score = self._run(server)

        assert bt_score == approx(0.4)

    def test_exact_score_token_match_is_preferred(self):
        server, _ = self._make(score_logprobs=((" Yes", math.log(0.3)), ("yes", math.log(0.7))))
        _, bt_score = self._run(server)

        assert bt_score == approx(0.7)

    def test_custom_score_token(self):
        server, _ = self._make(score_token="good", score_logprobs=(("good", math.log(0.2)), ("yes", math.log(0.8))))
        _, bt_score = self._run(server)

        assert bt_score == approx(0.2)

    def test_score_token_missing_from_top_logprobs_gives_zero(self):
        # A token outside the top 20 has a probability of at most 1/20, so 0 is the right order of magnitude.
        # It is deliberately not `default_ranking`, which is the neutral value for a failed GenRM call and would be
        # far too high here.
        server, _ = self._make(
            score_logprobs=(("no", math.log(0.99)), ("maybe", math.log(0.01))), default_ranking=0.3
        )
        score_1, bt_score = self._run(server)

        assert bt_score == 0.0
        # The evaluation itself is still parsed.
        assert score_1 == approx(4.0)

    def test_missing_score_token_logs_a_warning(self, caplog):
        # Falling back to 0.0 looks like a genuine low score in the output, so it must be visible in the log. For
        # example, a `score_token` that does not match the model's tokens makes every rollout take this path.
        server, _ = self._make(
            score_token="Yes", score_logprobs=((" Yes", math.log(0.6)), ("no", math.log(0.4)))
        )
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "'Yes'" in message  # the configured score token
        assert "' Yes'" in message and "'no'" in message  # the tokens that were returned
        assert "0.0" in message

    @pytest.mark.parametrize(
        "logprobs",
        [
            (("yes", math.log(0.7)), ("no", math.log(0.3))),  # exact match
            (("no", math.log(0.3)), (" YES", math.log(0.7))),  # case and whitespace insensitive match
        ],
    )
    def test_found_score_token_does_not_log_a_warning(self, caplog, logprobs):
        server, _ = self._make(score_logprobs=logprobs)
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_empty_top_logprobs_gives_zero(self):
        server, _ = self._make(score_logprobs=())
        _, bt_score = self._run(server)

        assert bt_score == 0.0

    def test_a_listed_score_token_with_tiny_probability_is_still_used(self):
        # Present but unlikely is different from missing: the real (tiny) probability is returned.
        server, _ = self._make(score_logprobs=(("no", math.log(0.999)), ("yes", -20.0)))
        _, bt_score = self._run(server)

        assert bt_score == approx(math.exp(-20.0))
        assert bt_score > 0.0

    def test_thinking_is_removed_before_the_judge_json_is_parsed(self):
        evaluation = '<think>draft: {"score_1": 1}</think>\n```json\n{"score_1": 4}\n```'
        server, _ = self._make(evaluation=evaluation)
        score_1, _ = self._run(server)

        assert score_1 == approx(4.0)

    def test_unterminated_thinking_prefix_is_removed(self):
        evaluation = 'draft {"score_1": 1}</think>```json\n{"score_1": 5}\n```'
        server, _ = self._make(evaluation=evaluation)
        score_1, _ = self._run(server)

        assert score_1 == approx(5.0)

    def test_unparseable_evaluation_uses_default_score_but_keeps_the_bt_score(self):
        server, _ = self._make(evaluation="I could not decide.", default_score=-100.0)
        score_1, bt_score = self._run(server)

        assert score_1 == approx(-100.0)
        assert bt_score == approx(0.75)

    def test_unparseable_evaluation_is_not_retried(self):
        # genrm_parse_retries only matters if parsing raises, which it does not here (raise_on_fail=False).
        server, client = self._make(evaluation="I could not decide.", genrm_parse_retries=3)
        self._run(server)

        assert client.post.call_count == 2

    def test_server_error_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=RuntimeError("GenRM server is down"))
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, 0.5)

    def test_malformed_server_response_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(return_value=_http_response({"unexpected": "shape"}))
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, 0.5)

    def test_missing_logprobs_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                _http_response(_evaluation_payload(JUDGE_JSON)),
                _http_response({"choices": [{"message": {"content": "yes"}}]}),
            ]
        )
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, 0.5)


class TestJudgeRollout:
    def test_runs_one_judge_call_per_judge(self, monkeypatch):
        calls = []

        async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
            calls.append(principle)
            return 3.0, 0.1 * len(calls)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
        server = _make_server(_make_config(num_judges_per_comparison=3))

        results = asyncio.run(server._judge_rollout([{"role": "user", "content": "q"}], _response_obj("a"), "plan"))

        assert len(results) == 3
        assert calls == ["plan", "plan", "plan"]


NUM_ROLLOUTS = 8

# The BT score the fake judge returns for "rollout-<n>".
BT_SCORES = {i: 0.05 + 0.1 * i for i in range(NUM_ROLLOUTS)}


class FakeJudge:
    """Stands in for `_run_single_comparison`. Each call blocks until the test releases its gate."""

    def __init__(self):
        self.gates = {}
        self.calls = []

    def gate(self, rollout_id):
        return self.gates.setdefault(rollout_id, asyncio.Event())

    async def __call__(self, server, conversation_history, response_obj, principle=None):
        rollout_id = int(extract_output_text(response_obj).rsplit("-", 1)[1])
        self.calls.append(rollout_id)
        await self.gate(rollout_id).wait()
        return 3.0, BT_SCORES[rollout_id]


@pytest.fixture
def judge(monkeypatch):
    fake = FakeJudge()

    # A plain function, so that it is bound to the server like the method it replaces.
    async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
        return await fake(server, conversation_history, response_obj, principle)

    monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
    return fake


@pytest.fixture(autouse=True)
def fresh_cohorts(monkeypatch):
    monkeypatch.setattr(app_module, "_cohorts", CohortRegistry())


async def _settle():
    """Give every runnable task enough event loop turns to run until it blocks again."""
    for _ in range(25):
        await asyncio.sleep(0)


async def _start(server, ids, prompt="What is 2+2?", principle=None, **identity):
    """Start one verify() per id, in order, and let each reach its judge call.

    `identity` (task_index and/or prompt_id) is sent with every request. A task_index is accompanied by the
    rollout's own index, like the requests that Gym's rollout collection sends.
    """

    def request(i):
        extra = dict(identity, rollout_index=i) if "task_index" in identity else identity
        return _make_verify_request(f"rollout-{i}", prompt, principle, **extra)

    tasks = [asyncio.create_task(server.verify(request(i))) for i in ids]
    await _settle()
    return tasks


async def _release(judge, ids):
    """Let the judge calls for `ids` finish, one at a time, in the given order."""
    for i in ids:
        judge.gate(i).set()
        await _settle()


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=10))


def _server(num_judges=1):
    return _make_server(_make_config(num_rollouts_per_prompt=NUM_ROLLOUTS, num_judges_per_comparison=num_judges))


class TestVerify:
    def test_single_rollout_per_prompt_returns_default_without_judging(self, judge):
        async def scenario():
            server = _make_server(_make_config(num_rollouts_per_prompt=1, default_score=-100.0))
            return await server.verify(_make_verify_request("rollout-0"))

        assert _run(scenario()).reward == -100.0
        assert judge.calls == []

    def test_response_carries_the_original_request(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert extract_output_text(results[2].response.model_dump()) == "rollout-2"
        assert results[2].responses_create_params.input[0].content == "What is 2+2?"

    def test_each_rollout_gets_its_own_bt_score_as_reward(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert sorted(judge.calls) == list(range(NUM_ROLLOUTS))

    def test_nothing_resolves_until_the_whole_cohort_is_judged(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS - 1))
            assert not any(t.done() for t in tasks)

            await _release(judge, [NUM_ROLLOUTS - 1])
            return await asyncio.gather(*tasks)

        assert len(_run(scenario())) == NUM_ROLLOUTS

    def test_last_arrival_finishing_first_does_not_trigger_extra_judge_calls(self, judge):
        # Regression test: the last rollout to arrive used to finalize the cohort as soon as its own judgment
        # returned, and re-judged every rollout that was still in flight (the `SHOULD_NOT_HAPPEN` log lines).
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))

            await _release(judge, [NUM_ROLLOUTS - 1])
            assert not any(t.done() for t in tasks), "cohort finished before every rollout was judged"

            await _release(judge, range(NUM_ROLLOUTS - 1))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert sorted(judge.calls) == list(range(NUM_ROLLOUTS))

    def test_any_completion_order_gives_the_same_rewards(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, [3, 7, 0, 5, 1, 6, 2, 4])
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert len(judge.calls) == NUM_ROLLOUTS

    def test_rollouts_arriving_after_others_were_already_judged(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(4))
            await _release(judge, range(4))
            assert not any(t.done() for t in tasks)

            tasks += await _start(server, range(4, NUM_ROLLOUTS))
            await _release(judge, range(4, NUM_ROLLOUTS))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert len(judge.calls) == NUM_ROLLOUTS

    def test_multiple_judges_are_averaged(self, judge):
        async def scenario():
            server = _server(num_judges=2)
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, reversed(range(NUM_ROLLOUTS)))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        # The fake returns the same BT score for both judges of a rollout, so the average is unchanged.
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert len(judge.calls) == 2 * NUM_ROLLOUTS

    def test_same_prompt_twice_forms_two_cohorts(self, judge):
        async def scenario():
            server = _server()
            first = await _start(server, range(NUM_ROLLOUTS))
            second = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*first), await asyncio.gather(*second)

        first, second = _run(scenario())
        expected = approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert [r.reward for r in first] == expected
        assert [r.reward for r in second] == expected
        assert len(judge.calls) == 2 * NUM_ROLLOUTS
        assert app_module._cohorts.num_open == 0

    def test_different_prompts_are_independent_cohorts(self, judge):
        async def scenario():
            server = _server()
            a = await _start(server, range(NUM_ROLLOUTS), prompt="prompt a")
            b = await _start(server, range(NUM_ROLLOUTS), prompt="prompt b")
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*a, *b)

        assert len(_run(scenario())) == 2 * NUM_ROLLOUTS
        assert app_module._cohorts.num_open == 0

    def test_identical_prompts_from_different_tasks_are_independent_cohorts(self, judge):
        async def scenario():
            server = _server()
            a = await _start(server, range(4), task_index=0)
            b = await _start(server, range(4, NUM_ROLLOUTS), task_index=1)
            # Same prompt text, but two tasks: two cohorts, each still filling. Without the task index both would
            # be one cohort that is already full.
            assert app_module._cohorts.num_open == 2

            a += await _start(server, range(4, NUM_ROLLOUTS), task_index=0)
            b += await _start(server, range(4), task_index=1)
            assert app_module._cohorts.num_open == 0
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*a), await asyncio.gather(*b)

        a, b = _run(scenario())
        assert [r.reward for r in a] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert [r.reward for r in b] == approx([BT_SCORES[i] for i in [*range(4, NUM_ROLLOUTS), *range(4)]])

    def test_identical_prompts_with_different_prompt_ids_are_independent_cohorts(self, judge):
        async def scenario():
            server = _server()
            a = await _start(server, range(4), prompt_id="a")
            b = await _start(server, range(4), prompt_id="b")
            assert app_module._cohorts.num_open == 2

            a += await _start(server, range(4, NUM_ROLLOUTS), prompt_id="a")
            b += await _start(server, range(4, NUM_ROLLOUTS), prompt_id="b")
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*a, *b)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)] * 2)

    def test_rollouts_of_one_task_join_a_single_cohort(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS - 1), task_index=5)
            assert app_module._cohorts.num_open == 1

            tasks += await _start(server, [NUM_ROLLOUTS - 1], task_index=5)
            assert app_module._cohorts.num_open == 0
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])

    def test_same_prompt_with_different_principles_are_independent_cohorts(self, judge):
        async def scenario():
            server = _server()
            a = await _start(server, range(NUM_ROLLOUTS), principle="Be concise")
            await _release(judge, range(NUM_ROLLOUTS - 1))
            b = await _start(server, range(NUM_ROLLOUTS), principle="Be thorough")
            # Both cohorts are full, and each is still waiting for its last judgment. If `b` had been merged into
            # `a`, `a` would hold 16 rollouts and would not be waiting for rollout 7 alone.
            assert not any(t.done() for t in a + b)
            assert app_module._cohorts.num_open == 0
            await _release(judge, [NUM_ROLLOUTS - 1])
            return await asyncio.gather(*a), await asyncio.gather(*b)

        a, b = _run(scenario())
        expected = approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])
        assert [r.reward for r in a] == expected
        assert [r.reward for r in b] == expected


class TestVerifyRepeatedRequests:
    """A request that repeats one that is already being handled (e.g. a client retry) must not take a slot.

    If it did, the cohort would fill up without one of its real rollouts, and that rollout would wait forever in a
    cohort of its own. A repeat is recognized by its task index, its rollout index and its response.
    """

    def test_repeat_while_the_cohort_is_still_filling(self, judge):
        async def scenario():
            server = _server()
            first = await _start(server, range(NUM_ROLLOUTS - 1), task_index=0)
            repeat = await _start(server, [3], task_index=0)
            last = await _start(server, [NUM_ROLLOUTS - 1], task_index=0)
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*first), await asyncio.gather(*repeat), await asyncio.gather(*last)

        first, repeat, last = _run(scenario())
        assert [r.reward for r in first] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS - 1)])
        assert repeat[0].reward == approx(BT_SCORES[3])
        assert last[0].reward == approx(BT_SCORES[NUM_ROLLOUTS - 1])
        # The repeat was not judged again.
        assert sorted(judge.calls) == list(range(NUM_ROLLOUTS))
        assert app_module._cohorts.num_active == 0

    def test_repeat_after_the_cohort_has_filled_up(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            assert app_module._cohorts.num_open == 0  # Full, and being judged.

            repeat = await _start(server, [5], task_index=0)
            assert app_module._cohorts.num_open == 0  # The repeat did not start a cohort of its own.
            assert app_module._cohorts.num_active == 1

            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks), await asyncio.gather(*repeat)

        tasks, repeat = _run(scenario())
        assert repeat[0].reward == approx(BT_SCORES[5])
        assert tasks[5].reward == approx(BT_SCORES[5])
        assert sorted(judge.calls) == list(range(NUM_ROLLOUTS))
        assert app_module._cohorts.num_active == 0

    def test_repeat_waits_for_the_original_when_the_original_finishes_last(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            repeat = await _start(server, [3], task_index=0)

            await _release(judge, [i for i in range(NUM_ROLLOUTS) if i != 3])
            assert not repeat[0].done()
            assert not tasks[3].done()

            await _release(judge, [3])
            return tasks[3].result(), repeat[0].result()

        original, repeat = _run(scenario())
        assert original.reward == repeat.reward == approx(BT_SCORES[3])

    def test_several_repeats_all_get_the_reward(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            repeats = await _start(server, [2, 2, 2], task_index=0)
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks), await asyncio.gather(*repeats)

        tasks, repeats = _run(scenario())
        assert [r.reward for r in repeats] == approx([BT_SCORES[2]] * 3)
        assert len(judge.calls) == NUM_ROLLOUTS

    def test_cancelling_a_repeat_does_not_affect_the_original_or_the_cohort(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            repeat = await _start(server, [3], task_index=0)

            repeat[0].cancel()
            await _settle()
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks), await asyncio.gather(*repeat, return_exceptions=True)

        tasks, repeat = _run(scenario())
        assert isinstance(repeat[0], asyncio.CancelledError)
        assert [r.reward for r in tasks] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])

    def test_cancelling_the_original_fails_the_repeat_and_not_the_cohort(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            repeat = await _start(server, [3], task_index=0)

            tasks[3].cancel()
            await _settle()
            await _release(judge, [i for i in range(NUM_ROLLOUTS) if i != 3])
            return await asyncio.gather(*tasks, *repeat, return_exceptions=True)

        results = _run(scenario())
        assert isinstance(results[3], asyncio.CancelledError)
        # The repeat is not handed the neutral reward that stands in for the cancelled judgment.
        assert isinstance(results[NUM_ROLLOUTS], MemberJudgmentFailed)
        for i in range(NUM_ROLLOUTS):
            if i != 3:
                assert results[i].reward == approx(BT_SCORES[i])

    def test_a_judge_error_in_the_original_fails_the_repeat_too(self, judge, monkeypatch):
        async def flaky_judge(server, conversation_history, response_obj, principle=None):
            if extract_output_text(response_obj) == "rollout-5":
                raise RuntimeError("judge exploded")
            return await judge(server, conversation_history, response_obj, principle)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", flaky_judge)

        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            repeat = await _start(server, [5], task_index=0)
            await _release(judge, [i for i in range(NUM_ROLLOUTS) if i != 5])
            return await asyncio.gather(*tasks, *repeat, return_exceptions=True)

        results = _run(scenario())
        assert type(results[5]) is RuntimeError  # The original reports its own error.
        assert isinstance(results[NUM_ROLLOUTS], MemberJudgmentFailed)
        assert "judge exploded" in str(results[NUM_ROLLOUTS])
        for i in range(NUM_ROLLOUTS):
            if i != 5:
                assert results[i].reward == approx(BT_SCORES[i])

    def test_a_different_response_with_the_same_index_is_not_a_repeat(self, judge):
        async def scenario():
            server = _server()
            first = await _start(server, range(NUM_ROLLOUTS - 1), task_index=0)
            # Same task and rollout index as rollout 3, but another response: a different rollout, so it takes the
            # last slot of the cohort.
            different = asyncio.create_task(
                server.verify(_make_verify_request(f"rollout-{NUM_ROLLOUTS - 1}", task_index=0, rollout_index=3))
            )
            await _settle()
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*first), await different

        first, different = _run(scenario())
        assert different.reward == approx(BT_SCORES[NUM_ROLLOUTS - 1])
        assert [r.reward for r in first] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS - 1)])
        assert sorted(judge.calls) == list(range(NUM_ROLLOUTS))

    def test_without_a_rollout_index_nothing_is_a_repeat(self, judge):
        # Callers that do not send indices behave as before: two identical requests are two members.
        ids = [0, 1, 2, 3, 3, 4, 5, 6]

        async def scenario():
            server = _server()
            tasks = await _start(server, ids)
            await _release(judge, set(ids))
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in ids])
        assert Counter(judge.calls)[3] == 2

    def test_the_same_rollout_in_another_task_is_not_a_repeat(self, judge):
        async def scenario():
            server = _server()
            a = await _start(server, [3], task_index=0)
            b = await _start(server, [3], task_index=1)
            assert app_module._cohorts.num_open == 2
            for task in a + b:
                task.cancel()
            await _settle()

        _run(scenario())

    def test_a_repeat_is_logged(self, judge, caplog):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=4)
            with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
                repeat = await _start(server, [3], task_index=4)
            await _release(judge, range(NUM_ROLLOUTS))
            await asyncio.gather(*tasks, *repeat)

        _run(scenario())
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(messages) == 1
        assert "Repeated verify request" in messages[0]
        assert "rollout_index=3" in messages[0] and "task_index=4" in messages[0]

    def test_a_repeat_after_the_cohort_was_finalized_is_not_recognized(self, judge):
        # Known limitation: a cohort forgets its members when it is finalized, so a repeat that arrives later is
        # treated as a new rollout and waits in a cohort of its own. Replace this test if that is ever handled.
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS), task_index=0)
            await _release(judge, range(NUM_ROLLOUTS))
            await asyncio.gather(*tasks)
            assert app_module._cohorts.num_active == 0

            late = await _start(server, [3], task_index=0)
            assert app_module._cohorts.num_open == 1
            assert not late[0].done()
            late[0].cancel()
            await _settle()

        _run(scenario())


class TestVerifyFailures:
    def test_cancelled_rollout_does_not_block_the_cohort(self, judge):
        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            tasks[3].cancel()
            await _settle()
            await _release(judge, [i for i in range(NUM_ROLLOUTS) if i != 3])
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())
        assert isinstance(results[3], asyncio.CancelledError)
        for i in range(NUM_ROLLOUTS):
            if i != 3:
                assert results[i].reward == approx(BT_SCORES[i])

    def test_judge_error_fails_that_rollout_only(self, judge, monkeypatch):
        async def flaky_judge(server, conversation_history, response_obj, principle=None):
            if extract_output_text(response_obj) == "rollout-5":
                raise RuntimeError("judge exploded")
            return await judge(server, conversation_history, response_obj, principle)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", flaky_judge)

        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, [i for i in range(NUM_ROLLOUTS) if i != 5])
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())
        assert isinstance(results[5], RuntimeError)
        for i in range(NUM_ROLLOUTS):
            if i != 5:
                assert results[i].reward == approx(BT_SCORES[i])

    def test_aggregation_failure_fails_every_rollout_instead_of_hanging(self, judge, monkeypatch):
        def broken_aggregate(**kwargs):
            raise ValueError("aggregation exploded")

        monkeypatch.setattr(app_module, "aggregate_scores", broken_aggregate)

        async def scenario():
            server = _server()
            tasks = await _start(server, range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks, return_exceptions=True)

        assert all(isinstance(r, ValueError) for r in _run(scenario()))


class TestVerifyRandomizedSchedules:
    """Many cohorts in flight at once, with random arrival times, judge latencies, failures and cancellations.

    The hand-written scenarios above each pin down one interleaving. This checks the invariants over many random ones:
    - nothing hangs, whatever the order in which the judgments finish;
    - every rollout gets its own reward, never that of another rollout;
    - every rollout is judged exactly once per judge (no redundant judge calls);
    - every cohort is finalized exactly once, and no cohort is left open.
    """

    NUM_SEEDS = 40
    ROLLOUTS_PER_PROMPT = 4

    def test_random_schedules(self, monkeypatch):
        state = {}

        async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
            rollout_id = int(extract_output_text(response_obj).rsplit("-", 1)[1])
            state["calls"][rollout_id] += 1
            await asyncio.sleep(state["rng"].random() * 0.003)
            if rollout_id in state["fail_ids"]:
                raise RuntimeError(f"judge failed for rollout {rollout_id}")
            return 3.0, state["bt"][rollout_id]

        finalized = []
        original_finalize = GenRMCompareResourcesServer._finalize_cohort

        def counting_finalize(server, cohort):
            finalized.append(cohort)
            return original_finalize(server, cohort)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
        monkeypatch.setattr(GenRMCompareResourcesServer, "_finalize_cohort", counting_finalize)

        for seed in range(self.NUM_SEEDS):
            monkeypatch.setattr(app_module, "_cohorts", CohortRegistry())
            finalized.clear()
            try:
                self._run_seed(seed, state, finalized)
            except AssertionError as e:
                raise AssertionError(f"seed={seed}: {e}") from e

    def _run_seed(self, seed, state, finalized):
        rng = random.Random(seed)
        n = self.ROLLOUTS_PER_PROMPT
        num_prompts = rng.randint(1, 4)
        repeats = rng.randint(1, 2)  # The same prompt can show up more than once; each copy needs its own cohort.
        num_judges = rng.choice([1, 2])

        # Every prompt gets `repeats * n` rollouts, so every cohort fills up.
        members = [(prompt, k) for prompt in range(num_prompts) for k in range(repeats * n)]
        ids = list(range(len(members)))
        prompt_of = {rollout_id: members[rollout_id][0] for rollout_id in ids}
        arrival_order = rng.sample(ids, len(ids))

        state.update(
            rng=rng,
            calls=Counter(),
            bt={rollout_id: rng.uniform(0.01, 0.99) for rollout_id in ids},
            fail_ids={rollout_id for rollout_id in ids if rng.random() < 0.05},
        )
        cancel_ids = {rollout_id for rollout_id in ids if rng.random() < 0.05}
        # How a request identifies its prompt: by hash only, by task index, or by prompt id.
        identity_mode = rng.choice(["none", "task_index", "prompt_id"])
        # With a task index (and so a rollout index) a request can also be delivered twice. The second delivery is made
        # right away, i.e. while the cohort is still being handled; a cancelled request is left out because its
        # cancellation could finalize the cohort before the repeat arrives.
        repeat_ids = {
            rollout_id
            for rollout_id in ids
            if identity_mode == "task_index" and rng.random() < 0.15 and rollout_id not in cancel_ids
        }

        async def scenario():
            server = _make_server(
                _make_config(num_rollouts_per_prompt=n, num_judges_per_comparison=num_judges)
            )
            loop = asyncio.get_running_loop()
            tasks = {}
            repeat_tasks = {}
            for rollout_id in arrival_order:
                await asyncio.sleep(rng.random() * 0.003)
                identity = {}
                if identity_mode == "task_index":
                    identity = dict(task_index=prompt_of[rollout_id], rollout_index=members[rollout_id][1])
                elif identity_mode == "prompt_id":
                    identity = dict(prompt_id=f"prompt-{prompt_of[rollout_id]}")
                request = _make_verify_request(
                    f"rollout-{rollout_id}", prompt=f"prompt {prompt_of[rollout_id]}", **identity
                )
                tasks[rollout_id] = asyncio.create_task(server.verify(request))
                await asyncio.sleep(0)  # Let it run up to its first await: it has joined its cohort.
                if rollout_id in repeat_ids:
                    repeat_tasks[rollout_id] = asyncio.create_task(server.verify(request))
                    await asyncio.sleep(0)
                if rollout_id in cancel_ids:
                    loop.call_later(rng.random() * 0.004, tasks[rollout_id].cancel)
            everything = [*tasks.values(), *repeat_tasks.values()]
            results = await asyncio.wait_for(asyncio.gather(*everything, return_exceptions=True), timeout=10)
            return dict(zip(tasks.keys(), results)), dict(zip(repeat_tasks.keys(), results[len(tasks) :]))

        try:
            results, repeat_results = asyncio.run(scenario())
        except (TimeoutError, asyncio.TimeoutError):
            raise AssertionError("verify() calls hung") from None

        for rollout_id, result in results.items():
            if isinstance(result, asyncio.CancelledError):
                assert rollout_id in cancel_ids, f"rollout {rollout_id} was cancelled without being asked to"
            elif isinstance(result, RuntimeError):
                assert rollout_id in state["fail_ids"], f"unexpected error for rollout {rollout_id}: {result!r}"
            elif isinstance(result, BaseException):
                raise AssertionError(f"unexpected {result!r} for rollout {rollout_id}")
            else:
                # The reward must be this rollout's own BT score (every judge returns it, so the mean is unchanged).
                assert result.reward == approx(state["bt"][rollout_id]), f"wrong reward for rollout {rollout_id}"
                assert extract_output_text(result.response.model_dump()) == f"rollout-{rollout_id}"

            calls = state["calls"][rollout_id]
            if rollout_id in state["fail_ids"] or rollout_id in cancel_ids:
                assert calls <= num_judges
            else:
                assert calls == num_judges, f"rollout {rollout_id} was judged {calls} times, not {num_judges}"

        for rollout_id, result in repeat_results.items():
            # A repeat gets the reward of the request it repeats, or fails if that request failed.
            if isinstance(result, BaseException):
                assert isinstance(result, MemberJudgmentFailed), f"unexpected {result!r} for a repeat of {rollout_id}"
                assert rollout_id in state["fail_ids"], f"repeat of {rollout_id} failed but its original did not"
            else:
                assert result.reward == approx(state["bt"][rollout_id]), f"wrong reward for a repeat of {rollout_id}"

        assert len(finalized) == num_prompts * repeats, "every cohort must be finalized exactly once"
        assert len({id(cohort) for cohort in finalized}) == len(finalized)
        assert all(cohort.is_complete for cohort in finalized)
        assert app_module._cohorts.num_open == 0
        assert app_module._cohorts.num_active == 0
