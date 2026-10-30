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
"""Tests for the DWRL pairwise GenRM resources server.

How DWRL pairwise works, which these tests pin down:
- A comparison of two responses (see `_run_single_comparison`) takes two GenRM calls: the judge first writes an
  evaluation of both responses, then a second 1-token call reads the probability of `score_token` ("response 1 is
  better") from the logprobs. That probability is the result of the comparison.
- The rollouts of a prompt form a cohort. Their comparisons (circular or all pairs, plus every rollout against the
  reference answer if there is one) are fitted together to Bradley-Terry ratings, which are the rewards. A rollout's
  reward depends on the whole cohort, so verify() returns only once the whole cohort has been judged.
- A comparison is judged as soon as both its responses have arrived, by the request that arrived last of the two. The
  cohort is finished by whichever request finishes last, whatever the order in which requests arrived or finished.
"""

import asyncio
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from pydantic import ValidationError
from pytest import approx

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.dwrl_genrm_compare_pairwise import app as app_module
from resources_servers.dwrl_genrm_compare_pairwise.app import (
    GenRMCompareConfig,
    GenRMCompareRequest,
    GenRMCompareResourcesServer,
    GenRMCompareResponse,
    GenRMCompareVerifyRequest,
    _input_to_conversation_history,
    _reference_response_obj,
    _response_digest,
    convert_to_chat_completion_format,
    flatten_to_single_turn,
)
from resources_servers.dwrl_genrm_compare_pairwise.cohort import CohortRegistry, MemberJudgmentFailed
from resources_servers.dwrl_genrm_compare_pairwise.utils import (
    aggregate_scores,
    extract_output_text,
    generate_comparison_pairs,
)


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "dwrl_genrm_compare_pairwise.yaml"

DWRL_PROMPT = "CONTEXT:\n{context}\nR1:\n{response_1}\nR2:\n{response_2}\nPLAN:\n{principle}"
BT_PROMPT = "Which response is better? Answer 1 or 2."


def _make_config(**overrides) -> GenRMCompareConfig:
    kwargs = dict(
        # Required fields from the base server config
        host="localhost",
        port=8000,
        entrypoint="app.py",
        domain="rlhf",
        # GenRMCompareConfig fields
        name="dwrl_pairwise_compare_resources_server",
        genrm_model_server=ModelServerRef(type="responses_api_models", name="dwrl_model"),
        genrm_responses_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[], max_tokens=1024, temperature=0.7, top_p=0.9
        ),
        # DWRL specific
        score_token="1",
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
    answer: str, prompt: str = "What is 2+2?", principle=None, ref_answer=None, **identity
) -> GenRMCompareVerifyRequest:
    """`identity` is any of task_index, rollout_index and prompt_id."""
    return GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=prompt, type="message")]
        ),
        response=_make_response(answer),
        principle=principle,
        ref_answer=ref_answer,
        **identity,
    )


def _response_obj(text: str) -> dict:
    return _make_response(text).model_dump()


@pytest.fixture(autouse=True)
def fresh_cohorts(monkeypatch):
    """Every test starts without cohorts, and ends by looking at the ones it left behind."""
    registry = CohortRegistry()
    monkeypatch.setattr(app_module, "_cohorts", registry)
    return registry


class TestGenRMCompareConfig:
    def test_defaults(self):
        config = _make_config()

        assert config.num_rollouts_per_prompt == 1
        assert config.comparison_strategy == "circular"
        assert config.num_judges_per_comparison == 1
        assert config.use_principle is False
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.group_style_penalty_coeff == 0.0
        assert config.default_score == 3.0
        assert config.default_ranking == 0.5
        assert config.genrm_parse_retries == 0
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
            score_token="1",
            bt_prompt=BT_PROMPT,
            dwrl_prompt=DWRL_PROMPT,
        )
        del kwargs[missing]
        with pytest.raises(ValidationError):
            GenRMCompareConfig(**kwargs)

    def test_at_least_one_judge_is_required(self):
        with pytest.raises(ValidationError):
            _make_config(num_judges_per_comparison=0)

    @pytest.mark.parametrize("default_ranking", [3.5, 1.5, -0.1])
    def test_default_ranking_is_a_probability(self, default_ranking):
        # It is added to the win counts that the ratings are fitted to, so a 1-6 ranking would corrupt the fit.
        with pytest.raises(ValidationError):
            _make_config(default_ranking=default_ranking)

    @pytest.mark.parametrize("default_ranking", [0.0, 0.5, 1.0])
    def test_default_ranking_accepts_probabilities(self, default_ranking):
        assert _make_config(default_ranking=default_ranking).default_ranking == default_ranking

    def test_cohorts_cannot_be_spread_over_several_worker_processes(self):
        with pytest.raises(ValidationError, match="num_workers must be 1"):
            _make_config(num_rollouts_per_prompt=8, num_workers=4)

    @pytest.mark.parametrize("num_workers", [None, 1])
    def test_a_single_worker_process_is_fine(self, num_workers):
        assert _make_config(num_rollouts_per_prompt=8, num_workers=num_workers).num_workers == num_workers

    def test_several_worker_processes_are_fine_without_cohorts(self):
        # With num_rollouts_per_prompt <= 1 verify() does not group anything.
        assert _make_config(num_rollouts_per_prompt=1, num_workers=4).num_workers == 4

    def test_packaged_config_matches_the_config_class(self):
        """configs/dwrl_genrm_compare_pairwise.yaml must stay in sync with GenRMCompareConfig."""
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        server_config = raw["dwrl_pairwise_compare_resources_server"]["resources_servers"][
            "dwrl_genrm_compare_pairwise"
        ]

        # `???` marks the values that are supplied when the run is launched: the size of a cohort.
        assert server_config["num_rollouts_per_prompt"] == "???"
        server_config["num_rollouts_per_prompt"] = 8
        assert "???" not in yaml.safe_dump(server_config)

        config = GenRMCompareConfig(host="localhost", port=8000, name="dwrl", **server_config)

        assert config.genrm_model_server.name == "dwrl_model"
        assert config.num_rollouts_per_prompt == 8
        assert config.comparison_strategy == "circular"
        assert config.score_token == "1"
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.group_style_penalty_coeff == 0.0
        assert 0.0 <= config.default_ranking <= 1.0
        # The prompts must contain the placeholders that _run_single_comparison fills in.
        for placeholder in ("{context}", "{response_1}", "{response_2}", "{principle}"):
            assert placeholder in config.dwrl_prompt
        # Literal braces in the JSON example are escaped so that str.format works.
        assert config.dwrl_prompt.strip().format(context="c", response_1="r1", response_2="r2", principle="p")


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
        response = GenRMCompareResponse(rewards=[0.7, -0.2], metrics={"mean_individual_score": 0.45})

        assert response.rewards == approx([0.7, -0.2])
        assert response.comparison_results is None

    def test_verify_request_principle_and_reference_answer_are_optional(self):
        request = _make_verify_request("4")
        assert request.principle is None
        assert request.ref_answer is None

        request = _make_verify_request("4", principle="Be concise", ref_answer="Four")
        assert request.principle == "Be concise"
        assert request.ref_answer == "Four"

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
        request = GenRMCompareVerifyRequest.model_validate(self._verify_request_payload())

        assert request.task_index is None
        assert request.rollout_index is None
        assert request.prompt_id is None

    def test_verify_request_reads_the_index_fields_that_gym_adds(self):
        # Gym's rollout collection stamps these keys on every row, and the agent forwards them to /verify.
        request = GenRMCompareVerifyRequest.model_validate(
            self._verify_request_payload(_ng_task_index=7, _ng_rollout_index=2, prompt_id="p1")
        )

        assert request.task_index == 7
        assert request.rollout_index == 2
        assert request.prompt_id == "p1"

    def test_verify_request_accepts_the_field_names_too(self):
        request = _make_verify_request("4", task_index=3, rollout_index=1)

        assert (request.task_index, request.rollout_index) == (3, 1)

    def test_verify_request_keeps_unknown_fields(self):
        request = GenRMCompareVerifyRequest.model_validate(self._verify_request_payload(some_other_field="kept"))

        assert request.model_extra["some_other_field"] == "kept"


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


class TestReferenceResponseObj:
    def test_the_reference_answer_reads_like_a_response(self):
        assert extract_output_text(_reference_response_obj("The reference.")) == "The reference."

    def test_an_empty_reference_answer_is_a_placeholder(self):
        assert extract_output_text(_reference_response_obj("")) == "None"


class TestResponseDigest:
    def test_equal_responses_have_equal_digests(self):
        assert _response_digest(_response_obj("a")) == _response_digest(_response_obj("a"))

    def test_different_responses_have_different_digests(self):
        assert _response_digest(_response_obj("a")) != _response_digest(_response_obj("b"))

    def test_the_digest_ignores_key_order(self):
        assert _response_digest({"a": 1, "b": 2}) == _response_digest({"b": 2, "a": 1})


class TestCohortKey:
    SERVER = _make_server()
    MESSAGES = [{"role": "user", "content": "What is 2+2?"}]

    def _key(self, body, principle=None, ref_answer=None):
        return self.SERVER._get_verify_cohort_key(body, self.MESSAGES, principle, ref_answer)

    def test_same_prompt_same_key(self):
        assert self._key(_make_verify_request("a")) == self._key(_make_verify_request("b"))

    def test_the_task_index_scopes_the_key(self):
        # The same prompt text in two different tasks must not end up in one cohort.
        one = self._key(_make_verify_request("a", task_index=1))
        two = self._key(_make_verify_request("a", task_index=2))

        assert one != two
        assert one.startswith("task_idx::1::")

    def test_the_prompt_id_scopes_the_key_when_there_is_no_task_index(self):
        key = self._key(_make_verify_request("a", prompt_id="p"))

        assert key.startswith("prompt_id::p::")
        assert key != self._key(_make_verify_request("a"))

    def test_the_task_index_takes_precedence_over_the_prompt_id(self):
        key = self._key(_make_verify_request("a", task_index=4, prompt_id="p"))

        assert key.startswith("task_idx::4::")

    def test_the_rollout_index_does_not_scope_the_key(self):
        # It identifies a rollout within its cohort, which is what the cohort is made of.
        body = _make_verify_request("a", task_index=1)
        other = _make_verify_request("a", task_index=1, rollout_index=5)

        assert self._key(body) == self._key(other)

    def test_the_principle_scopes_the_key(self):
        body = _make_verify_request("a")

        assert self._key(body, principle="x") != self._key(body, principle="y")

    def test_the_reference_answer_scopes_the_key(self):
        body = _make_verify_request("a")

        assert self._key(body, ref_answer="x") != self._key(body, ref_answer="y")
        assert self._key(body, ref_answer="x") != self._key(body)


class TestComparisonSlots:
    @staticmethod
    def _server(**config):
        return _make_server(_make_config(num_rollouts_per_prompt=4, **config))

    def test_circular_has_one_pair_per_rollout(self):
        slots = self._server()._comparison_slots(with_reference=False)

        assert slots == [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)]

    def test_all_pairs(self):
        slots = self._server(comparison_strategy="all_pairs")._comparison_slots(with_reference=False)

        assert len(slots) == 6

    def test_every_rollout_is_compared_with_the_reference(self):
        slots = self._server()._comparison_slots(with_reference=True)

        assert slots[-4:] == [(0, 4, 0), (1, 4, 0), (2, 4, 0), (3, 4, 0)]
        assert len(slots) == 8

    def test_one_slot_per_judge_pass(self):
        slots = self._server(num_judges_per_comparison=3)._comparison_slots(with_reference=True)

        assert len(slots) == 3 * 8
        assert Counter(judge_idx for _, _, judge_idx in slots) == {0: 8, 1: 8, 2: 8}


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


JUDGE_JSON = (
    '```json\n{"response_1_analysis": "Good.", "response_2_analysis": "Worse.", '
    '"score_1": 4, "score_2": 2, "ranking": 2}\n```'
)
FIRST_BETTER = (("1", math.log(0.75)), ("2", math.log(0.25)))


class TestRunSingleComparison:
    """`_run_single_comparison` talks to the GenRM twice per comparison: evaluate, then read the probability of 1."""

    @staticmethod
    def _make(evaluation=JUDGE_JSON, score_logprobs=FIRST_BETTER, **config):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                _http_response(_evaluation_payload(evaluation)),
                _http_response(_score_payload(*score_logprobs)),
            ]
        )
        return _make_server(_make_config(**config), client), client

    @staticmethod
    def _run(server, text_1="4", text_2="Four", conversation=None, principle=None, pair_idx=(0, 1)):
        conversation = conversation or [{"role": "user", "content": "What is 2+2?"}]
        return asyncio.run(
            server._run_single_comparison(
                conversation, _response_obj(text_1), _response_obj(text_2), pair_idx=pair_idx, principle=principle
            )
        )

    @staticmethod
    def _call(client, index):
        return client.post.call_args_list[index].kwargs

    def test_returns_parsed_scores_and_win_probability(self):
        server, _ = self._make()
        score_1, score_2, win_probability = self._run(server)

        assert score_1 == approx(4.0)
        assert score_2 == approx(2.0)
        assert win_probability == approx(0.75)

    def test_makes_two_calls_to_the_genrm_server(self):
        server, client = self._make()
        self._run(server)

        assert client.post.call_count == 2
        for index in (0, 1):
            assert self._call(client, index)["server_name"] == "dwrl_model"
            assert self._call(client, index)["url_path"] == "/v1/chat/completions"

    def test_evaluation_request_contains_context_both_responses_and_principle(self):
        server, client = self._make()
        self._run(
            server,
            text_1="The answer is 4.",
            text_2="It is five.",
            conversation=[{"role": "user", "content": "What is 2+2?"}],
            principle="Be accurate.",
        )

        body = self._call(client, 0)["json"]
        assert len(body.messages) == 1
        assert body.messages[0]["role"] == "user"
        assert body.messages[0]["content"] == (
            "CONTEXT:\nUser: What is 2+2?\nR1:\nThe answer is 4.\nR2:\nIt is five.\nPLAN:\nBe accurate."
        )

    def test_the_order_of_the_responses_is_kept(self):
        server, client = self._make()
        self._run(server, text_1="first", text_2="second")

        assert "R1:\nfirst\nR2:\nsecond" in self._call(client, 0)["json"].messages[0]["content"]

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

    def test_empty_responses_are_replaced_by_a_placeholder(self):
        server, client = self._make()
        self._run(server, text_1="", text_2="")

        assert "R1:\nNone\nR2:\nNone\nPLAN:" in self._call(client, 0)["json"].messages[0]["content"]

    def test_only_the_final_answers_are_judged(self):
        server, client = self._make()
        with_reasoning = {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hidden chain of thought"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Final answer."}]},
            ]
        }
        asyncio.run(
            server._run_single_comparison([{"role": "user", "content": "q"}], with_reasoning, _response_obj("Other."))
        )

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

    def test_win_probability_is_the_probability_of_the_score_token(self):
        server, _ = self._make(score_logprobs=(("2", math.log(0.9)), ("1", math.log(0.1))))
        _, _, win_probability = self._run(server)

        assert win_probability == approx(0.1)

    def test_score_token_is_matched_ignoring_whitespace(self):
        server, _ = self._make(score_logprobs=(("2", math.log(0.6)), (" 1", math.log(0.4))))
        _, _, win_probability = self._run(server)

        assert win_probability == approx(0.4)

    def test_exact_score_token_match_is_preferred(self):
        server, _ = self._make(score_logprobs=((" 1", math.log(0.3)), ("1", math.log(0.7))))
        _, _, win_probability = self._run(server)

        assert win_probability == approx(0.7)

    def test_custom_score_token(self):
        server, _ = self._make(score_token="A", score_logprobs=(("A", math.log(0.2)), ("B", math.log(0.8))))
        _, _, win_probability = self._run(server)

        assert win_probability == approx(0.2)

    def test_score_token_missing_from_top_logprobs_gives_zero(self):
        # A token outside the top 20 has a probability of at most 1/20, so 0 is the right order of magnitude.
        # It is deliberately not `default_ranking`, which is the neutral value for a failed GenRM call and would be
        # far too high here.
        server, _ = self._make(
            score_logprobs=(("2", math.log(0.99)), ("3", math.log(0.01))), default_ranking=0.3
        )
        score_1, score_2, win_probability = self._run(server)

        assert win_probability == 0.0
        # The evaluation itself is still parsed.
        assert (score_1, score_2) == approx((4.0, 2.0))

    def test_missing_score_token_logs_a_warning(self, caplog):
        # Falling back to 0.0 looks like a genuine low probability in the output, so it must be visible in the log.
        # For example, a `score_token` that does not match the model's tokens makes every comparison take this path.
        server, _ = self._make(score_token="One", score_logprobs=(("1", math.log(0.6)), ("2", math.log(0.4))))
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server, pair_idx=(2, 3))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "'One'" in message  # the configured score token
        assert "'1'" in message and "'2'" in message  # the tokens that were returned
        assert "(2, 3)" in message  # the pair
        assert "0.0" in message

    @pytest.mark.parametrize(
        "logprobs",
        [
            (("1", math.log(0.7)), ("2", math.log(0.3))),  # exact match
            (("2", math.log(0.3)), (" 1 ", math.log(0.7))),  # whitespace insensitive match
        ],
    )
    def test_found_score_token_does_not_log_a_warning(self, caplog, logprobs):
        server, _ = self._make(score_logprobs=logprobs)
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_empty_top_logprobs_gives_zero(self):
        server, _ = self._make(score_logprobs=())
        _, _, win_probability = self._run(server)

        assert win_probability == 0.0

    def test_a_listed_score_token_with_tiny_probability_is_still_used(self):
        # Present but unlikely is different from missing: the real (tiny) probability is returned.
        server, _ = self._make(score_logprobs=(("2", math.log(0.999)), ("1", -20.0)))
        _, _, win_probability = self._run(server)

        assert win_probability == approx(math.exp(-20.0))
        assert win_probability > 0.0

    def test_thinking_is_removed_before_the_judge_json_is_parsed(self):
        evaluation = '<think>draft: {"score_1": 1, "score_2": 1}</think>\n```json\n{"score_1": 4, "score_2": 3}\n```'
        server, _ = self._make(evaluation=evaluation)
        score_1, score_2, _ = self._run(server)

        assert (score_1, score_2) == approx((4.0, 3.0))

    def test_unterminated_thinking_prefix_is_removed(self):
        evaluation = 'draft {"score_1": 1}</think>```json\n{"score_1": 5, "score_2": 1}\n```'
        server, _ = self._make(evaluation=evaluation)
        score_1, score_2, _ = self._run(server)

        assert (score_1, score_2) == approx((5.0, 1.0))

    def test_unparseable_evaluation_uses_default_scores_but_keeps_the_win_probability(self):
        server, _ = self._make(evaluation="I could not decide.", default_score=-100.0)
        score_1, score_2, win_probability = self._run(server)

        assert (score_1, score_2) == approx((-100.0, -100.0))
        assert win_probability == approx(0.75)

    def test_unparseable_evaluation_is_not_retried_by_default(self):
        # The 1-5 score does not influence the reward, so unless retries are configured a retry would only cost
        # GenRM calls.
        server, client = self._make(evaluation="I could not decide.")
        self._run(server)

        assert server.config.genrm_parse_retries == 0
        assert client.post.call_count == 2

    def test_unparseable_evaluation_logs_the_parser_warning_when_there_are_no_retries(self, caplog):
        server, _ = self._make(evaluation="I could not decide.")
        with caplog.at_level(logging.WARNING):
            self._run(server)

        assert "No parseable JSON found in GenRM output" in caplog.text
        assert "retrying" not in caplog.text

    def test_server_error_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=RuntimeError("GenRM server is down"))
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, -100.0, 0.5)

    def test_malformed_server_response_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(return_value=_http_response({"unexpected": "shape"}))
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, -100.0, 0.5)

    def test_missing_logprobs_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                _http_response(_evaluation_payload(JUDGE_JSON)),
                _http_response({"choices": [{"message": {"content": "1"}}]}),
            ]
        )
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.5), client)

        assert self._run(server) == (-100.0, -100.0, 0.5)

    def test_a_failed_comparison_is_neutral_in_the_fit(self):
        # The default for a comparison that could not be judged must be usable by aggregate_scores as it is.
        client = MagicMock()
        client.post = AsyncMock(side_effect=RuntimeError("GenRM server is down"))
        server = _make_server(_make_config(), client)

        _, _, win_probability = self._run(server)

        assert win_probability == 0.5


class TestRunSingleComparisonWithParseRetries:
    """With `genrm_parse_retries > 0` an evaluation that cannot be parsed is repeated: both GenRM calls, from scratch.
    Every attempt has to be an independent evaluate-then-score exchange."""

    UNPARSEABLE = "cut off before any json"

    @staticmethod
    def _make(probabilities, evaluations, retries=1, **config):
        """The GenRM answers attempt k with evaluations[k] and a probability of "1" of probabilities[k]."""
        responses = []
        for evaluation, probability in zip(evaluations, probabilities):
            responses.append(_http_response(_evaluation_payload(evaluation)))
            responses.append(
                _http_response(_score_payload(("1", math.log(probability)), ("2", math.log(1 - probability))))
            )
        client = MagicMock()
        client.post = AsyncMock(side_effect=responses)
        config.setdefault("genrm_parse_retry_sleep_s", 0.0)
        config = _make_config(genrm_parse_retries=retries, **config)
        return _make_server(config, client), client

    @staticmethod
    def _run(server):
        return asyncio.run(
            server._run_single_comparison(
                [{"role": "user", "content": "q"}], _response_obj("a"), _response_obj("b"), pair_idx=(0, 1)
            )
        )

    def test_a_retry_uses_the_result_of_the_later_attempt(self):
        server, client = self._make([0.9, 0.2], [self.UNPARSEABLE, JUDGE_JSON])

        assert self._run(server) == (4.0, 2.0, approx(0.2))
        assert client.post.call_count == 4

    def test_no_retry_when_the_first_attempt_parses(self):
        server, client = self._make([0.9], [JUDGE_JSON], retries=3)

        assert self._run(server) == (4.0, 2.0, approx(0.9))
        assert client.post.call_count == 2

    def test_the_second_attempt_does_not_see_the_first_one(self):
        server, client = self._make([0.9, 0.2], [self.UNPARSEABLE, "second evaluation " + JUDGE_JSON])
        self._run(server)

        second_score_request = client.post.call_args_list[3].kwargs["json"]
        assert [m["role"] for m in second_score_request.messages] == ["user", "assistant", "user"]
        assert second_score_request.messages[1]["content"].startswith("second evaluation")
        assert second_score_request.messages[2]["content"] == BT_PROMPT

    def test_every_attempt_evaluates_the_same_prompt(self):
        server, client = self._make([0.9, 0.2], [self.UNPARSEABLE, JUDGE_JSON])
        self._run(server)

        first, second = (client.post.call_args_list[i].kwargs["json"].messages for i in (0, 2))
        assert first == second
        assert len(first) == 1

    def test_the_number_of_attempts_is_the_number_of_retries_plus_one(self):
        server, client = self._make([0.5] * 3, [self.UNPARSEABLE] * 3, retries=2)
        self._run(server)

        assert client.post.call_count == 2 * 3

    def test_giving_up_keeps_the_bt_score_of_the_last_attempt(self, caplog):
        server, client = self._make([0.9, 0.2], [self.UNPARSEABLE] * 2, default_score=-100.0)
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            result = self._run(server)

        assert result == (-100.0, -100.0, approx(0.2))
        assert client.post.call_count == 4
        assert "after 2 attempts; falling back to defaults" in caplog.text

    def test_every_retry_is_logged(self, caplog):
        server, _ = self._make([0.5] * 3, [self.UNPARSEABLE] * 3, retries=2)
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server)

        retries = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
        assert len(retries) == 2
        assert "attempt 1 of 3" in retries[0]
        assert "attempt 2 of 3" in retries[1]

    def test_a_successful_retry_is_logged_but_does_not_give_up(self, caplog):
        server, _ = self._make([0.9, 0.2], [self.UNPARSEABLE, JUDGE_JSON])
        with caplog.at_level(logging.WARNING, logger=app_module.logger.name):
            self._run(server)

        assert "retrying" in caplog.text
        assert "falling back to defaults" not in caplog.text

    @pytest.mark.parametrize("retries, raises", [(0, False), (1, True), (3, True)])
    def test_the_parser_raises_only_if_retries_are_configured(self, monkeypatch, retries, raises):
        seen = []
        real_parse = app_module.parse_genrm_output

        def spy(*args, **kwargs):
            seen.append(kwargs["raise_on_fail"])
            return real_parse(*args, **kwargs)

        monkeypatch.setattr(app_module, "parse_genrm_output", spy)
        server, _ = self._make([0.9] * (retries + 1), [self.UNPARSEABLE] * (retries + 1), retries=retries)
        self._run(server)

        assert seen == [raises] * (retries + 1)

    def test_the_sleep_between_retries_is_the_configured_one(self, monkeypatch):
        sleep = AsyncMock()
        monkeypatch.setattr(app_module.asyncio, "sleep", sleep)
        server, _ = self._make(
            [0.9, 0.2], [self.UNPARSEABLE, JUDGE_JSON], genrm_parse_retry_sleep_s=0.5
        )
        self._run(server)

        sleep.assert_awaited_once_with(0.5)


class FakeJudge:
    """Stands in for `_run_single_comparison`.

    Responses are told apart by their text: "rollout-<n>" is a rollout, anything starting with "ref" is the reference
    answer. Every call blocks until the test releases the gate of its pair, so the test decides in which order
    comparisons finish.
    """

    def __init__(self):
        self.gates = {}
        self.calls = []  # (id of response 1, id of response 2), in the order the calls started
        self.principles = []
        self.histories = []
        self.pair_indices = []
        self.errors = {}  # (id, id) -> exception to raise instead of returning a result
        self._nth = Counter()

    def gate(self, pair):
        return self.gates.setdefault(pair, asyncio.Event())

    def reset(self):
        self.gates.clear()
        self.calls.clear()
        self._nth.clear()

    async def __call__(
        self, server, conversation_history, response_obj_1, response_obj_2, pair_idx=(0, 0), principle=None
    ):
        pair = (_rid(response_obj_1), _rid(response_obj_2))
        nth = self._nth[pair]
        self._nth[pair] += 1
        self.calls.append(pair)
        self.principles.append(principle)
        self.histories.append(conversation_history)
        self.pair_indices.append(pair_idx)
        await self.gate(pair).wait()
        if pair in self.errors:
            raise self.errors[pair]
        return 3.0, 3.0, win_probability(*pair, nth)


def _rid(response_obj):
    text = extract_output_text(response_obj)
    if text.startswith("ref"):
        return "ref"
    return int(re.search(r"rollout-(\d+)", text).group(1))


def _skill(rid):
    return 1.0 if rid == "ref" else 0.35 * rid


def win_probability(a, b, nth=0):
    """What the fake judge says is the probability that `a` is better than `b`. `nth` is the judge pass."""
    p = 1.0 / (1.0 + math.exp(-(_skill(a) - _skill(b))))
    return min(max(p + 0.02 * nth, 0.01), 0.99)


@pytest.fixture
def judge(monkeypatch):
    fake = FakeJudge()

    # A plain function, so that it is bound to the server like the method it replaces.
    async def fake_run_single_comparison(
        server, conversation_history, response_obj_1, response_obj_2, pair_idx=(0, 0), principle=None
    ):
        return await fake(server, conversation_history, response_obj_1, response_obj_2, pair_idx, principle)

    monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
    return fake


REFERENCE = "ref answer"


def _text(rid, texts=None):
    if rid == "ref":
        return REFERENCE
    return (texts or {}).get(rid, f"rollout-{rid}")


def expected_rewards(
    arrival, strategy="circular", num_judges=1, with_reference=False, neutral_pairs=(), texts=None, **aggregate
):
    """The rewards that a cohort whose rollouts arrived in the order `arrival` (rollout ids) must get.

    This is aggregate_scores over every comparison, written in terms of arrival indices. `neutral_pairs` are the
    comparisons (as id pairs) whose result is the neutral default instead of the fake judge's.
    """
    n = len(arrival)
    pairs = generate_comparison_pairs(strategy, n)
    if with_reference:
        pairs = pairs + [(i, n) for i in range(n)]
    ids = list(arrival) + (["ref"] if with_reference else [])
    results, metadata = [], []
    for judge_idx in range(num_judges):
        for i, j in pairs:
            pair = (ids[i], ids[j])
            probability = 0.5 if pair in neutral_pairs else win_probability(*pair, judge_idx)
            results.append((3.0, 3.0, probability))
            metadata.append((i, j, judge_idx))
    response_objs = [_response_obj(_text(rid, texts)) for rid in ids]
    rewards, _, _, _ = aggregate_scores(
        comparison_results=results,
        comparison_metadata=metadata,
        response_objs=response_objs,
        aggregator_method="simple_tiebreaker",
        default_score=3.0,
        reasoning_bonus=0.0,
        answer_bonus=0.0,
        top_percentile=0.2,
        group_reasoning_length_penalty_coeff=0.0,
        group_answer_length_penalty_coeff=0.0,
        ref_answer_last=with_reference,
        **aggregate,
    )
    return rewards


async def _settle():
    """Give every runnable task enough event loop turns to run until it blocks again."""
    for _ in range(40):
        await asyncio.sleep(0)


async def _arrive(server, ids, prompt="What is 2+2?", principle=None, ref_answer=None, texts=None, **identity):
    """Start one verify() per rollout id, in order, and let each run until it blocks."""
    tasks = []
    for rid in ids:
        request = _make_verify_request(_text(rid, texts), prompt, principle, ref_answer, **identity)
        tasks.append(asyncio.create_task(server.verify(request)))
        await _settle()
    return tasks


async def _release(judge, pairs):
    """Let the judge calls for `pairs` finish, one pair at a time, in the given order."""
    for pair in pairs:
        judge.gate(pair).set()
        await _settle()


async def _release_all(judge):
    await _release(judge, list(judge.gates))


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=20))


def _server(size=4, num_judges=1, strategy="circular", **config):
    return _make_server(
        _make_config(
            num_rollouts_per_prompt=size,
            num_judges_per_comparison=num_judges,
            comparison_strategy=strategy,
            **config,
        )
    )


class TestVerify:
    def test_the_rewards_are_the_ratings_of_the_cohort(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)
        # The better rollouts (higher skill) are rewarded more.
        assert results[3].reward > results[0].reward

    def test_every_caller_gets_the_reward_of_its_own_rollout(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [2, 0, 3, 1])  # In an order that is not the order of the skills
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([2, 0, 3, 1]), abs=1e-6)
        # `results` is in arrival order: rollout 3 (arrival index 2) is the best one.
        assert max(range(4), key=lambda k: results[k].reward) == 2

    def test_response_carries_the_original_request(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3], prompt="Name a prime.")
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [extract_output_text(r.response.model_dump()) for r in results] == [f"rollout-{i}" for i in range(4)]
        assert all(r.responses_create_params.input[0].content == "Name a prime." for r in results)

    def test_nobody_is_rewarded_until_the_whole_cohort_has_been_judged(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            pairs = list(judge.gates)
            assert len(pairs) == 4
            await _release(judge, pairs[:-1])
            before = [t.done() for t in tasks]
            await _release(judge, pairs[-1:])
            return before, [t.done() for t in tasks]

        before, after = _run(scenario())

        assert before == [False] * 4
        assert after == [True] * 4

    def test_nobody_is_rewarded_until_the_whole_cohort_has_arrived(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2])  # The 4th rollout has not arrived
            await _release_all(judge)
            return [t.done() for t in tasks]

        assert _run(scenario()) == [False] * 3

    def test_a_comparison_is_judged_as_soon_as_both_its_responses_have_arrived(self, judge):
        async def scenario():
            server = _server()
            snapshots = []
            tasks = []
            for rid in range(4):
                tasks += await _arrive(server, [rid])
                snapshots.append(list(judge.calls))
            await _release_all(judge)
            await asyncio.gather(*tasks)
            return snapshots

        snapshots = _run(scenario())

        assert snapshots == [
            [],  # Rollout 0 has nobody to be compared with yet
            [(0, 1)],
            [(0, 1), (1, 2)],
            [(0, 1), (1, 2), (2, 3), (3, 0)],  # The last rollout closes the circle
        ]

    def test_the_comparison_that_closes_the_circle_has_the_last_arrival_first(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [5, 6, 7, 8])
            await _release_all(judge)
            await asyncio.gather(*tasks)

        _run(scenario())

        assert judge.calls[-1] == (8, 5)
        assert judge.pair_indices[-1] == (3, 0)  # By arrival index

    def test_every_comparison_is_judged_exactly_once(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release_all(judge)
            await asyncio.gather(*tasks)

        _run(scenario())

        assert Counter(judge.calls) == {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}

    def test_all_pairs_strategy(self, judge):
        async def scenario():
            tasks = await _arrive(_server(strategy="all_pairs"), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert len(judge.calls) == 6
        assert len(set(judge.calls)) == 6
        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3], strategy="all_pairs"), abs=1e-6)

    def test_several_judge_passes_are_averaged_in_the_fit(self, judge):
        async def scenario():
            tasks = await _arrive(_server(num_judges=3), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert Counter(judge.calls) == {(0, 1): 3, (1, 2): 3, (2, 3): 3, (3, 0): 3}
        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3], num_judges=3), abs=1e-6)

    def test_the_principle_and_the_conversation_are_passed_to_the_judge(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3], prompt="Name a prime.", principle="Be concise")
            await _release_all(judge)
            await asyncio.gather(*tasks)

        _run(scenario())

        assert set(judge.principles) == {"Be concise"}
        assert all(h == [{"role": "user", "content": "Name a prime."}] for h in judge.histories)

    def test_a_cohort_of_one_gets_the_default_score_without_judging(self, judge):
        async def scenario():
            server = _server(size=1, default_score=2.5)
            return await server.verify(_make_verify_request("rollout-0"))

        assert _run(scenario()).reward == approx(2.5)
        assert judge.calls == []

    def test_the_cohorts_are_gone_when_the_rewards_have_been_returned(self, judge, fresh_cohorts):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            assert fresh_cohorts.num_active == 1
            await _release_all(judge)
            await asyncio.gather(*tasks)

        _run(scenario())

        assert fresh_cohorts.num_active == 0
        assert fresh_cohorts.num_open == 0

    def test_a_rollout_that_never_arrives_leaves_the_others_waiting(self, judge):
        # Known limitation. Rewards depend on the whole cohort, so the cohort cannot be finished without it.
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 3])
            await _release_all(judge)
            return [t.done() for t in tasks]

        assert _run(scenario()) == [False] * 3


class TestVerifyCompletionOrder:
    def test_the_last_arrival_finishing_first_does_not_finish_the_cohort(self, judge):
        # The regression the cohort bookkeeping exists for: the request that arrived last finishes its comparisons
        # before an earlier request has finished its own, and must not decide that the cohort is done.
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release(judge, [(2, 3), (3, 0)])  # The last arrival's comparisons
            before = [t.done() for t in tasks]
            await _release(judge, [(0, 1), (1, 2)])
            return before, await asyncio.gather(*tasks)

        before, results = _run(scenario())

        assert before == [False] * 4
        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)

    def test_rewards_do_not_depend_on_the_order_in_which_comparisons_finish(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            pairs = list(judge.gates)
            random.Random(11).shuffle(pairs)
            await _release(judge, pairs)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)

    def test_rollouts_arriving_after_comparisons_have_finished(self, judge):
        async def scenario():
            server = _server()
            first = await _arrive(server, [0, 1])
            await _release(judge, [(0, 1)])  # Everything that can be judged so far is done
            second = await _arrive(server, [2, 3])
            await _release_all(judge)
            return await asyncio.gather(*first, *second)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)


class TestVerifyWithReferenceAnswer:
    def test_rewards_are_relative_to_the_reference(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3], ref_answer=REFERENCE)
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3], with_reference=True), abs=1e-6)
        assert len(results) == 4  # The reference is not a rollout and gets no reward.

    def test_every_rollout_is_compared_with_the_reference_as_response_2(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3], ref_answer=REFERENCE)
            await _release_all(judge)
            await asyncio.gather(*tasks)

        _run(scenario())

        assert Counter(judge.calls) == {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 1,
            (3, 0): 1,
            (0, "ref"): 1,
            (1, "ref"): 1,
            (2, "ref"): 1,
            (3, "ref"): 1,
        }
        assert judge.pair_indices[judge.calls.index((2, "ref"))] == (2, 4)

    def test_the_reference_comparison_of_a_rollout_is_judged_when_it_arrives(self, judge):
        # It does not wait for the rest of the cohort.
        async def scenario():
            server = _server()
            tasks = await _arrive(server, [0], ref_answer=REFERENCE)
            first = list(judge.calls)
            tasks += await _arrive(server, [1], ref_answer=REFERENCE)
            second = list(judge.calls)
            await _release_all(judge)
            tasks += await _arrive(server, [2, 3], ref_answer=REFERENCE)
            await _release_all(judge)
            await asyncio.gather(*tasks)
            return first, second

        first, second = _run(scenario())

        assert first == [(0, "ref")]
        assert second == [(0, "ref"), (0, 1), (1, "ref")]

    def test_the_reference_text_is_what_the_judge_sees(self, monkeypatch):
        texts = []

        async def fake_run_single_comparison(
            server, conversation_history, response_obj_1, response_obj_2, pair_idx=(0, 0), principle=None
        ):
            texts.append((extract_output_text(response_obj_1), extract_output_text(response_obj_2)))
            return 3.0, 3.0, 0.5

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)

        async def scenario():
            tasks = await _arrive(_server(size=2), [0, 1], ref_answer="the reference text")
            await asyncio.gather(*tasks)

        _run(scenario())

        assert ("rollout-0", "the reference text") in texts
        assert ("rollout-1", "the reference text") in texts

    def test_different_reference_answers_are_different_cohorts(self, judge):
        async def scenario():
            server = _server(size=2)
            tasks = await _arrive(server, [0], ref_answer="ref A")
            tasks += await _arrive(server, [1], ref_answer="ref B")
            await _release_all(judge)
            return [t.done() for t in tasks]

        # Neither cohort is full, although two rollouts of the same prompt arrived.
        assert _run(scenario()) == [False, False]

    def test_several_judge_passes_with_a_reference(self, judge):
        async def scenario():
            tasks = await _arrive(_server(num_judges=2), [0, 1, 2, 3], ref_answer=REFERENCE)
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert len(judge.calls) == 2 * 8
        assert [r.reward for r in results] == approx(
            expected_rewards([0, 1, 2, 3], num_judges=2, with_reference=True), abs=1e-6
        )


class TestVerifyGrouping:
    def test_prompts_do_not_mix(self, judge):
        async def scenario():
            server = _server()
            tasks_a, tasks_b = [], []
            for a, b in zip([0, 1, 2, 3], [10, 11, 12, 13]):  # Arrivals of two prompts interleave
                tasks_a += await _arrive(server, [a], prompt="Prompt A")
                tasks_b += await _arrive(server, [b], prompt="Prompt B")
            await _release_all(judge)
            return await asyncio.gather(*tasks_a), await asyncio.gather(*tasks_b)

        a, b = _run(scenario())

        assert [r.reward for r in a] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)
        assert [r.reward for r in b] == approx(expected_rewards([10, 11, 12, 13]), abs=1e-6)
        # A response was never compared with one of the other prompt.
        assert all((x < 10) == (y < 10) for x, y in judge.calls)

    def test_the_same_prompt_in_two_tasks_is_two_cohorts(self, judge):
        async def scenario():
            server = _server(size=2)
            tasks_1, tasks_2 = [], []
            for a, b in [(0, 10), (1, 11)]:
                tasks_1 += await _arrive(server, [a], task_index=1)
                tasks_2 += await _arrive(server, [b], task_index=2)
            await _release_all(judge)
            return await asyncio.gather(*tasks_1), await asyncio.gather(*tasks_2)

        first, second = _run(scenario())

        assert [r.reward for r in first] == approx(expected_rewards([0, 1]), abs=1e-6)
        assert [r.reward for r in second] == approx(expected_rewards([10, 11]), abs=1e-6)
        assert all((x < 10) == (y < 10) for x, y in judge.calls)

    def test_gym_index_fields_group_the_rollouts(self, judge):
        # The agent forwards `_ng_task_index` and `_ng_rollout_index`; the aliases must be what is grouped on.
        def request(text, task_index, rollout_index):
            base = _make_verify_request(text)
            return GenRMCompareVerifyRequest.model_validate(
                {**base.model_dump(), "_ng_task_index": task_index, "_ng_rollout_index": rollout_index}
            )

        async def scenario():
            server = _server(size=2)
            tasks = []
            for text, task_index, rollout_index in [
                ("rollout-0", 0, 0),
                ("rollout-10", 1, 0),
                ("rollout-1", 0, 1),
                ("rollout-11", 1, 1),
            ]:
                tasks.append(asyncio.create_task(server.verify(request(text, task_index, rollout_index))))
                await _settle()
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results[0::2]] == approx(expected_rewards([0, 1]), abs=1e-6)
        assert [r.reward for r in results[1::2]] == approx(expected_rewards([10, 11]), abs=1e-6)

    def test_the_same_prompt_arriving_twice_makes_two_cohorts_in_order(self, judge):
        # No index to tell the arrivals apart (e.g. the same prompt twice in a batch, or in a later epoch): the first
        # `size` arrivals are a cohort, and the next ones start another.
        async def scenario():
            tasks = await _arrive(_server(size=2), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        assert [r.reward for r in results[:2]] == approx(expected_rewards([0, 1]), abs=1e-6)
        assert [r.reward for r in results[2:]] == approx(expected_rewards([2, 3]), abs=1e-6)

    def test_a_bad_request_does_not_leave_a_member_behind(self, judge, fresh_cohorts):
        async def scenario():
            server = _server(strategy="round_robin")  # Not a strategy: building the comparisons fails
            with pytest.raises(ValueError, match="Unknown comparison strategy"):
                await server.verify(_make_verify_request("rollout-0"))

        _run(scenario())

        assert fresh_cohorts.num_active == 0
        assert judge.calls == []


class TestVerifyRepeatedRequests:
    """A request can be delivered twice. The repeat must not take a slot of the cohort or be judged again."""

    @staticmethod
    def _request(rid, rollout_index, **extra):
        return _make_verify_request(f"rollout-{rid}", task_index=0, rollout_index=rollout_index, **extra)

    def test_a_repeat_gets_the_same_reward_and_does_not_take_a_slot(self, judge):
        async def scenario():
            server = _server()
            tasks = []
            for rid, rollout_index in [(0, 0), (1, 1), (2, 2)]:
                tasks.append(asyncio.create_task(server.verify(self._request(rid, rollout_index))))
                await _settle()
            repeat = asyncio.create_task(server.verify(self._request(1, 1)))  # A repeat of the second rollout
            await _settle()
            calls_before_last_member = list(judge.calls)
            tasks.append(asyncio.create_task(server.verify(self._request(3, 3))))
            await _settle()
            await _release_all(judge)
            return calls_before_last_member, await asyncio.gather(*tasks, repeat)

        calls_before_last_member, results = _run(scenario())

        # The repeat did not count as a 4th member, so the real 4th member still completed the cohort.
        assert [r.reward for r in results[:4]] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)
        assert results[4].reward == results[1].reward
        # The repeat was not judged: three members have been judged, and nothing new started for the repeat.
        assert calls_before_last_member == [(0, 1), (1, 2)]
        assert Counter(judge.calls) == {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 0): 1}

    def test_a_repeat_after_the_cohort_has_filled_up_but_not_finished(self, judge):
        async def scenario():
            server = _server()
            tasks = []
            for rid in range(4):
                tasks.append(asyncio.create_task(server.verify(self._request(rid, rid))))
                await _settle()
            repeat = asyncio.create_task(server.verify(self._request(2, 2)))
            await _settle()
            await _release_all(judge)
            return await asyncio.gather(*tasks, repeat)

        results = _run(scenario())

        assert results[4].reward == results[2].reward
        assert len(judge.calls) == 4

    def test_a_different_rollout_with_the_same_index_is_not_a_repeat(self, judge):
        async def scenario():
            server = _server(size=2)
            first = asyncio.create_task(server.verify(self._request(0, 0)))
            await _settle()
            second = asyncio.create_task(server.verify(self._request(1, 0)))  # Same index, other response
            await _settle()
            await _release_all(judge)
            return await asyncio.gather(first, second)

        results = _run(scenario())

        assert [r.reward for r in results] == approx(expected_rewards([0, 1]), abs=1e-6)

    def test_cancelling_a_repeat_does_not_cancel_the_original(self, judge):
        async def scenario():
            server = _server()
            tasks = []
            for rid in range(4):
                tasks.append(asyncio.create_task(server.verify(self._request(rid, rid))))
                await _settle()
            repeat = asyncio.create_task(server.verify(self._request(1, 1)))
            await _settle()
            repeat.cancel()
            await _settle()
            await _release_all(judge)
            return repeat, await asyncio.gather(*tasks)

        repeat, results = _run(scenario())

        assert repeat.cancelled()
        assert [r.reward for r in results] == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)

    def test_a_repeat_of_a_request_that_failed_fails_too(self, judge):
        judge.errors[(1, 2)] = RuntimeError("judge exploded")  # Judged by rollout 2

        async def scenario():
            server = _server()
            tasks = []
            for rid in range(4):
                tasks.append(asyncio.create_task(server.verify(self._request(rid, rid))))
                await _settle()
            repeat = asyncio.create_task(server.verify(self._request(2, 2)))
            await _settle()
            await _release_all(judge)
            return await asyncio.gather(*tasks, repeat, return_exceptions=True)

        results = _run(scenario())

        assert isinstance(results[2], RuntimeError)
        # The repeat is not given the reward that was computed with a neutral stand-in for the failed comparison.
        assert isinstance(results[4], MemberJudgmentFailed)


class TestVerifyFailures:
    def test_cancelling_a_request_does_not_leave_the_cohort_waiting(self, judge):
        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            tasks[2].cancel()  # Judges the comparison of 1 against 2
            await _settle()
            others = [0, 1, 3]
            await _release_all(judge)
            return tasks[2], await asyncio.gather(*[tasks[i] for i in others])

        cancelled, results = _run(scenario())

        assert cancelled.cancelled()
        expected = expected_rewards([0, 1, 2, 3], neutral_pairs={(1, 2)})
        assert [r.reward for r in results] == approx([expected[0], expected[1], expected[3]], abs=1e-6)

    def test_a_judge_error_fails_that_request_and_neutralizes_its_comparisons(self, judge):
        judge.errors[(1, 2)] = RuntimeError("judge exploded")

        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())

        assert isinstance(results[2], RuntimeError)
        expected = expected_rewards([0, 1, 2, 3], neutral_pairs={(1, 2)})
        for i in (0, 1, 3):
            assert results[i].reward == approx(expected[i], abs=1e-6)

    def test_all_comparisons_of_a_failed_request_are_neutralized(self, judge):
        # Rollout 3 judges two comparisons. When it fails, both are replaced by the neutral result.
        judge.errors[(3, 0)] = RuntimeError("judge exploded")

        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())

        assert isinstance(results[3], RuntimeError)
        expected = expected_rewards([0, 1, 2, 3], neutral_pairs={(2, 3), (3, 0)})
        for i in (0, 1, 2):
            assert results[i].reward == approx(expected[i], abs=1e-6)

    def test_a_failure_in_the_aggregation_fails_every_member_instead_of_hanging(
        self, judge, monkeypatch, fresh_cohorts
    ):
        def broken_aggregate(**kwargs):
            raise AssertionError("ELO_VIOLATION! The fit diverged")

        monkeypatch.setattr(app_module, "aggregate_scores", broken_aggregate)

        async def scenario():
            tasks = await _arrive(_server(), [0, 1, 2, 3])
            await _release_all(judge)
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())

        assert all(isinstance(r, AssertionError) for r in results)
        assert fresh_cohorts.num_active == 0


class TestVerifyStylePenalty:
    FANCY = "# Heading\n- **bold point**\n- **another**\nrollout-2"

    def test_a_formatted_rollout_is_rewarded_less(self, judge):
        texts = {2: self.FANCY}

        async def scenario(coeff):
            judge.reset()
            server = _server(group_style_penalty_coeff=coeff)
            tasks = await _arrive(server, [0, 1, 2, 3], texts=texts)
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        plain = [r.reward for r in _run(scenario(0.0))]
        penalized = [r.reward for r in _run(scenario(0.5))]

        assert plain == approx(expected_rewards([0, 1, 2, 3], texts=texts), abs=1e-6)
        assert penalized == approx(
            expected_rewards([0, 1, 2, 3], texts=texts, group_style_penalty_coeff=0.5), abs=1e-6
        )
        assert penalized[2] < plain[2]
        assert penalized[0] > plain[0]

    def test_the_penalty_applies_to_the_rollouts_only_when_there_is_a_reference(self, judge):
        texts = {2: self.FANCY}

        async def scenario():
            server = _server(group_style_penalty_coeff=0.5)
            tasks = await _arrive(server, [0, 1, 2, 3], ref_answer=REFERENCE, texts=texts)
            await _release_all(judge)
            return await asyncio.gather(*tasks)

        results = _run(scenario())

        expected = expected_rewards([0, 1, 2, 3], with_reference=True, texts=texts, group_style_penalty_coeff=0.5)
        assert [r.reward for r in results] == approx(expected, abs=1e-6)


class TestVerifyRandomizedSchedules:
    """Random cohort sizes, strategies, judge counts, reference answers, arrival orders and finish orders, with
    arrivals and finishes mixed: each rollout is rewarded with the rating that aggregate_scores gives it over all
    the comparisons of its cohort, whatever the schedule."""

    NUM_SEEDS = 40

    def test_random_schedules(self, monkeypatch):
        for seed in range(self.NUM_SEEDS):
            self._run_seed(seed, monkeypatch)

    def _run_seed(self, seed, monkeypatch):
        rng = random.Random(seed)
        size = rng.randint(2, 6)
        strategy = rng.choice(["circular", "all_pairs"])
        num_judges = rng.randint(1, 3)
        with_reference = rng.random() < 0.5
        arrival = rng.sample(range(20), size)

        fake = FakeJudge()

        async def fake_run_single_comparison(
            server, conversation_history, response_obj_1, response_obj_2, pair_idx=(0, 0), principle=None
        ):
            return await fake(server, conversation_history, response_obj_1, response_obj_2, pair_idx, principle)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
        monkeypatch.setattr(app_module, "_cohorts", CohortRegistry())

        async def scenario():
            server = _server(size=size, num_judges=num_judges, strategy=strategy)
            ref_answer = REFERENCE if with_reference else None
            tasks = []
            to_arrive = list(arrival)
            released = set()
            while True:
                pending = [pair for pair in fake.gates if pair not in released]
                if not to_arrive and not pending:
                    break
                if to_arrive and (not pending or rng.random() < 0.5):
                    tasks += await _arrive(server, [to_arrive.pop(0)], ref_answer=ref_answer)
                else:
                    pair = rng.choice(pending)
                    released.add(pair)
                    await _release(fake, [pair])
            await _settle()
            return await asyncio.gather(*tasks)

        results = asyncio.run(asyncio.wait_for(scenario(), timeout=30))

        expected = expected_rewards(arrival, strategy, num_judges, with_reference)
        assert [r.reward for r in results] == approx(expected, abs=1e-6), seed
        calls = Counter(fake.calls)
        assert set(calls.values()) == {num_judges}, seed  # Every comparison was judged, once per judge pass
        assert app_module._cohorts.num_active == 0, seed


class TestRunCompare:
    """The batch /compare path, which does not use cohorts."""

    def test_rewards_are_the_ratings_over_all_the_comparisons(self, judge):
        async def scenario():
            server = _server(size=1)
            task = asyncio.create_task(
                server._run_compare(
                    [{"role": "user", "content": "q"}], [_response_obj(f"rollout-{i}") for i in range(4)]
                )
            )
            await _settle()
            await _release_all(judge)
            return await task

        rewards, metrics, results, metadata = _run(scenario())

        assert rewards == approx(expected_rewards([0, 1, 2, 3]), abs=1e-6)
        assert metadata == [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 0, 0)]
        assert len(results) == 4
        assert metrics["mean_individual_score"] == approx(3.0)

    def test_fewer_than_two_responses_get_the_default_score(self):
        server = _make_server(_make_config(default_score=2.0))

        rewards, metrics, results, metadata = asyncio.run(server._run_compare([], [_response_obj("a")]))

        assert (rewards, metrics, results, metadata) == ([2.0], {}, [], [])

    def test_the_compare_endpoint_reports_the_detail_of_every_comparison(self, judge):
        async def scenario():
            server = _server(size=1)
            request = GenRMCompareRequest(
                conversation_history=[{"role": "user", "content": "q"}],
                response_objs=[_response_obj(f"rollout-{i}") for i in range(3)],
            )
            task = asyncio.create_task(server.compare(request))
            await _settle()
            await _release_all(judge)
            return await task

        response = _run(scenario())

        assert response.rewards == approx(expected_rewards([0, 1, 2]), abs=1e-6)
        assert [(d["response_i"], d["response_j"], d["judge_idx"]) for d in response.comparison_results] == [
            (0, 1, 0),
            (1, 2, 0),
            (2, 0, 0),
        ]
        assert response.comparison_results[0]["ranking"] == approx(win_probability(0, 1))
        assert response.metrics["mean_individual_score"] == approx(3.0)

    def test_the_compare_endpoint_with_a_single_response(self):
        server = _make_server(_make_config(default_score=2.0))
        request = GenRMCompareRequest(
            conversation_history=[{"role": "user", "content": "q"}], response_objs=[_response_obj("a")]
        )

        response = asyncio.run(server.compare(request))

        assert response.rewards == [2.0]
        assert response.comparison_results is None
