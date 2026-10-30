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
- Every rollout is judged on its own (see `_run_single_comparison`): the judge first writes an evaluation of the
  response, then a second 1-token call reads the probability of `score_token` ("yes") from the logprobs. That
  probability is the BT score, and the mean over the judge passes is the rollout's reward.
- verify() returns that reward as soon as the rollout has been judged. It never waits for, or depends on, any
  other rollout of the same prompt.
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
    GenRMCompareResourcesServer,
    GenRMCompareVerifyRequest,
    _input_to_conversation_history,
    convert_to_chat_completion_format,
    flatten_to_single_turn,
)
from resources_servers.dwrl_genrm_compare_pointwise.utils import extract_output_text


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


def _make_verify_request(answer: str, prompt: str = "What is 2+2?", principle=None) -> GenRMCompareVerifyRequest:
    return GenRMCompareVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[NeMoGymEasyInputMessage(role="user", content=prompt, type="message")]
        ),
        response=_make_response(answer),
        principle=principle,
    )


def _response_obj(text: str) -> dict:
    return _make_response(text).model_dump()


class TestGenRMCompareConfig:
    def test_defaults(self):
        config = _make_config()

        assert config.num_judges_per_comparison == 1
        assert config.use_principle is False
        assert config.aggregator_method == "simple_tiebreaker"
        assert config.default_score == 3.0
        assert config.default_ranking == 0.0
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

    def test_several_worker_processes_are_accepted(self):
        # Rollouts are judged independently, so they need not be handled by the same process.
        assert _make_config(num_workers=4).num_workers == 4

    def test_packaged_config_matches_the_config_class(self):
        """configs/dwrl_genrm_compare_pointwise.yaml must stay in sync with GenRMCompareConfig."""
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        server_config = raw["dwrl_pointwise_compare_resources_server"]["resources_servers"][
            "dwrl_genrm_compare_pointwise"
        ]

        # Nothing has to be supplied when the run is launched, and rollouts are not grouped by prompt.
        assert "???" not in yaml.safe_dump(server_config)
        assert "num_rollouts_per_prompt" not in server_config

        config = GenRMCompareConfig(host="localhost", port=8000, name="dwrl", **server_config)

        assert config.genrm_model_server.name == "dwrl_model"
        assert config.score_token == "yes"
        assert config.aggregator_method == "simple_tiebreaker"
        # A failed GenRM call must not be worth more than a real judgment.
        assert "default_ranking" in server_config
        assert config.default_ranking == 0.0
        # The prompts must contain the placeholders that _run_single_comparison fills in.
        for placeholder in ("{context}", "{response}", "{principle}"):
            assert placeholder in config.dwrl_prompt
        # Literal braces in the JSON example are escaped so that str.format works.
        assert config.dwrl_prompt.strip().format(context="c", response="r", principle="p")


class TestRequestAndResponseModels:
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

    def test_verify_request_keeps_the_index_fields_that_gym_adds_as_extras(self):
        # Gym's rollout collection stamps these keys on every row, and the agent forwards them to /verify.
        request = GenRMCompareVerifyRequest.model_validate(
            self._verify_request_payload(_ng_task_index=7, _ng_rollout_index=2)
        )

        assert request.model_extra == {"_ng_task_index": 7, "_ng_rollout_index": 2}

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
        server = _make_server(_make_config(default_score=-100.0), client)

        assert self._run(server) == (-100.0, 0.0)

    def test_the_result_of_a_failed_call_is_the_configured_default_ranking(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=RuntimeError("GenRM server is down"))
        server = _make_server(_make_config(default_score=-100.0, default_ranking=0.2), client)

        assert self._run(server) == (-100.0, 0.2)

    def test_malformed_server_response_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(return_value=_http_response({"unexpected": "shape"}))
        server = _make_server(_make_config(default_score=-100.0), client)

        assert self._run(server) == (-100.0, 0.0)

    def test_missing_logprobs_returns_default_results(self):
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                _http_response(_evaluation_payload(JUDGE_JSON)),
                _http_response({"choices": [{"message": {"content": "yes"}}]}),
            ]
        )
        server = _make_server(_make_config(default_score=-100.0), client)

        assert self._run(server) == (-100.0, 0.0)


class TestRunSingleComparisonWithParseRetries:
    """With `genrm_parse_retries > 0` an evaluation that cannot be parsed is repeated: both GenRM calls, from scratch.
    Every attempt has to be an independent evaluate-then-score exchange."""

    UNPARSEABLE = "cut off before any json"

    @staticmethod
    def _make(probabilities, evaluations, retries=1, **config):
        """The GenRM answers attempt k with evaluations[k] and a probability of "yes" of probabilities[k]."""
        responses = []
        for evaluation, probability in zip(evaluations, probabilities):
            responses.append(_http_response(_evaluation_payload(evaluation)))
            responses.append(
                _http_response(_score_payload(("yes", math.log(probability)), ("no", math.log(1 - probability))))
            )
        client = MagicMock()
        client.post = AsyncMock(side_effect=responses)
        config.setdefault("genrm_parse_retry_sleep_s", 0.0)
        config = _make_config(genrm_parse_retries=retries, **config)
        return _make_server(config, client), client

    @staticmethod
    def _run(server):
        return asyncio.run(server._run_single_comparison([{"role": "user", "content": "q"}], _response_obj("a")))

    def test_a_retry_uses_the_result_of_the_later_attempt(self):
        server, client = self._make([0.9, 0.2], [self.UNPARSEABLE, JUDGE_JSON])

        assert self._run(server) == (4.0, approx(0.2))
        assert client.post.call_count == 4

    def test_no_retry_when_the_first_attempt_parses(self):
        server, client = self._make([0.9], [JUDGE_JSON], retries=3)

        assert self._run(server) == (4.0, approx(0.9))
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

        assert result == (-100.0, approx(0.2))
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
        self.principles = []
        self.errors = {}  # rollout id -> exception to raise instead of returning a score

    def gate(self, rollout_id):
        return self.gates.setdefault(rollout_id, asyncio.Event())

    async def __call__(self, server, conversation_history, response_obj, principle=None):
        rollout_id = int(extract_output_text(response_obj).rsplit("-", 1)[1])
        self.calls.append(rollout_id)
        self.principles.append(principle)
        await self.gate(rollout_id).wait()
        if rollout_id in self.errors:
            raise self.errors[rollout_id]
        return 3.0, BT_SCORES[rollout_id]


@pytest.fixture
def judge(monkeypatch):
    fake = FakeJudge()

    # A plain function, so that it is bound to the server like the method it replaces.
    async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
        return await fake(server, conversation_history, response_obj, principle)

    monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)
    return fake


async def _settle():
    """Give every runnable task enough event loop turns to run until it blocks again."""
    for _ in range(25):
        await asyncio.sleep(0)


async def _start(server, ids, prompt="What is 2+2?", principle=None):
    """Start one verify() per id, in order, and let each reach its judge call."""
    tasks = [
        asyncio.create_task(server.verify(_make_verify_request(f"rollout-{i}", prompt, principle))) for i in ids
    ]
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
    return _make_server(_make_config(num_judges_per_comparison=num_judges))


class TestVerify:
    def test_reward_is_the_bt_score_of_the_rollout(self, judge):
        async def scenario():
            task = (await _start(_server(), [3]))[0]
            await _release(judge, [3])
            return await task

        assert _run(scenario()).reward == approx(BT_SCORES[3])
        assert judge.calls == [3]

    def test_a_lone_rollout_is_judged(self, judge):
        # There is no group to wait for, so one rollout per prompt is judged like any other.
        async def scenario():
            server = _make_server(_make_config(default_score=-100.0))
            task = (await _start(server, [0]))[0]
            await _release(judge, [0])
            return await task

        assert _run(scenario()).reward == approx(BT_SCORES[0])

    def test_response_carries_the_original_request(self, judge):
        async def scenario():
            task = (await _start(_server(), [2]))[0]
            await _release(judge, [2])
            return await task

        result = _run(scenario())
        assert extract_output_text(result.response.model_dump()) == "rollout-2"
        assert result.responses_create_params.input[0].content == "What is 2+2?"

    def test_a_rollout_is_rewarded_without_waiting_for_the_others(self, judge):
        # The point of the design: of 8 rollouts of one prompt only rollout 3 has been judged.
        async def scenario():
            tasks = await _start(_server(), range(NUM_ROLLOUTS))
            await _release(judge, [3])
            return [t.done() for t in tasks], tasks[3].result().reward

        done, reward = _run(scenario())
        assert done == [i == 3 for i in range(NUM_ROLLOUTS)]
        assert reward == approx(BT_SCORES[3])

    def test_a_rollout_that_never_arrives_does_not_block_the_others(self, judge):
        async def scenario():
            ids = [i for i in range(NUM_ROLLOUTS) if i != 5]
            tasks = await _start(_server(), ids)
            await _release(judge, ids)
            return ids, await asyncio.gather(*tasks)

        ids, results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in ids])

    def test_any_completion_order_gives_every_rollout_its_own_reward(self, judge):
        async def scenario():
            tasks = await _start(_server(), range(NUM_ROLLOUTS))
            order = list(range(NUM_ROLLOUTS))
            random.Random(7).shuffle(order)
            await _release(judge, order)
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in range(NUM_ROLLOUTS)])

    def test_rollouts_arriving_after_others_were_already_rewarded(self, judge):
        async def scenario():
            server = _server()
            early = await _start(server, [0, 1])
            await _release(judge, [0, 1])
            early_rewards = [t.result().reward for t in early]
            late = await _start(server, [2, 3])
            await _release(judge, [2, 3])
            return early_rewards, [t.result().reward for t in late]

        early, late = _run(scenario())
        assert early == approx([BT_SCORES[0], BT_SCORES[1]])
        assert late == approx([BT_SCORES[2], BT_SCORES[3]])

    def test_the_judge_is_called_once_per_rollout_and_judge_pass(self, judge):
        async def scenario():
            tasks = await _start(_server(num_judges=3), range(2))
            await _release(judge, range(2))
            return await asyncio.gather(*tasks)

        _run(scenario())
        assert Counter(judge.calls) == {0: 3, 1: 3}

    def test_multiple_judges_are_averaged(self, monkeypatch):
        scores = iter([0.2, 0.5, 0.8])

        async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
            return 3.0, next(scores)

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)

        result = _run(_server(num_judges=3).verify(_make_verify_request("rollout-0")))

        assert result.reward == approx(0.5)

    def test_the_principle_is_passed_to_the_judge(self, judge):
        async def scenario():
            tasks = await _start(_server(), [0, 1], principle="Be concise")
            await _release(judge, [0, 1])
            return await asyncio.gather(*tasks)

        _run(scenario())
        assert judge.principles == ["Be concise", "Be concise"]

    def test_gym_index_fields_do_not_change_the_reward(self, judge):
        # Gym stamps `_ng_task_index` and `_ng_rollout_index` on every request. Rollouts are not grouped by them.
        async def scenario():
            server = _server()
            requests = [
                GenRMCompareVerifyRequest.model_validate(
                    {
                        **_make_verify_request(f"rollout-{i}").model_dump(),
                        "_ng_task_index": task_index,
                        "_ng_rollout_index": i,
                    }
                )
                for i, task_index in [(0, 0), (1, 0), (2, 1)]
            ]
            tasks = [asyncio.create_task(server.verify(r)) for r in requests]
            await _settle()
            await _release(judge, [2, 0, 1])
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[0], BT_SCORES[1], BT_SCORES[2]])

    def test_a_repeated_request_is_simply_judged_again(self, judge):
        # Nothing is shared between requests, so a request delivered twice costs a second judgment and nothing else.
        async def scenario():
            server = _server()
            tasks = await _start(server, [4, 4])
            await _release(judge, [4])
            return await asyncio.gather(*tasks)

        results = _run(scenario())
        assert [r.reward for r in results] == approx([BT_SCORES[4], BT_SCORES[4]])
        assert judge.calls == [4, 4]

class TestVerifyFailures:
    def test_cancelling_a_request_does_not_affect_the_others(self, judge):
        async def scenario():
            tasks = await _start(_server(), range(NUM_ROLLOUTS))
            tasks[2].cancel()
            await _settle()
            others = [i for i in range(NUM_ROLLOUTS) if i != 2]
            await _release(judge, others)
            results = await asyncio.gather(*[tasks[i] for i in others])
            return tasks[2].cancelled(), others, results

        cancelled, others, results = _run(scenario())
        assert cancelled
        assert [r.reward for r in results] == approx([BT_SCORES[i] for i in others])

    def test_a_judge_error_fails_that_request_only(self, judge):
        judge.errors[2] = RuntimeError("judge exploded")

        async def scenario():
            tasks = await _start(_server(), range(NUM_ROLLOUTS))
            await _release(judge, range(NUM_ROLLOUTS))
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = _run(scenario())
        assert isinstance(results[2], RuntimeError)
        for i in range(NUM_ROLLOUTS):
            if i != 2:
                assert results[i].reward == approx(BT_SCORES[i])


class TestVerifyGenRMFailure:
    def test_a_rollout_whose_genrm_call_failed_gets_a_reward_of_zero(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=RuntimeError("GenRM server is down"))
        server = _make_server(_make_config(), client)

        result = _run(server.verify(_make_verify_request("4")))

        assert result.reward == 0.0

    def test_every_pass_of_a_failed_call_counts_in_the_mean(self):
        # One pass fails (0.0) and the other judges the rollout 0.8: the mean is 0.4.
        client = MagicMock()
        client.post = AsyncMock(
            side_effect=[
                RuntimeError("GenRM server is down"),
                _http_response(_evaluation_payload(JUDGE_JSON)),
                _http_response(_score_payload(("yes", math.log(0.8)), ("no", math.log(0.2)))),
            ]
        )
        server = _make_server(_make_config(num_judges_per_comparison=2), client)

        result = _run(server.verify(_make_verify_request("4")))

        assert result.reward == approx(0.4)


class TestVerifyRandomizedSchedules:
    """Random arrival orders, finish orders, judge counts, cancellations and errors: each rollout is rewarded with the
    mean BT score of its own judgments, or fails, or is cancelled, and never depends on anything else."""

    NUM_SEEDS = 40

    def test_random_schedules(self, monkeypatch):
        for seed in range(self.NUM_SEEDS):
            self._run_seed(seed, monkeypatch)

    def _run_seed(self, seed, monkeypatch):
        rng = random.Random(seed)
        num_judges = rng.randint(1, 3)
        ids = list(range(NUM_ROLLOUTS))
        rng.shuffle(ids)  # Arrival order
        ids = ids[: rng.randint(1, NUM_ROLLOUTS)]  # Some rollouts never arrive
        failing = {i for i in ids if rng.random() < 0.2}
        cancelled = {i for i in ids if i not in failing and rng.random() < 0.2}

        # Every judge pass of a rollout gets its own score, so the mean over passes is what the reward must be.
        scores = {i: [rng.random() for _ in range(num_judges)] for i in ids}
        next_pass = {i: 0 for i in ids}
        gates = {i: asyncio.Event() for i in ids}

        async def fake_run_single_comparison(server, conversation_history, response_obj, principle=None):
            rollout_id = int(extract_output_text(response_obj).rsplit("-", 1)[1])
            pass_idx = next_pass[rollout_id]
            next_pass[rollout_id] += 1
            await gates[rollout_id].wait()
            if rollout_id in failing:
                raise RuntimeError(f"judge failed for {rollout_id}")
            return 3.0, scores[rollout_id][pass_idx]

        monkeypatch.setattr(GenRMCompareResourcesServer, "_run_single_comparison", fake_run_single_comparison)

        async def scenario():
            server = _server(num_judges)
            tasks = {}
            for i in ids:
                tasks[i] = asyncio.create_task(server.verify(_make_verify_request(f"rollout-{i}")))
                await _settle()
            for i in cancelled:
                tasks[i].cancel()
            await _settle()
            finish_order = list(ids)
            rng.shuffle(finish_order)
            for i in finish_order:
                gates[i].set()
                await _settle()
            return await asyncio.gather(*tasks.values(), return_exceptions=True)

        results = dict(zip(ids, _run(scenario())))
        for i in ids:
            if i in cancelled:
                assert isinstance(results[i], asyncio.CancelledError), (seed, i)
            elif i in failing:
                assert isinstance(results[i], RuntimeError), (seed, i)
            else:
                assert results[i].reward == approx(sum(scores[i]) / num_judges), (seed, i)
