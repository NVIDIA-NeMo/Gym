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
"""OpenCode compaction-summary recovery: a bounded, checked summary call, selected by the harness's agent header."""

import copy
import json
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from responses_api_models.vllm_model.app import (
    OPENCODE_AGENT_HEADER,
    VLLMModel,
    VLLMModelConfig,
    _bound_output_budget,
)


HEADINGS = [
    "Goal",
    "Constraints & Preferences",
    "Progress",
    "Key Decisions",
    "Next Steps",
    "Critical Context",
    "Relevant Files",
]
GOOD = {
    "choices": [
        {"finish_reason": "stop", "message": {"content": "\n".join(f"## {heading}\nFacts." for heading in HEADINGS)}}
    ]
}
# A model that kept reasoning instead of summarizing: no visible text.
BAD = {"choices": [{"finish_reason": "stop", "message": {"content": "", "reasoning_content": "Continue the task."}}]}
# An ordinary build-agent reply, which the ordinary path serves as it is.
PLAIN = {"choices": [{"finish_reason": "stop", "message": {"content": "I will continue."}}]}


def _model(**overrides) -> VLLMModel:
    config = VLLMModelConfig(
        host="localhost",
        port=8080,
        entrypoint="",
        name="test",
        model="test",
        base_url="http://worker/v1",
        api_key="unused",  # pragma: allowlist secret
        return_token_id_information=False,
        uses_reasoning_parser=False,
        **overrides,
    )
    return VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


class Worker:
    """An engine that answers the scripted replies in order and records what it was asked."""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls: list[dict] = []

    async def create_chat_completion(self, **body):
        self.calls.append(copy.deepcopy(body))
        reply = copy.deepcopy(next(self.replies))
        reply.update(id="summary-test", object="chat.completion", created=0, model="test")
        reply["choices"][0].update(index=0)
        reply["choices"][0]["message"]["role"] = "assistant"
        return reply


@pytest.mark.asyncio
async def test_a_retry_quotes_the_intact_history_and_keeps_the_transport_options() -> None:
    model = _model(compaction_summary_max_tokens=8192)
    worker = Worker([BAD, GOOD])
    messages = [{"role": "user", "content": "Original task"}, {"role": "tool", "content": "Actual tool result"}]
    body = {"messages": copy.deepcopy(messages), "_resend_after_disconnect": False}

    result = await model._create_compaction_summary(lambda payload: worker.create_chat_completion(**payload), body)

    assert result["choices"][0]["message"]["content"] == GOOD["choices"][0]["message"]["content"]
    assert len(worker.calls) == 2
    assert worker.calls[0]["messages"] == messages, "the first attempt is OpenCode's own request"
    retry = worker.calls[1]
    record = (
        retry["messages"][1]["content"].split("<conversation_record>\n", 1)[1].rsplit("\n</conversation_record>", 1)[0]
    )
    assert json.loads(record) == messages, "the retry quotes the whole original history as data"
    assert "## Goal" in retry["messages"][0]["content"] and "## Relevant Files" in retry["messages"][0]["content"]
    assert retry["_resend_after_disconnect"] is False


@pytest.mark.asyncio
async def test_unusable_summaries_fail_with_422_after_the_configured_attempts() -> None:
    model = _model(compaction_summary_max_tokens=8192, compaction_summary_max_attempts=3)
    worker = Worker([BAD, BAD, BAD])
    with pytest.raises(HTTPException) as caught:
        await model._create_compaction_summary(
            lambda payload: worker.create_chat_completion(**payload),
            {"messages": [{"role": "user", "content": "task"}]},
        )
    assert caught.value.status_code == 422
    assert caught.value.detail["error"]["code"] == "invalid_compaction_summary"
    assert len(worker.calls) == 3
    assert worker.calls[1]["messages"] != worker.calls[2]["messages"], "each retry names the previous failure"


@pytest.mark.parametrize(
    "choice, problem",
    [
        (BAD["choices"][0], "no visible text"),
        ({"finish_reason": "length", "message": GOOD["choices"][0]["message"]}, "finish_reason='length'"),
        ({"finish_reason": "stop", "message": {"content": "I will continue."}}, "sections []"),
        ({"finish_reason": "stop", "message": {"content": "<think>still thinking about the"}}, "no visible text"),
        (
            {"finish_reason": "stop", "message": {"content": "\n".join(f"## {h}\n" for h in HEADINGS)}},
            "an empty section",
        ),
    ],
)
def test_an_unusable_summary_is_refused_with_its_reason(choice, problem) -> None:
    with pytest.raises(HTTPException) as caught:
        _model(compaction_summary_max_tokens=8192)._check_compaction_summary(choice)
    assert problem in caught.value.detail["error"]["message"]


def test_a_summary_with_inline_thinking_before_the_sections_is_accepted() -> None:
    content = "<think>plan the summary</think>\n" + GOOD["choices"][0]["message"]["content"]
    _model(compaction_summary_max_tokens=8192)._check_compaction_summary(
        {"finish_reason": "stop", "message": {"content": content}}
    )


def test_without_expected_sections_only_finished_visible_text_is_required() -> None:
    model = _model(compaction_summary_max_tokens=8192, compaction_summary_sections=[])
    model._check_compaction_summary({"finish_reason": "stop", "message": {"content": "A plain summary."}})
    with pytest.raises(HTTPException):
        model._check_compaction_summary({"finish_reason": "stop", "message": {"content": "  "}})


def test_bound_output_budget_lowers_named_budgets_and_sets_an_unnamed_one() -> None:
    body = {"max_tokens": 100000, "max_completion_tokens": 4000}
    _bound_output_budget(body, 8192)
    assert body == {"max_tokens": 8192, "max_completion_tokens": 4000}
    body = {}
    _bound_output_budget(body, 8192)
    assert body == {"max_tokens": 8192}


def test_the_compaction_agent_header_selects_the_bounded_checked_summary_over_http() -> None:
    model = _model(compaction_summary_max_tokens=8192)
    worker = Worker([BAD, GOOD])
    model._clients = [worker]
    with TestClient(model.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Original task"}], "max_tokens": 100000},
            headers={OPENCODE_AGENT_HEADER: "compaction"},
        )
    assert response.status_code == 200, response.text
    assert len(worker.calls) == 2
    assert all(call["max_tokens"] == 8192 for call in worker.calls), "the summary's output is bounded on every attempt"
    assert response.json()["choices"][0]["message"]["content"] == GOOD["choices"][0]["message"]["content"]


@pytest.mark.parametrize("headers", [{}, {OPENCODE_AGENT_HEADER: "build"}])
def test_other_calls_and_an_unset_cap_keep_the_ordinary_path(headers) -> None:
    model = _model(compaction_summary_max_tokens=8192)
    worker = Worker([PLAIN])
    model._clients = [worker]
    with TestClient(model.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Build"}], "max_tokens": 100000},
            headers=headers,
        )
    assert response.status_code == 200, response.text
    assert len(worker.calls) == 1 and worker.calls[0]["max_tokens"] == 100000

    unset = _model()
    unset._clients = [Worker([PLAIN])]
    with TestClient(unset.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Original task"}]},
            headers={OPENCODE_AGENT_HEADER: "compaction"},
        )
    assert response.status_code == 200, response.text


def test_the_staged_header_plugin_sets_the_header_this_server_reads() -> None:
    """The Agentic-VBench compaction plugin and the model server agree on the agent header's name."""
    from pathlib import Path

    plugin = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/agentic_vbench/opencode_plugins/compaction_summary_header.mjs"
    )
    assert f'export const HEADER = "{OPENCODE_AGENT_HEADER}";' in plugin.read_text()
