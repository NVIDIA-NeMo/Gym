# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge asks judge.yaml's questions with its model settings and turns the replies into decisions."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from aiohttp import ClientResponseError

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_if.if_grading import judge as judge_module
from resources_servers.agentic_if.if_grading.configuration import config_hash, load_config
from resources_servers.agentic_if.if_grading.judge import Judge


def reply(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "resp",
            "object": "response",
            "created_at": 0,
            "model": "judge",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        }
    )


YES = '{"why":"It did.","answer":"yes"}'


def configuration(**changes):
    config = load_config()
    config["model"].update({"attempts": 1, **changes})
    return config


def recording(text=YES):
    sent = []
    return sent, lambda body: sent.append(body) or reply(text)


def test_judge_sends_the_calibrated_prompt_and_model_settings():
    sent, send = recording()
    config = configuration()
    judge = Judge(send, config=config)
    result = judge.evaluate(rubric="Did it use FINISH?", context="run", evidence="ASSISTANT: FINISH")
    assert result.status == "pass" and result.explanation == "[votes: yes, yes, yes] It did."
    expected = config["prompt"].format(
        question="Did it use FINISH?", context_name=config["contexts"]["run"]["title"], context="ASSISTANT: FINISH"
    )
    assert sent[0] == {
        "model": "nvidia/zai-org/glm-5.3",
        "input": [{"role": "user", "content": expected}],
        "temperature": 0.6,
        "reasoning": {"effort": "low", "summary": "auto"},
        "max_output_tokens": 8192,
    }
    assert len(sent) == 3  # the run context asks three times
    assert result.metadata["config_hash"] == config_hash(config)
    assert result.metadata["version"] == config["version"]


def test_the_shipped_judge_configuration_keeps_its_identity():
    # Results record this hash; prompts, model settings and votes are all part of it.
    assert config_hash(load_config()) == "5bf772a4a461"
    assert Judge(lambda body: reply(YES)).hash == "5bf772a4a461"


@pytest.mark.parametrize(
    "text,status",
    [
        ('{"answer":" YES "}', "pass"),
        ('{"kept":true,"why":"ok"}', "pass"),
        ('"answer": "yes"', "pass"),
        ('Checking the evidence.\n```json\n{"why":"Missing FINISH","answer":"no"}\n```', "fail"),
        ("not JSON", "error"),
        ('{"why":"?","answer":"maybe"}', "error"),
    ],
)
def test_reply_parsing_keeps_calibrated_verdicts_and_never_guesses_one(text, status):
    judge = Judge(lambda body: reply(text), config=configuration())
    result = judge.evaluate(rubric="Use FINISH.", context="message", evidence="Done")
    assert result.status == status
    if status == "error":
        assert result.explanation == "No judge verdict after 1 attempt(s): ValueError"
        assert not judge._cache


def test_verdicts_are_cached_by_question_for_one_grade():
    sent, send = recording()
    judge = Judge(send, config=configuration())
    args = dict(rubric="Keep it short.", context="message", evidence="done")
    first, again = judge.evaluate(**args), judge.evaluate(**args)
    assert len(sent) == 1 and first == again
    for changed in (dict(rubric="Use JSON."), dict(context="message_and_request"), dict(evidence="different")):
        judge.evaluate(**(args | changed))
    assert len(sent) == 4
    Judge(send, config=configuration()).evaluate(**args)  # another grade asks again
    assert len(sent) == 5


def test_parallel_same_judgment_shares_one_model_request():
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def send(body):
        calls.append(body)
        entered.set()
        assert release.wait(timeout=5)
        return reply(YES)

    judge = Judge(send, config=configuration())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(judge.evaluate, rubric="Be brief", context="message", evidence="done") for _ in range(3)
        ]
        assert entered.wait(timeout=5)
        release.set()
        assert [future.result().status for future in futures] == ["pass"] * 3
    assert len(calls) == 1


def test_a_failed_call_is_logged_retried_and_reported_as_an_error_without_its_details(monkeypatch, caplog):
    sleeps = []
    monkeypatch.setattr(judge_module.time, "sleep", sleeps.append)
    calls = []

    def send(body):
        calls.append(body)
        failure = ClientResponseError(
            SimpleNamespace(real_url="http://judge/v1/responses"), (), status=500, message="sensitive server detail"
        )
        raise JudgeError(f"ClientResponseError: {failure}") from failure

    judge = Judge(send, config=configuration(attempts=2))
    with caplog.at_level(logging.WARNING, logger=judge_module.__name__):
        result = judge.evaluate(rubric="Be brief", context="message", evidence="done")
    assert result.status == "error" and len(calls) == 2 and sleeps == [3]
    assert result.explanation == "No judge verdict after 2 attempt(s): ClientResponseError (HTTP 500)"
    assert [record.getMessage() for record in caplog.records] == [
        "Judge vote 0: attempt 1 of 2 failed: ClientResponseError (HTTP 500)",
        "Judge vote 0: attempt 2 of 2 failed: ClientResponseError (HTTP 500)",
    ]
    assert "sensitive" not in caplog.text
    assert not judge._cache
    # The next attempt of a vote that got no verdict is asked again; a verdict then counts.
    replies = iter([ValueError("no verdict"), reply(YES)])

    def flaky(body):
        answer = next(replies)
        if isinstance(answer, Exception):
            raise answer
        return answer

    result = Judge(flaky, config=configuration(attempts=2)).evaluate(
        rubric="Be brief", context="message", evidence="x"
    )
    assert result.status == "pass" and result.metadata["votes"][0]["metadata"]["attempts"] == 2


def test_invalid_judge_settings_fail_at_construction():
    with pytest.raises(ValueError, match="odd"):
        Judge(lambda body: reply(""), config=configuration(votes=2))
    with pytest.raises(ValueError, match="attempts"):
        Judge(lambda body: reply(""), config=configuration(attempts=0))
    with pytest.raises(ValueError, match="responses"):
        Judge(lambda body: reply(""), config=configuration(api="chat"))


def test_majority_votes_keep_all_decision_metadata():
    replies = iter(['{"why":"no 1","answer":"no"}', '{"why":"yes","answer":"yes"}', '{"why":"no 2","answer":"no"}'])
    judge = Judge(lambda body: reply(next(replies)), config=configuration(votes=3))
    result = judge.evaluate(rubric="Use FINISH", context="message", evidence="done")
    assert result.status == "fail"
    assert result.explanation == "[votes: no, yes, no] no 1"
    assert [vote["status"] for vote in result.metadata["votes"]] == ["fail", "pass", "fail"]


def test_context_votes_follow_the_long_run_threshold():
    sent, send = recording()
    judge = Judge(send, config=configuration())
    judge.evaluate(rubric="q", context="run", evidence="run")
    assert len(sent) == 3
    judge.evaluate(rubric="q", context="after_event", evidence="short", n_steps=59)
    assert len(sent) == 4
    judge.evaluate(rubric="q", context="after_event", evidence="long", n_steps=60)
    assert len(sent) == 7


def test_instruction_prompt_shows_the_literal_instruction_and_its_surroundings():
    sent, send = recording()
    judge = Judge(send, config=configuration())
    instruction = {"text": "End with FINISH.", "at": "task_message", "carrier": "Fix the bug. End with FINISH."}
    args = dict(rubric="Did it end as requested?", context="message", evidence="FINISH", instruction=instruction)
    result = judge.evaluate(**args)
    assert result.status == "pass"
    prompt = sent[0]["input"][0]["content"]
    assert "Fix the bug. [[End with FINISH.]]" in prompt
    assert result.metadata["judge_saw_instruction"] is True
    judge.evaluate(**args)
    assert len(sent) == 1
    judge.evaluate(**(args | {"instruction": instruction | {"carrier": "Only for summaries. End with FINISH."}}))
    assert len(sent) == 2
    judge.evaluate(**(args | {"instruction": instruction | {"at": "system_prompt"}}))
    assert len(sent) == 3


def test_a_question_without_its_instruction_uses_the_plain_prompt():
    sent, send = recording()
    config = configuration()
    result = Judge(send, config=config).evaluate(rubric="Use FINISH.", context="message", evidence="FINISH")
    assert result.metadata["judge_saw_instruction"] is False
    assert sent[0]["input"][0]["content"] == config["prompt"].format(
        question="Use FINISH.", context_name=config["contexts"]["message"]["title"], context="FINISH"
    )
