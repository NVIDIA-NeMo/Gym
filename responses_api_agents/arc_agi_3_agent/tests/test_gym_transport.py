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
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import requests

from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter as VLLMConverter
from responses_api_agents.arc_agi_3_agent.call_log import call_log_entries, harness_log_entries
from responses_api_agents.arc_agi_3_agent.gym_transport import (
    GymChatTransport,
    GymHTTPResponse,
    RequestsProxy,
    chat_messages_summary,
    continues_closed_reasoning,
    from_gym_chat_response,
    history_message_to_gym,
    split_reasoning,
    to_gym_chat_request,
)


def harness_payload(**extra):
    payload = {
        "model": "policy",
        "messages": [
            {"role": "system", "content": "rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "board"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "think about it",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "python", "arguments": '{"code": "x"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ],
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": 12288,
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    payload.update(extra)
    return payload


class TestRequestMapping:
    def test_extras_go_to_metadata(self) -> None:
        body = to_gym_chat_request(harness_payload(), chat_template_kwargs={"truncate_history_thinking": False})
        assert "top_k" not in body and "chat_template_kwargs" not in body and "stream" not in body
        assert json.loads(body["metadata"]["extra_body"]) == {"top_k": 20}
        assert json.loads(body["metadata"]["chat_template_kwargs"]) == {
            "truncate_history_thinking": False,
            "enable_thinking": True,
        }
        assert body["temperature"] == 0.7 and body["max_tokens"] == 12288 and body["tool_choice"] == "auto"
        # Gym's request model accepts the body unchanged (nothing is silently dropped).
        validated = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body).model_dump(exclude_unset=True)
        assert set(body) <= set(validated)

    def test_request_values_win_over_config(self) -> None:
        body = to_gym_chat_request(
            harness_payload(chat_template_kwargs={"enable_thinking": False}),
            chat_template_kwargs={"enable_thinking": True, "medium_effort": True},
            extra_body={"top_k": 5, "min_p": 0.1},
        )
        assert json.loads(body["metadata"]["chat_template_kwargs"]) == {
            "enable_thinking": False,
            "medium_effort": True,
        }
        assert json.loads(body["metadata"]["extra_body"]) == {"top_k": 20, "min_p": 0.1}

    def test_reasoning_effort_on_thinking_requests(self) -> None:
        def kwargs(payload: dict[str, Any], **effort: Any) -> dict[str, Any]:
            body = to_gym_chat_request(payload, **effort)
            return json.loads(body["metadata"]["chat_template_kwargs"])

        assert kwargs(harness_payload(), reasoning_effort="low") == {
            "enable_thinking": True,
            "reasoning_effort": "low",
        }
        assert kwargs(harness_payload(), reasoning_effort="high", reasoning_effort_kwarg="thinking_effort") == {
            "enable_thinking": True,
            "thinking_effort": "high",
        }
        # a request without thinking, or with its own effort (the harness's effort ladder), is unchanged
        no_thinking = harness_payload(chat_template_kwargs={"enable_thinking": False})
        assert kwargs(no_thinking, reasoning_effort="low") == {"enable_thinking": False}
        ladder = harness_payload(chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "low"})
        assert kwargs(ladder, reasoning_effort="max")["reasoning_effort"] == "low"
        top = to_gym_chat_request(harness_payload(reasoning_effort="high"), reasoning_effort="max")
        assert top["reasoning_effort"] == "high" and json.loads(top["metadata"]["chat_template_kwargs"]) == {
            "enable_thinking": True
        }
        assert kwargs(harness_payload()) == {"enable_thinking": True}  # unset: the template's default

    def test_continuation_sets_add_generation_prompt(self) -> None:
        body = to_gym_chat_request(harness_payload(continue_final_message=True, stop=["</think>"]))
        extras = json.loads(body["metadata"]["extra_body"])
        assert extras["continue_final_message"] is True and extras["add_generation_prompt"] is False
        assert body["stop"] == ["</think>"]

    def test_reasoning_becomes_think_prefix_and_round_trips(self) -> None:
        body = to_gym_chat_request(harness_payload())
        assistant = body["messages"][2]
        assert assistant["content"] == "<think>think about it</think>"
        assert "reasoning_content" not in assistant
        # What the Gym vLLM model server does with it (uses_reasoning_parser): the same reasoning comes back out.
        matches, rest = VLLMConverter(
            return_token_id_information=False, uses_reasoning_parser=True
        )._extract_reasoning_from_content(assistant["content"])
        assert matches == ["think about it"] and rest == ""

    def test_history_message_variants(self) -> None:
        assert history_message_to_gym({"role": "user", "content": "x"}) == {"role": "user", "content": "x"}
        assert history_message_to_gym({"role": "assistant", "content": "a"}) == {"role": "assistant", "content": "a"}
        out = history_message_to_gym({"role": "assistant", "content": None, "reasoning": "r"})
        assert out == {"role": "assistant", "content": "<think>r</think>"}
        out = history_message_to_gym(
            {"role": "assistant", "content": [{"type": "text", "text": "a"}], "reasoning": "r"}
        )
        assert out["content"][0] == {"type": "text", "text": "<think>r</think>"}
        with pytest.raises(TypeError):
            history_message_to_gym({"role": "assistant", "content": 3, "reasoning": "r"})
        # an engine's extras on a tool call (SGLang's index, at either level) are dropped: Gym's request model refuses them
        call = {
            "id": "c1",
            "index": 0,
            "type": "function",
            "function": {"name": "python", "arguments": "{}", "index": 0},
        }
        out = history_message_to_gym({"role": "assistant", "content": "", "tool_calls": [call]})
        assert out["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
        ]

    def test_refuses_streaming_and_engine_fields(self) -> None:
        with pytest.raises(ValueError, match="non-streaming"):
            to_gym_chat_request(harness_payload(stream=True))
        with pytest.raises(ValueError, match="gdn_ownership"):
            to_gym_chat_request(harness_payload(gdn_ownership={"x": 1}))

    def test_summary(self) -> None:
        assert chat_messages_summary(harness_payload()["messages"]) == {"messages": 4, "images": 1}


class TestResponseMapping:
    def test_split_reasoning(self) -> None:
        assert split_reasoning("<think>a\nb</think>call") == ("a\nb", "call")
        assert split_reasoning("plain") == (None, "plain")
        assert split_reasoning(None) == (None, None)

    def test_from_gym(self) -> None:
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "<think>R</think>C", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3},
        }
        out = from_gym_chat_response(payload)
        message = out["choices"][0]["message"]
        assert message["content"] == "C" and message["reasoning_content"] == "R" and message["reasoning"] == "R"
        assert out["usage"] == {"prompt_tokens": 3}
        assert payload["choices"][0]["message"]["content"] == "<think>R</think>C"

    def test_http_response(self) -> None:
        ok = GymHTTPResponse(200, json.dumps({"choices": [{"message": {"content": "<think>R</think>C"}}]}), "u")
        assert ok.json()["choices"][0]["message"]["reasoning"] == "R"
        ok.raise_for_status()
        bad = GymHTTPResponse(500, json.dumps({"error": "x"}), "u")
        assert bad.json() == {"error": "x"}
        with pytest.raises(requests.HTTPError):
            bad.raise_for_status()
        with pytest.raises(RuntimeError):
            ok.iter_lines()


class TestTransport:
    def run_loop(self):
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        return loop

    def test_post_routes_through_gym_with_sticky_cookies(self) -> None:
        loop = self.run_loop()
        seen = []

        async def post(body, cookies, game_key=None):
            seen.append((body, cookies))
            reply = {"choices": [{"message": {"role": "assistant", "content": "<think>R</think>"}}]}
            return 200, json.dumps(reply), {"session": f"s{len(seen)}"}

        transport = GymChatTransport(loop, post, chat_template_kwargs={"truncate_history_thinking": False})
        proxy = RequestsProxy(transport)
        transport.bind_game("0:ls20")
        first = proxy.post("http://policy/v1/chat/completions", json=harness_payload(), timeout=(10, 30))
        second = proxy.post("http://policy/v1/chat/completions", json=harness_payload(), timeout=30)
        transport.bind_game(None)
        proxy.post("http://policy/v1/chat/completions", json=harness_payload(), timeout=30)
        assert first.json()["choices"][0]["message"]["reasoning_content"] == "R"
        assert second.status_code == 200
        assert [c for _, c in seen] == [None, {"session": "s1"}, None]
        assert transport.requests_sent == 3
        assert proxy.exceptions is requests.exceptions
        loop.call_soon_threadsafe(loop.stop)

    def test_timeout_and_connection_errors(self) -> None:
        loop = self.run_loop()

        async def slow(body, cookies, game_key=None):
            await asyncio.sleep(5)

        async def broken(body, cookies, game_key=None):
            raise OSError("refused")

        with pytest.raises(requests.exceptions.ReadTimeout):
            GymChatTransport(loop, slow).post_chat("u", harness_payload(), (1, 0.2))
        with pytest.raises(requests.exceptions.ConnectionError):
            GymChatTransport(loop, broken).post_chat("u", harness_payload(), 5)
        loop.call_soon_threadsafe(loop.stop)

    def test_calls_are_logged_with_the_games_progress_and_failures(self) -> None:
        loop = self.run_loop()
        answers = iter(
            [
                (200, json.dumps({"id": "r0", "choices": []}), {}),
                OSError("refused"),
                (503, json.dumps({"error": "busy"}), {}),
            ]
        )

        async def post(body, cookies, game_key=None):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        transport = GymChatTransport(loop, post)
        game = {"step": 3, "level": 2}
        transport.bind_game("0:ls20", progress=lambda: (game["step"], game["level"]))
        transport.post_chat("u", harness_payload(), 30)
        game.update(step=9, level=3)
        with transport.call_role("helper", "drain-summary"), pytest.raises(requests.exceptions.ConnectionError):
            transport.post_chat("u", harness_payload(), 30)
        transport.bind_game("0:ls20", progress=lambda: 1 / 0)  # unreadable progress: logged without it
        assert transport.post_chat("u", harness_payload(), 30).status_code == 503
        transport.bind_game(None)
        entries = call_log_entries(transport.take_call_log("0:ls20"))
        assert [(e["role"], e["history_id"], e["status"], e["response_id"]) for e in entries] == [
            ("gameplay", "main", 200, "r0"),
            ("helper", "drain-summary", 0, None),
            ("gameplay", "main", 503, None),
        ]
        assert [(e["step"], e["level"]) for e in entries] == [(3, 2), (9, 3), (None, None)]
        assert entries[1]["error"].startswith("ConnectionError: Gym model server request failed")
        assert all(e["started_at"] <= e["completed_at"] for e in entries)
        loop.call_soon_threadsafe(loop.stop)

    def test_harness_warnings_are_logged_per_game_with_its_progress(self) -> None:
        """The harness's warnings on a game's thread go to that game's log (the rollout's harness_log), with the game's
        step and level then; other threads, other games and lines below WARNING are not recorded."""
        transport = GymChatTransport(asyncio.new_event_loop(), None)
        transport.capture_logs()
        transport.capture_logs()  # once per transport
        harness = logging.getLogger("inference.agent.tool_agent")
        try:
            harness.warning("not in a game")
            transport.bind_game("0:ls20", progress=lambda: (12, 2))
            harness.warning("drain summary unusable (%d chars), plain drain kept", 12)
            harness.info("level summary written")

            def other_game() -> None:
                transport.bind_game("1:ft09")
                harness.warning("turn compaction request failed, raw turn kept: boom")

            thread = threading.Thread(target=other_game)
            thread.start()
            thread.join()
            transport.bind_game(None)
            harness.warning("after the game")
        finally:
            logging.getLogger().removeHandler(transport._log_capture)
        lines = harness_log_entries(transport.take_call_log("0:ls20"))["lines"]
        assert [(e["step"], e["level"], e["severity"], e["logger"], e["message"]) for e in lines] == [
            (12, 2, "WARNING", "inference.agent.tool_agent", "drain summary unusable (12 chars), plain drain kept")
        ]
        other = harness_log_entries(transport.take_call_log("1:ft09"))["lines"]
        assert [(e["step"], e["level"], e["message"]) for e in other] == [
            (None, None, "turn compaction request failed, raw turn kept: boom")
        ]

    def test_other_urls_refused(self) -> None:
        proxy = RequestsProxy(GymChatTransport(asyncio.new_event_loop(), None))
        with pytest.raises(requests.RequestException, match="only POST /chat/completions"):
            proxy.post("http://engine/set_internal_state", json={})
        with pytest.raises(requests.RequestException, match="without a JSON body"):
            proxy.post("http://policy/v1/chat/completions")


class TestContinuations:
    def closed_continuation(self, prefix="R\n</think>\n\n"):
        payload = harness_payload(continue_final_message=True)
        payload["messages"].append({"role": "assistant", "content": prefix})
        return payload

    def test_detects_closed_reasoning_prefix(self) -> None:
        assert continues_closed_reasoning(self.closed_continuation())
        assert not continues_closed_reasoning(self.closed_continuation(prefix="no close tag"))
        assert not continues_closed_reasoning(harness_payload())

    def test_reasoning_only_reply_becomes_content(self) -> None:
        reply = {"choices": [{"message": {"role": "assistant", "content": "<think>TURN RECORD ...</think>"}}]}
        out = from_gym_chat_response(reply, reasoning_is_content=True)["choices"][0]["message"]
        assert out["content"] == "TURN RECORD ..." and "reasoning" not in out and "reasoning_content" not in out
        # A reply that already has content keeps the split.
        reply = {"choices": [{"message": {"role": "assistant", "content": "<think>R</think>C"}}]}
        out = from_gym_chat_response(reply, reasoning_is_content=True)["choices"][0]["message"]
        assert out["content"] == "C" and out["reasoning"] == "R"

    def test_transport_marks_continuations(self) -> None:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        async def post(body, cookies, game_key=None):
            reply = {"choices": [{"message": {"role": "assistant", "content": "<think>record</think>"}}]}
            return 200, json.dumps(reply), {}

        transport = GymChatTransport(loop, post)
        continued = transport.post_chat("u", self.closed_continuation(), 5).json()
        fresh = transport.post_chat("u", harness_payload(), 5).json()
        assert continued["choices"][0]["message"]["content"] == "record"
        assert fresh["choices"][0]["message"]["reasoning"] == "record"
        loop.call_soon_threadsafe(loop.stop)


class TestCancellation:
    """F6: a cancelled attempt's model calls stop waiting and raise into the harness's request-failure path."""

    def run_loop(self):
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        return loop

    def test_cancelled_before_send(self) -> None:
        sent = []

        async def post(body, cookies, game_key=None):
            sent.append(body)
            return 200, "{}", {}

        transport = GymChatTransport(self.run_loop(), post)
        transport.bind_game("1:ls20")
        transport.cancel("1:ls20")
        with pytest.raises(requests.RequestException, match="cancelled"):
            transport.post_chat("u", harness_payload(), 30)
        assert sent == []
        transport.bind_game("2:ft09")  # another attempt is unaffected
        transport.post_chat("u", harness_payload(), 30)
        assert len(sent) == 1

    def test_cancelled_while_waiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from responses_api_agents.arc_agi_3_agent import gym_transport

        monkeypatch.setattr(gym_transport, "CANCEL_POLL_S", 0.05)

        async def slow(body, cookies, game_key=None):
            await asyncio.sleep(30)

        transport = GymChatTransport(self.run_loop(), slow)
        transport.bind_game("1:ls20")
        threading.Timer(0.2, transport.cancel_all).start()
        start = time.monotonic()
        with pytest.raises(requests.RequestException, match="cancelled during a model call"):
            transport.post_chat("u", harness_payload(), 30)
        assert time.monotonic() - start < 5

    def call_in_thread(self, transport, timeout):
        """post_chat in a daemon thread (a regression that hangs must fail the test, not hang the suite)."""
        out = {}

        def call():
            start = time.monotonic()
            try:
                transport.post_chat("u", harness_payload(), timeout)
            except BaseException as exc:  # noqa: BLE001 - the test inspects it
                out["error"] = exc
            out["seconds"] = time.monotonic() - start

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), "post_chat did not return"
        return out

    @pytest.mark.parametrize("timeout", [30, None, (5, 30)])
    def test_upstream_timeout_is_raised_at_once(self, timeout) -> None:
        # F9: a call that finished by raising a timeout is not a wait that ran out: no polling until the deadline.
        async def times_out(body, cookies, game_key=None):
            raise asyncio.TimeoutError("body read timed out")

        out = self.call_in_thread(GymChatTransport(self.run_loop(), times_out), timeout)
        assert isinstance(out["error"], requests.exceptions.ReadTimeout)
        assert "timed out" in str(out["error"]) and out["seconds"] < 1

    def test_upstream_os_error_is_a_connection_error(self) -> None:
        async def refused(body, cookies, game_key=None):
            raise ConnectionRefusedError("refused")

        out = self.call_in_thread(GymChatTransport(self.run_loop(), refused), None)
        assert isinstance(out["error"], requests.exceptions.ConnectionError) and out["seconds"] < 1

    def test_pending_call_times_out_at_the_deadline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from responses_api_agents.arc_agi_3_agent import gym_transport

        monkeypatch.setattr(gym_transport, "CANCEL_POLL_S", 0.05)

        async def slow(body, cookies, game_key=None):
            await asyncio.sleep(30)

        out = self.call_in_thread(GymChatTransport(self.run_loop(), slow), 0.3)
        assert isinstance(out["error"], requests.exceptions.ReadTimeout)
        assert "did not answer in 0.3 s" in str(out["error"]) and 0.25 < out["seconds"] < 2


class TestContextLengthRejection:
    """F3: the model server's 400 reaches the harness as the request exception its trim-and-retry path expects."""

    ENGINE_400 = (
        "This model's maximum context length is 262144 tokens. However, you requested 270000 tokens in the "
        "messages, Please reduce the length of the messages."
    )

    def harness_error(self, response):
        # The pinned harness's handling of a non-2xx reply (tool_agent._chat_completion).
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return requests.RequestException(f"{exc} | response: {response.text.strip()}")
        return None

    def test_400_becomes_a_context_length_error(self) -> None:
        response = GymHTTPResponse(400, json.dumps({"detail": self.ENGINE_400}), "http://policy/v1/chat/completions")
        error = self.harness_error(response)
        assert error is not None and "maximum context length" in str(error)
        harness_root = (
            Path(__file__).resolve().parents[1]
            / ".arc3_harness/harness/taaf/src/ARC3-Inference/inference/agent/tool_agent.py"
        )
        if not harness_root.is_file():
            pytest.skip("harness not installed: checked against the message only")
        source = harness_root.read_text()
        start = source.index("def _is_context_length_error")
        namespace: dict = {}
        exec(source[start : source.index("\n\n\n", start)], namespace)  # the harness's own predicate, verbatim
        assert namespace["_is_context_length_error"](error)
