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
import os
import tempfile
from unittest.mock import MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    RunTrajectory,
    SimpleResponsesAPIModel,
    TokenIDBufferingMixin,
    _main_chain,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMConverter


class _BufferingHost(TokenIDBufferingMixin):
    def __init__(self) -> None:
        d = tempfile.mkdtemp()
        self._trajectory = RunTrajectory(d)


class _BufferPathHost(TokenIDBufferingMixin):
    def __init__(self, global_config=None) -> None:
        self.server_client = MagicMock(global_config_dict=global_config or {})


class TestInjectPrefixTokenIds:
    def _request(self, token):
        req = MagicMock()
        req.state.run_token = token
        return req

    def _train(self, text, prompt, gen):
        return NeMoGymResponseOutputMessageForTraining(
            id=f"m-{text}",
            content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            role="assistant",
            status="completed",
            type="message",
            prompt_token_ids=prompt,
            generation_token_ids=gen,
            generation_log_probs=[-0.1] * len(gen),
        )

    def test_injects_prev_turn_onto_last_assistant_message(self) -> None:
        host = _BufferingHost()
        host._trajectory.append("tok", [self._train("a", [5, 6], [7])])
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "r"},
        ]
        host.attach_tokens_and_logprobs(self._request("tok"), messages)
        assert messages[1]["prompt_token_ids"] == [5, 6]
        assert messages[1]["generation_token_ids"] == [7]
        assert "prompt_token_ids" not in messages[0]

    def test_noop_without_token_or_empty(self) -> None:
        host = _BufferingHost()
        messages = [{"role": "assistant", "content": "a"}]
        host.attach_tokens_and_logprobs(self._request(None), messages)
        host.attach_tokens_and_logprobs(self._request("tok"), messages)
        assert "prompt_token_ids" not in messages[0]


class TestRunTokenMiddleware:
    """The /runs/<token> middleware strips the prefix, sets request.state.run_token so normal routes still match."""

    def _client(self) -> tuple[_BufferingHost, TestClient]:
        host = _BufferingHost()
        app = FastAPI()

        @app.get("/echo")
        async def echo(request: Request):
            return {"token": request.state.run_token, "path": request.url.path}

        host.setup_token_id_buffering(app)
        return host, TestClient(app)

    def test_strips_prefix_and_sets_token(self) -> None:
        _, client = self._client()
        scoped = client.get("/runs/abc/echo").json()
        assert scoped == {"token": "abc", "path": "/echo"}
        assert client.get("/echo").json()["token"] is None


class TestRunTrajectory:
    def test_fallback_buffer_dir_is_shared_across_model_instances(self, monkeypatch, tmp_path) -> None:
        monkeypatch.delenv("NEMO_GYM_TOKEN_ID_BUFFER_DIR", raising=False)
        monkeypatch.setenv("SLURM_JOB_ID", "job/123")
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

        first = _BufferPathHost().token_id_buffer_dir()
        second = _BufferPathHost().token_id_buffer_dir()

        assert first == second == str(tmp_path / f"nemo_gym_token_id_buffer_{os.getuid()}_job_123")
        first_trajectory = RunTrajectory(first)
        second_trajectory = RunTrajectory(second)
        first_trajectory.append("tok", [NeMoGymResponseOutputMessage(id="m", content=[], role="assistant")])
        assert len(second_trajectory.read("tok")) == 1

    def test_explicit_buffer_dir_overrides_global_log_dir(self, monkeypatch, tmp_path) -> None:
        configured = tmp_path / "configured"
        monkeypatch.setenv("NEMO_GYM_TOKEN_ID_BUFFER_DIR", str(configured))
        host = _BufferPathHost({"nemo_gym_log_dir": str(tmp_path / "logs")})
        assert host.token_id_buffer_dir() == str(configured)

    def test_append_read_pop_roundtrips_and_clears(self) -> None:
        t = RunTrajectory(tempfile.mkdtemp())
        assert t.read("tok") == []  # missing file -> empty
        t.append("tok", [NeMoGymResponseOutputMessage(id="m", content=[], role="assistant")])
        t.append(
            "tok",
            [NeMoGymFunctionCallOutput(call_id="c1", output="ok", status="completed")],
        )
        items = t.read("tok")
        assert [type(i) for i in items] == [NeMoGymResponseOutputMessage, NeMoGymFunctionCallOutput]
        assert t.read("tok")  # read is non-destructive
        popped = t.pop("tok")
        assert len(popped) == 2 and t.read("tok") == []  # pop clears

    def test_append_empty_is_noop(self) -> None:
        t = RunTrajectory(tempfile.mkdtemp())
        t.append("tok", [])
        assert t.read("tok") == []


def _gen_message(content=None, tool_calls=None, prompt=None, gen=None) -> dict:
    """An assistant message_dict as _generate_chat_completion produces it: content/tool_calls plus
    token IDs attached (prompt_token_ids/generation_token_ids/generation_log_probs) when run-scoped."""
    m: dict = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    if prompt is not None:
        m.update(
            prompt_token_ids=prompt,
            generation_token_ids=gen,
            generation_log_probs=[-0.1] * len(gen),
        )
    return m


def _tc(call_id: str, name: str = "Bash", args: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


class TestRecordTrajectoryTurnOnline:
    """The MORE-ONLINE design: the model server builds the trajectory itself across turns, from the
    actual incoming prompts (previous-turn tool outputs) + the generation it just produced."""

    def _host(self):
        host = _BufferingHost()
        # the real converter, exactly as vllm_model wires it (handles think-tags, tool_calls, and
        # upconverts the last trainable item to its ...ForTraining variant when token IDs are present).
        build = VLLMConverter(return_token_id_information=True).postprocess_assistant_message_dict
        return host, build

    def _request(self, token):
        req = MagicMock()
        req.state.run_token = token
        return req

    def test_multi_turn_trajectory_is_ordered_contiguous_with_token_ids(self) -> None:
        host, build = self._host()
        req = self._request("t")

        # turn 1: assistant text + a tool call. prompt [1,2,3], gen [4,5].
        host.buffer_turn(
            req,
            [{"role": "user", "content": "q"}],
            _gen_message(content="thinking", tool_calls=[_tc("c1")], prompt=[1, 2, 3], gen=[4, 5]),
            build,
        )
        # turn 2: the request now carries the kept assistant msg + the tool output for c1. The next
        # generation's prompt continues the chain: [1,2,3,4,5] + the tool-output tokens [6,7].
        host.buffer_turn(
            req,
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "thinking", "tool_calls": [_tc("c1")]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ],
            _gen_message(content="final answer", prompt=[1, 2, 3, 4, 5, 6, 7], gen=[8]),
            build,
        )

        items = host._trajectory.pop("t")
        # ordered: turn1 msg, turn1 tool_call, c1 output, turn2 msg
        assert [type(i) for i in items] == [
            NeMoGymResponseOutputMessage,
            NeMoGymResponseFunctionToolCallForTraining,
            NeMoGymFunctionCallOutput,
            NeMoGymResponseOutputMessageForTraining,
        ]
        # token IDs land on the LAST trainable item of each turn (the tool call, then the final msg)
        assert items[1].prompt_token_ids == [1, 2, 3] and items[1].generation_token_ids == [4, 5]
        assert items[2].call_id == "c1" and items[2].output == "result"
        assert items[3].prompt_token_ids == [1, 2, 3, 4, 5, 6, 7] and items[3].generation_token_ids == [8]
        # contiguity by construction: turn 2's prompt extends turn 1's prompt+generation.
        assert items[3].prompt_token_ids[: 3 + 2] == items[1].prompt_token_ids + items[1].generation_token_ids

    def test_retried_discarded_turn_dropped_by_main_chain(self) -> None:
        host, build = self._host()
        req = self._request("t")

        # turn 1 generated, then retried/discarded and re-generated. Recording is append-only, so both
        # attempts land in the trajectory; _main_chain keeps only the retry (the later sibling sharing
        # the same prompt) and drops the discarded attempt.
        host.buffer_turn(
            req, [{"role": "user", "content": "q"}], _gen_message(content="a1", prompt=[1], gen=[2]), build
        )
        host.buffer_turn(
            req, [{"role": "user", "content": "q"}], _gen_message(content="a1-retry", prompt=[1], gen=[9]), build
        )

        items = host._trajectory.pop("t")
        assert len(items) == 2  # append-only: both attempts recorded
        trainable = [it for it in _main_chain(items) if getattr(it, "prompt_token_ids", None) is not None]
        assert len(trainable) == 1 and trainable[0].generation_token_ids == [9]  # the retry, not [2]

    def test_main_chain_picks_main_not_longer_subagent(self) -> None:
        host, build = self._host()
        req = self._request("t")
        um = [{"role": "user", "content": "main"}]
        us = [{"role": "user", "content": "sub"}]
        # main turn 1, then a sub-agent (different starting prompt) that runs THREE turns, then main
        # turn 2 continuing turn 1. The sub-agent chain is longer, but it is not seeded (its prompt
        # differs from the first generation's), so the 2-turn main chain is the one kept.
        host.buffer_turn(req, um, _gen_message(content="m1", prompt=[1, 2, 3], gen=[4]), build)
        host.buffer_turn(req, us, _gen_message(content="s1", prompt=[9, 9, 9], gen=[8]), build)
        host.buffer_turn(req, us, _gen_message(content="s2", prompt=[9, 9, 9, 8, 1], gen=[2]), build)
        host.buffer_turn(req, us, _gen_message(content="s3", prompt=[9, 9, 9, 8, 1, 2, 3], gen=[5]), build)
        host.buffer_turn(req, um, _gen_message(content="m2", prompt=[1, 2, 3, 4, 5, 6], gen=[7]), build)

        kept = _main_chain(host._trajectory.read("t"))
        gens = [it.generation_token_ids for it in kept if getattr(it, "prompt_token_ids", None) is not None]
        assert gens == [[4], [7]], gens  # the 2 main turns, not the 3 longer sub-agent turns

    def test_function_call_output_role_input_items_recorded(self) -> None:
        # some harnesses send Responses-style function_call_output items rather than role=="tool".
        host, build = self._host()
        req = self._request("t")
        host.buffer_turn(
            req,
            [{"role": "user", "content": "q"}],
            _gen_message(content="", tool_calls=[_tc("c1")], prompt=[1], gen=[2]),
            build,
        )
        host.buffer_turn(
            req,
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [_tc("c1")]},
                {"type": "function_call_output", "call_id": "c1", "output": "done"},
            ],
            _gen_message(content="ok", prompt=[1, 2, 3], gen=[4]),
            build,
        )
        items = host._trajectory.read("t")
        outputs = [i for i in items if isinstance(i, NeMoGymFunctionCallOutput)]
        assert len(outputs) == 1 and outputs[0].call_id == "c1" and outputs[0].output == "done"

    def test_noop_without_token(self) -> None:
        host, build = self._host()
        host.buffer_turn(
            self._request(None),
            [{"role": "user", "content": "q"}],
            _gen_message(content="a", prompt=[1], gen=[2]),
            build,
        )
        assert host._trajectory.read("t") == []


class TestPopTrajectoryEndpoint:
    def _client(self):
        host = _BufferingHost()
        app = FastAPI()
        host.setup_token_id_buffering(app)
        return host, TestClient(app)

    def test_trajectory_pop_and_empty(self) -> None:
        host, client = self._client()
        host._trajectory.append("abc", [NeMoGymResponseOutputMessage(id="m", content=[], role="assistant")])
        out = client.get("/runs/abc/trajectory").json()["output"]
        assert len(out) == 1 and out[0]["type"] == "message"
        assert client.get("/runs/abc/trajectory").json()["output"] == []  # pop cleared it
        assert client.get("/trajectory").json()["output"] == []  # no token -> empty


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")

        class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
            async def chat_completions(
                self, request: NeMoGymResponseCreateParamsNonStreaming
            ) -> NeMoGymChatCompletion:
                raise NotImplementedError

            async def responses(self, request: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
                raise NotImplementedError

        model = TestSimpleResponsesAPIModel(config=config, server_client=MagicMock(spec=ServerClient))
        model.setup_webserver()
