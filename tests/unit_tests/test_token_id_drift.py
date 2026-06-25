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
"""Multi-turn on-policy / retokenization-drift prevention, the Gym side.

NeMo RL splices each prior turn's exact token IDs into the next turn's engine prompt (reading
them off the request) instead of re-tokenizing the assistant text. Gym's job, tested here: with
``forward_prefix_token_ids=True`` the model server injects the previous turn's *buffered* token
IDs onto each outgoing request's last assistant message, so the splice gets exact prior tokens.
A fake vLLM captures what Gym forwards; a plain vLLM never sees these fields, so the flag is off
by default.
"""

import asyncio
import socket
import threading
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from omegaconf import OmegaConf

import nemo_gym.server_utils as server_utils
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, BaseServerConfig, ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


# turn 1's exact generated tokens, and the *drifted* tokens turn 1's assistant text would
# re-tokenize to (what a naive next-turn /tokenize yields). the whole point is these differ.
TURN1_GEN_TOKENS = [101, 102]
TURN1_PROMPT_TOKENS = [1, 2, 3]
DRIFTED_A1_RETOKENIZATION = [999, 998]


@pytest.fixture(autouse=True)
def _seed_and_reset_global_client():
    # seed an empty global config (avoid Hydra parsing pytest argv) and give each test a fresh
    # global aiohttp client so per-test asyncio.run loops don't collide.
    get_global_config_dict(GlobalConfigDictParserConfig(skip_load_from_cli=True))
    server_utils._GLOBAL_AIOHTTP_CLIENT = None
    yield
    server_utils._GLOBAL_AIOHTTP_CLIENT = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_fake_vllm(captured: list) -> int:
    """A fake OpenAI-compatible vLLM that records each request's last assistant message (so we
    can see what Gym forwarded) and returns deterministic per-turn generation token IDs."""
    calls = {"n": 0}
    gen_by_turn = [TURN1_GEN_TOKENS, [201]]
    port = _free_port()

    async def chat_completions(request):
        body = await request.json()
        last_assistant = next((m for m in reversed(body.get("messages", [])) if m.get("role") == "assistant"), None)
        captured.append(last_assistant)
        gen = gen_by_turn[min(calls["n"], len(gen_by_turn) - 1)]
        calls["n"] += 1
        return web.json_response(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "m"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ans", "tool_calls": None},
                        "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": -0.1} for t in gen]},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": len(gen), "total_tokens": 3 + len(gen)},
            }
        )

    async def tokenize(request):
        body = await request.json()
        n = len(body.get("messages", []))
        # turn 1 prompt (system+user) -> TURN1_PROMPT_TOKENS. turn 2 re-tokenizes the whole
        # conversation, the assistant turn drifts to DRIFTED_A1_RETOKENIZATION (!= what was generated).
        toks = TURN1_PROMPT_TOKENS if n <= 2 else (TURN1_PROMPT_TOKENS + DRIFTED_A1_RETOKENIZATION + [4, 5])
        return web.json_response({"tokens": toks, "count": len(toks)})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/tokenize", tokenize)
    threading.Thread(
        target=lambda: web.run_app(app, host="127.0.0.1", port=port, handle_signals=False, print=None),
        daemon=True,
    ).start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)
    return port


def _model(port: int, forward_prefix_token_ids: bool) -> VLLMModel:
    cfg = VLLMModelConfig(
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        name="vllm_model",
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="x",
        model="m",
        return_token_id_information=True,
        uses_reasoning_parser=False,
        forward_prefix_token_ids=forward_prefix_token_ids,
    )
    return VLLMModel(
        config=cfg,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create({}),
        ),
    )


def _req():
    # minimal stand-in for a run-scoped FastAPI request: a run token (set by the /runs/<token>
    # middleware in production) plus a session id (used to pick the upstream client).
    return SimpleNamespace(state=SimpleNamespace(run_token="run-1"), session={SESSION_ID_KEY: "s1"})


def _params(messages):
    return NeMoGymChatCompletionCreateParamsNonStreaming(model="m", messages=messages, max_tokens=16)


_SYS = {"role": "system", "content": "be brief"}
_U1 = {"role": "user", "content": "turn one"}
_A1 = {"role": "assistant", "content": "the answer is four"}
_U2 = {"role": "user", "content": "turn two"}


async def _two_turns(model: VLLMModel):
    req = _req()
    await model.chat_completions(req, _params([_SYS, _U1]))  # turn 1
    await model.chat_completions(req, _params([_SYS, _U1, _A1, _U2]))  # turn 2 (carries assistant 1)


def test_forward_injection_carries_exact_prior_tokens_across_turns():
    captured: list = []
    port = _start_fake_vllm(captured)
    asyncio.run(_two_turns(_model(port, forward_prefix_token_ids=True)))

    assert len(captured) == 2
    assert captured[0] is None  # turn 1 had no prior assistant message
    a1 = captured[1]  # turn 2's last assistant message, as Gym forwarded it to vLLM
    assert a1 is not None
    # Gym injected turn 1's EXACT buffered tokens (prompt from /tokenize, generation from logprobs).
    assert a1["prompt_token_ids"] == TURN1_PROMPT_TOKENS
    assert a1["generation_token_ids"] == TURN1_GEN_TOKENS
    # the point of drift prevention: these are the tokens the model actually generated, NOT the
    # drifted re-tokenization of the assistant text, so NeMo RL splices the on-policy sequence.
    assert a1["generation_token_ids"] != DRIFTED_A1_RETOKENIZATION


def test_no_injection_when_flag_off():
    captured: list = []
    port = _start_fake_vllm(captured)
    asyncio.run(_two_turns(_model(port, forward_prefix_token_ids=False)))

    assert len(captured) == 2
    a1 = captured[1]
    assert a1 is not None
    # with the flag off, the request carries no token IDs, so a patched vLLM would have to
    # re-tokenize the assistant text (drift). off is the safe default for plain-vLLM eval.
    assert "prompt_token_ids" not in a1
    assert "generation_token_ids" not in a1
