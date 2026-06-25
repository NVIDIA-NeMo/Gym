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
"""End-to-end: the real ``claude`` CLI drives the messages server, and NeMo Gym recovers
the exact token IDs the CLI never surfaces — proving token-ID handling lives in Gym, not
in the (un-editable) harness binary.

Only the GPU model is faked (a tiny OpenAI-compatible server returning synthetic token
IDs). Everything else is the production path: real CLI -> ClaudeMessagesModel SSE/converter
-> token-ID buffering -> reconciliation.
"""

import json
import os
import shutil
import socket
import subprocess
import threading
import time

import pytest
import requests
import uvicorn
from aiohttp import web
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_agent import reconcile_token_ids
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.openai_utils import (
    GenerationRecord,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from responses_api_agents.claude_code_agent.app import (
    parse_stream_json,
)
from responses_api_models.claude_messages_model.app import ClaudeMessagesModel, ClaudeMessagesModelConfig


pytestmark = pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")


@pytest.fixture(autouse=True)
def _seed_global_config():
    # The lazy global-aiohttp-client init reads the global config; without seeding it would
    # try to parse pytest's argv as Hydra overrides and SystemExit. Seed an empty config.
    get_global_config_dict(GlobalConfigDictParserConfig(skip_load_from_cli=True))


GEN_TOKENS = [101, 102, 103, 104]
PROMPT_TOKENS = list(range(10, 26))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_fake_vllm():
    async def chat_completions(request):
        body = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "The answer is 4.", "tool_calls": None},
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{t}", "logprob": -0.1 * i} for i, t in enumerate(GEN_TOKENS)
                            ]
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": len(PROMPT_TOKENS),
                    "completion_tokens": len(GEN_TOKENS),
                    "total_tokens": len(PROMPT_TOKENS) + len(GEN_TOKENS),
                },
            }
        )

    async def tokenize(request):
        return web.json_response({"tokens": PROMPT_TOKENS, "count": len(PROMPT_TOKENS)})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/tokenize", tokenize)
    return app


def _serve_aiohttp(app, port: int):
    web.run_app(app, host="127.0.0.1", port=port, handle_signals=False, print=None)


def _serve_uvicorn(app, port: int):
    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")).run()


def _wait_ready(url: str, timeout: float = 15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.head(url, timeout=1).status_code < 500:
                return
        except requests.RequestException:
            time.sleep(0.2)
    raise RuntimeError(f"server at {url} did not become ready")


def _start_messages_server(
    base_url: str, model: str, api_key: str = "EMPTY", uses_reasoning_parser: bool = False
) -> int:
    """Start a real ClaudeMessagesModel pointed at ``base_url``; return its port.

    ``base_url`` may be a fake vLLM (CI) or a real vLLM serve (the actual proof). The server
    runs locally as a plain Python process and forwards to ``base_url`` — no GPU needed here,
    only a reachable vLLM endpoint that supports logprobs + return_tokens_as_token_ids +
    /tokenize."""
    msgs_port = _free_port()
    cfg = ClaudeMessagesModelConfig(
        host="127.0.0.1",
        port=msgs_port,
        entrypoint="app.py",
        name="claude_messages_model",
        base_url=base_url,
        api_key=api_key,
        model=model,
        return_token_id_information=True,
        uses_reasoning_parser=uses_reasoning_parser,
    )
    server = ClaudeMessagesModel(
        config=cfg,
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    threading.Thread(target=_serve_uvicorn, args=(server.setup_webserver(), msgs_port), daemon=True).start()
    _wait_ready(f"http://127.0.0.1:{msgs_port}/")
    return msgs_port


def _start_fake_stack() -> int:
    """Start a fake vLLM + a real ClaudeMessagesModel in threads; return the messages port."""
    vllm_port = _free_port()
    threading.Thread(target=_serve_aiohttp, args=(_make_fake_vllm(), vllm_port), daemon=True).start()
    return _start_messages_server(f"http://127.0.0.1:{vllm_port}/v1", "fake", api_key="x")


_DISALLOWED_TOOLS = "Bash,Edit,Read,Write,Glob,Grep,WebFetch,WebSearch,Task"


def _run_claude_cli(base_url: str, model: str, prompt: str, cfgdir, timeout: int = 120) -> subprocess.CompletedProcess:
    """Drive the real claude CLI against ``base_url`` (a /runs/<token> messages endpoint)."""
    cfgdir.mkdir(exist_ok=True)
    (cfgdir / "settings.json").write_text(
        json.dumps(
            {
                "env": {
                    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                }
            }
        )
    )
    env = {
        **os.environ,
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_AUTH_TOKEN": "dummy",
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "IS_SANDBOX": "1",
        "CLAUDE_CONFIG_DIR": str(cfgdir),
    }
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--bare",
        "--max-turns",
        "2",
        "--disallowedTools",
        _DISALLOWED_TOOLS,
        "--model",
        model,
        "--",
        prompt,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


def test_real_cli_trajectory_carries_server_token_ids(tmp_path):
    msgs_port = _start_fake_stack()

    token = "run-test-123"
    base = f"http://127.0.0.1:{msgs_port}/runs/{token}"

    proc = _run_claude_cli(
        base, "fake", "What is 2+2? Reply in one short sentence.", tmp_path / "claude_cfg", timeout=90
    )
    assert proc.returncode == 0, proc.stderr[-1000:]

    # 1) server captured the token IDs the CLI never sees
    records_json = requests.get(f"{base}/buffered_generations", timeout=5).json()["generations"]
    assert records_json, "messages server buffered no generations"
    records = [GenerationRecord(**r) for r in records_json]

    # 2) the CLI's own stdout carries no token IDs
    output_items, _ = parse_stream_json(proc.stdout)
    assert output_items, "CLI produced no trajectory"

    # 3) Gym reconciles them on, with zero edits to the CLI binary
    reconciled = reconcile_token_ids(output_items, records)
    train_items = [i for i in reconciled if isinstance(i, NeMoGymResponseOutputMessageForTraining)]
    assert train_items, "reconciliation attached no token IDs"
    assert train_items[0].prompt_token_ids == PROMPT_TOKENS
    assert train_items[0].generation_token_ids == GEN_TOKENS


# NOTE: A full in-process ClaudeCodeAgent.responses() e2e is intentionally omitted. Co-locating
# the agent and the messages server in one process makes them share NeMo Gym's module-global
# aiohttp client (bound to the first loop that creates it), so the agent's reconcile fetch and
# the server's own upstream calls collide across event loops. In real ng_run they are separate
# processes and this never occurs. The agent's mint-token/resolve/fetch/reconcile orchestration
# is covered by the mocked TestTokenIdWiring unit test; the real token-ID path is covered below.


@pytest.mark.skipif(
    not os.getenv("VLLM_BASE_URL"),
    reason="set VLLM_BASE_URL (+ VLLM_MODEL) to a real vLLM serve to run the real-vLLM proof",
)
def test_real_vllm_through_claude_harness_buffers_token_ids(tmp_path):
    """THE proof: the real claude CLI drives a real vLLM-served open model through
    claude_messages_model, and the actual vLLM-produced token IDs (logprobs +
    return_tokens_as_token_ids + /tokenize) are buffered and reconciled onto the trajectory.

    Run, e.g.:
        VLLM_BASE_URL=http://<host>:8000/v1 VLLM_MODEL=<served-model> \\
          uv run pytest responses_api_models/claude_messages_model/tests/test_e2e_claude_cli.py \\
          -k real_vllm -s
    """
    base_url = os.environ["VLLM_BASE_URL"]
    model = os.environ.get("VLLM_MODEL", "")
    reasoning = os.getenv("VLLM_USES_REASONING_PARSER", "").lower() in ("1", "true", "yes")

    msgs_port = _start_messages_server(
        base_url, model, api_key=os.getenv("VLLM_API_KEY", "EMPTY"), uses_reasoning_parser=reasoning
    )
    token = "run-real-vllm-1"
    base = f"http://127.0.0.1:{msgs_port}/runs/{token}"

    proc = _run_claude_cli(
        base, model, "What is 2+2? Reply in one short sentence.", tmp_path / "claude_cfg", timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    # 1) The real vLLM's token IDs were buffered, and are well-formed (not synthetic).
    generations = requests.get(f"{base}/buffered_generations", timeout=10).json()["generations"]
    assert generations, "no generations buffered from the real vLLM"
    for g in generations:
        assert g["prompt_token_ids"], "empty prompt_token_ids from real vLLM"
        assert g["generation_token_ids"], "empty generation_token_ids from real vLLM"
        # vLLM returns one logprob per generated token.
        assert len(g["generation_token_ids"]) == len(g["generation_log_probs"])

    # 2) They reconcile onto the trajectory the CLI emitted, with zero edits to the binary.
    records = [GenerationRecord(**g) for g in generations]
    output_items, _ = parse_stream_json(proc.stdout)
    train_items = [
        i for i in reconcile_token_ids(output_items, records) if isinstance(i, NeMoGymResponseOutputMessageForTraining)
    ]
    assert train_items, "reconciliation attached no token IDs from the real vLLM"
    assert train_items[0].prompt_token_ids and train_items[0].generation_token_ids
    print(
        f"\nREAL-vLLM PROOF: model={model!r} buffered {len(generations)} generation(s); "
        f"first turn prompt_token_ids={len(train_items[0].prompt_token_ids)} "
        f"generation_token_ids={len(train_items[0].generation_token_ids)}"
    )
