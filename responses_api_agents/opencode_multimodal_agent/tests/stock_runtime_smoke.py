# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the published stock binary with a scripted SSE model (no GPU/model claim)."""

import argparse
import asyncio
import base64
import io
import json
import os
import shlex
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from aiohttp import web
from omegaconf import OmegaConf
from PIL import Image

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from responses_api_agents.opencode_multimodal_agent.app import (
    OpenCodeMultimodalAgent,
    OpenCodeMultimodalAgentConfig,
)
from responses_api_agents.opencode_multimodal_agent.media import final_answer
from responses_api_models.vllm_model.app_opencode import normalize_opencode_final_instruction


async def main(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(Path(args.opencode_bin).resolve().parent) + os.pathsep + os.environ["PATH"]
    calls = []
    expected_turns = 2 if args.early_final else 3

    async def complete(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        calls.append({"request": payload, "headers": dict(request.headers)})
        index = len(calls)
        if index > expected_turns:
            raise web.HTTPBadRequest(text="Unexpected additional model call")
        if index == 1:
            command = shlex.join(
                [sys.executable, "-c", "from PIL import Image; Image.new('RGB',(4,4),'red').save('crop.png')"]
            )
            name, arguments = "bash", {"command": command, "description": "Create a test image"}
        else:
            name, arguments = "read", {"filePath": "crop.png"}
        delta = (
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": f"call{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            if index < expected_turns
            else {"role": "assistant", "content": "3"}
        )
        if index == expected_turns and args.misbehave_final:
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "forbidden",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps(
                                {
                                    "command": "touch " + shlex.quote(str(output / "unexpected-final-tool")),
                                    "description": "This final-turn tool must be blocked",
                                }
                            ),
                        },
                    }
                ],
            }
        calls[-1]["response"] = delta
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for data in [
            {
                "id": f"chat{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "policy",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"chat{index}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "policy",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if index < expected_turns or args.misbehave_final else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 47480 if args.early_final else 100,
                    "completion_tokens": 20,
                    "total_tokens": 47500 if args.early_final else 120,
                },
            },
        ]:
            await response.write(("data: " + json.dumps(data) + "\n\n").encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    server = web.Application()
    server.router.add_post("/v1/chat/completions", complete)
    http = web.AppRunner(server)
    await http.setup()
    site = web.TCPSite(http, "127.0.0.1", 0)
    await site.start()
    port = http.addresses[0][1]
    try:
        with TemporaryDirectory(prefix="stock-opencode-smoke-") as temp:
            cfg = OpenCodeMultimodalAgentConfig(
                host="127.0.0.1",
                port=1,
                entrypoint="app.py",
                name="visual",
                model="policy",
                model_server={"type": "responses_api_models", "name": "policy_model"},
                resources_server={"type": "resources_servers", "name": "string_match"},
                workspace_root=str(Path(temp) / "workspaces"),
                concurrency=1,
                max_turns=3,
                timeout=120,
                env={
                    "XDG_CONFIG_HOME": str(Path(temp) / "config"),
                    "XDG_CACHE_HOME": str(Path(temp) / "cache"),
                    "OPENCODE_DISABLE_MODELS_FETCH": "1",
                },
            )
            client = ServerClient(
                head_server_config={"host": "127.0.0.1", "port": 0}, global_config_dict=OmegaConf.create({})
            )
            agent = OpenCodeMultimodalAgent(config=cfg, server_client=client)
            stream = io.BytesIO()
            Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
                {
                    "temperature": 1,
                    "top_p": 1,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Create an image with bash, read it, then answer 3."},
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,"
                                    + base64.b64encode(stream.getvalue()).decode(),
                                    "detail": "auto",
                                },
                            ],
                        }
                    ],
                }
            )
            # Replace address discovery only; installer, provider and execution remain inherited.
            with patch.object(
                OpenCodeMultimodalAgent, "_resolve_model_base_url", return_value=f"http://127.0.0.1:{port}/v1"
            ):
                try:
                    episode = await agent._create_episode(body, rollout_id="stock-smoke")
                except (RuntimeError, ValueError):
                    if not args.misbehave_final:
                        raise
                    assert len(calls) == expected_turns, calls
                    assert calls[-1]["request"]["tool_choice"] == "none"
                    assert not (output / "unexpected-final-tool").exists()
                    summary = {
                        "stock_runtime_smoke": "passed",
                        "malformed_final_tool_blocked": True,
                        "extra_model_call_blocked": True,
                        "model": "scripted",
                        "turns": len(calls),
                    }
                    (output / "passed.json").write_text(json.dumps(summary, indent=2))
                    print(json.dumps(summary), flush=True)
                    return
            assert not args.misbehave_final, "Malformed final response must reject the rollout"

            result = episode.response.model_dump(mode="json")
            (output / "result.json").write_text(json.dumps(result, indent=2))
            assert len(calls) == expected_turns
            final_request = calls[-1]["request"]
            normalized = normalize_opencode_final_instruction(final_request)
            assert normalized is final_request, "Plugin must normalize the final prompt before Gym"
            assert final_request["tool_choice"] == "none"
            assert final_request["messages"][-1]["role"] == "user"
            assert "final answer inside" in final_request["messages"][-1]["content"]
            assert final_request["tools"] == calls[0]["request"]["tools"]
            previous_output = {
                **calls[-2]["response"],
                "tool_calls": [
                    {key: value for key, value in tool.items() if key != "index"}
                    for tool in calls[-2]["response"]["tool_calls"]
                ],
            }
            expected = assistant_fingerprint(calls[-2]["request"]["messages"] + [previous_output])
            assert assistant_fingerprint(normalized["messages"]) == expected
            assert final_answer(result["output"]) == "3"
            assert result["opencode_evidence"]["tool_reward"]["bash_success_count"] == 1
            assert result["opencode_evidence"]["tool_reward"]["tool_error_count"] == 0
            images = [
                part
                for item in result["output"]
                if item["type"] == "function_call_output" and isinstance(item["output"], list)
                for part in item["output"]
                if part["type"] == "input_image"
            ]
            assert len(images) == (0 if args.early_final else 1)
            system = "\n".join(m["content"] for m in calls[0]["request"]["messages"] if m["role"] == "system")
            prefix = cfg.system_prompt
            assert prefix and prefix not in system
            user_text = "\n".join(
                message["content"]
                if isinstance(message["content"], str)
                else "\n".join(part["text"] for part in message["content"] if part["type"] == "text")
                for message in calls[0]["request"]["messages"]
                if message["role"] == "user"
            )
            assert user_text.count(prefix) == 1
            assert prefix + "\n\nCreate an image with bash, read it, then answer 3." in user_text
            if args.default_prompt:
                assert system.startswith(Path(args.default_prompt).read_text().strip())
            assert all(c["request"]["temperature"] == c["request"]["top_p"] == 1 for c in calls)
            assert any("image_url" in str(message) for message in calls[-1]["request"]["messages"])
            summary = {
                "stock_runtime_smoke": "passed",
                "opencode_version": "1.17.11",
                "model": "scripted",
                "training_tested": False,
                "turns": expected_turns,
                "early_final": args.early_final,
                "final_turn_lineage_normalization": True,
                "tool_images": len(images),
                "stock_default_with_visual_user_prefix": True,
                "assistant_correlation_headers": [
                    {
                        key: value
                        for key, value in c["headers"].items()
                        if key.lower() == "x-opencode-assistant-message-id"
                    }
                    for c in calls
                ],
            }
            (output / "PASS.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary))
    finally:
        (output / "calls.json").write_text(json.dumps(calls, indent=2))
        await http.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencode-bin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--default-prompt", help="Optional stock source prompt path for unchanged-prefix verification")
    parser.add_argument("--early-final", action="store_true")
    parser.add_argument("--misbehave-final", action="store_true")
    asyncio.run(main(parser.parse_args()))
