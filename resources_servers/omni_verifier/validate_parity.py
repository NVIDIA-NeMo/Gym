#!/usr/bin/env python3
"""Validate that the Gym omni verifier returns NeMo RL mmpr rewards."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from nemo_rl.environments.mmpr_filtered_reward import mmpr_filtered_reward
from resources_servers.omni_verifier.app import (
    OmniVerifierResourcesServer,
    OmniVerifierResourcesServerConfig,
    OmniVerifierVerifyRequest,
)


def _response(text: str, reasoning: str | None = None) -> NeMoGymResponse:
    output = []
    if reasoning:
        output.append(
            {
                "id": "rs_test",
                "summary": [{"text": reasoning, "type": "summary_text"}],
                "type": "reasoning",
            }
        )
    output.append(
        {
            "id": "msg_test",
            "content": [
                {
                    "annotations": [],
                    "text": text,
                    "type": "output_text",
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    )
    return NeMoGymResponse(
        id="resp_test",
        created_at=0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


async def main() -> None:
    server = OmniVerifierResourcesServer(
        config=OmniVerifierResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="omni_verifier",
        ),
        server_client=MagicMock(spec=ServerClient),
    )

    cases = [
        ("think:multiple-choice:D", "<think>x</think> \\boxed{D}", None, "<think>x</think> \\boxed{D}"),
        ("think:multiple-choice:D", "\\boxed{D}", "x", "<think>x</think>\n\\boxed{D}"),
        ("nothink:string-match:cat", "\\boxed{cat}", None, "\\boxed{cat}"),
        ("think:python-list:[1, 2]", "<think>x</think> \\boxed{[1,2]}", None, "<think>x</think> \\boxed{[1,2]}"),
        ("think:mathruler:2", "<think>x</think> \\boxed{2}", None, "<think>x</think> \\boxed{2}"),
        ("think:asr:hello world", "hello world", None, "hello world"),
    ]

    for ground_truth, response, reasoning, expected_response in cases:
        req = OmniVerifierVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Q?"}],
                "parallel_tool_calls": False,
                "temperature": 0,
            },
            response=_response(response, reasoning),
            ground_truth=ground_truth,
            dynamic_format_reward=True,
            format_weight=0.1,
            asr_reward_min=-1.0,
        )
        gym_result = await server.verify(req)
        nemo_reward, nemo_correct = mmpr_filtered_reward(
            ground_truth,
            expected_response,
            dynamic_format_reward=True,
            format_weight=0.1,
            asr_reward_min=-1.0,
        )
        assert gym_result.reward == float(nemo_reward), (
            ground_truth,
            gym_result.reward,
            nemo_reward,
        )
        assert gym_result.is_correct == bool(nemo_correct), (
            ground_truth,
            gym_result.is_correct,
            nemo_correct,
        )
        verifier_name = ground_truth.split(":", 1)[1].split(":", 1)[0]
        print(
            f"[OMNI_VERIFIER_PARITY] {verifier_name} "
            f"reward={gym_result.reward:.4f} is_correct={gym_result.is_correct}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
