# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select by continuation likelihood and grade with the existing MCQA verifier."""

from time import time
from uuid import uuid4

from fastapi import Request
from pydantic import Field

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import get_response_json, raise_for_status
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentRunRequest, SimpleAgentVerifyResponse
from responses_api_models.vllm_loglikelihood.app import LogLikelihoodResponse


class MultipleChoiceRunRequest(SimpleAgentRunRequest):
    choices: list[str] = Field(min_length=2, max_length=26)


class MultipleChoiceAgent(SimpleAgent):
    """Select the option with the highest continuation likelihood."""

    async def run(self, request: Request, body: MultipleChoiceRunRequest) -> SimpleAgentVerifyResponse:
        inputs = body.responses_create_params.input
        if isinstance(inputs, str):
            context = inputs
        elif len(inputs) == 1 and inputs[0].role == "user" and isinstance(inputs[0].content, str):
            context = inputs[0].content
        else:
            raise ValueError("Multiple-choice likelihood requires a string or one user text message")
        cookies = dict(request.cookies)
        seeded = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(mode="json"),
            cookies=cookies,
        )
        await raise_for_status(seeded)
        cookies.update(dict(seeded.cookies or {}))
        scored = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/loglikelihood",
            json={"context": context, "continuations": body.choices},
            cookies=cookies,
        )
        await raise_for_status(scored)
        cookies.update(dict(scored.cookies or {}))
        likelihood = LogLikelihoodResponse.model_validate(await get_response_json(scored))
        if [score.continuation for score in likelihood.scores] != body.choices:
            raise ValueError("Model returned missing, reordered, or substituted choices")
        selected = max(range(len(body.choices)), key=lambda i: likelihood.scores[i].logprob)
        selected_letter = chr(ord("A") + selected)
        response = NeMoGymResponse.model_validate(
            {
                "id": str(uuid4()),
                "created_at": time(),
                "model": likelihood.model,
                "object": "response",
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "metadata": {"evaluation_method": "conditional_loglikelihood", "not_a_generated_answer": "true"},
                "output": [
                    {
                        "id": str(uuid4()),
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": f"\\boxed{{{selected_letter}}}", "annotations": []}
                        ],
                    }
                ],
            }
        )
        payload = body.model_dump(mode="json") | {
            "response": response.model_dump(mode="json"),
            "likelihood": likelihood.model_dump(mode="json"),
            "selected_choice": selected,
        }
        if self.config.skip_verification:
            payload.update(reward=float(self.config.skip_verification_reward), verification_skipped=True)
        else:
            verified = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=payload,
                cookies=cookies,
            )
            await raise_for_status(verified)
            payload.update(await get_response_json(verified))
        return SimpleAgentVerifyResponse.model_validate(payload)


if __name__ == "__main__":  # pragma: no cover
    MultipleChoiceAgent.run_webserver()
