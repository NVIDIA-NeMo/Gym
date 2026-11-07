# Source: https://gitlab-master.nvidia.com/pritamb/agentvista
# Revision: 06f6a8f1532766d876415757107a11731b6148ae (Apache-2.0 adapter files).
# Reference: https://gitlab-master.nvidia.com/pritamb/saas-bench
# Base revision: ad48579c1af66bb7100501f45460bb3e9ec5eaae (Apache-2.0).
# Source: integrations/gym/responses_api_models/vllm_model/saas_training.py; local reference file SHA-256:
# 0e81dbc01f656c957fe7d22a68c446008ca6090913a8d9dce867c3de7fff1a0b
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain served completion identities for AgentVista on-policy terminal attribution."""

import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path

from fastapi import Body, Request

from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.server_utils import get_global_config_dict, is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.agentvista_opencode import AgentVistaOpenCodeModel


class AgentVistaTrainingModel(AgentVistaOpenCodeModel):
    """Use Gym's generation/capture implementation without retokenizing outputs."""

    async def chat_completions(
        self,
        request: Request,
        body: NeMoGymChatCompletionCreateParamsNonStreaming = Body(),  # noqa: B008 -- FastAPI dependency
    ) -> NeMoGymChatCompletion:
        rollout_id = current_rollout_id()
        if not rollout_id:
            raise ValueError("AgentVista training model requires rollout correlation")
        root = Path(get_global_config_dict()["agentvista_training_capture_root"])
        request_path = root / (hashlib.sha256(rollout_id.encode()).hexdigest() + ".request.json")
        sampling = json.loads(request_path.read_text())
        if sampling["rollout_id"] != rollout_id:
            raise ValueError("Training sampling belongs to a different rollout")
        body = body.model_copy(update=sampling["sampling"])
        response = await super().chat_completions(request, body)
        payload = response.model_dump(mode="json", exclude_unset=True)

        def append():
            root.mkdir(parents=True, exist_ok=True)
            path = root / (hashlib.sha256(rollout_id.encode()).hexdigest() + ".jsonl")
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(
                    json.dumps(
                        {
                            "rollout_id": rollout_id,
                            "request": body.model_dump(mode="json", exclude_unset=True),
                            "response": payload,
                        }
                    )
                    + "\n"
                )

        await asyncio.to_thread(append)
        return response


if __name__ == "__main__":
    AgentVistaTrainingModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaTrainingModel.run_webserver()
