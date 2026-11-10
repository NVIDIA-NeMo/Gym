# Source: https://gitlab-master.nvidia.com/pritamb/agentvista
# Revision: 06f6a8f1532766d876415757107a11731b6148ae (Apache-2.0 adapter files).
# Reference: https://gitlab-master.nvidia.com/pritamb/saas-bench
# Base revision: ad48579c1af66bb7100501f45460bb3e9ec5eaae (Apache-2.0).
# Source: integrations/gym/responses_api_agents/saas_bench_agent/training.py; local reference file SHA-256:
# aa026a9c4350e9779f3f02aed82e5b554bfdd92ddaedfa65aa3e300ae52d6392
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-only agent: original answer-judge reward plus witnessed terminal capture."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.agentvista_agent.app import AgentVistaAgent, AgentVistaConfig


class AgentVistaTrainingConfig(AgentVistaConfig):
    harness: Literal["opencode"] = "opencode"
    training_capture_root: str


class AgentVistaTrainingAgent(AgentVistaAgent):
    config: AgentVistaTrainingConfig

    async def _episode(self, body, output, fingerprint, rollout_id):
        # The trainer stamps sampling on the Gym row. OpenCode's own defaults
        # must not replace it when the external process calls the model server.
        params = body.responses_create_params
        sampling = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_output_tokens,
        }
        if any(value is None for value in sampling.values()):
            raise ValueError("Trainer must provide temperature, top_p and max_output_tokens")
        root = Path(self.config.training_capture_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / (hashlib.sha256(rollout_id.encode()).hexdigest() + ".request.json")
        payload = {"rollout_id": rollout_id, "sampling": sampling}
        try:
            with path.open("x") as stream:
                json.dump(payload, stream)
        except FileExistsError:
            if json.loads(path.read_text()) != payload:
                raise ValueError("Rollout sampling changed during retry") from None
        return await super()._episode(body, output, fingerprint, rollout_id)

    def _response(self, body, result):
        from responses_api_agents.agentvista_agent.runtime.training import read_calls, terminal_response_id

        response = super()._response(body, result)
        rollout_id = maybe_rollout_id_from_run_body(body)
        if not rollout_id:
            raise ValueError("Training episode has no rollout ID")
        path = Path(self.config.training_capture_root) / (hashlib.sha256(rollout_id.encode()).hexdigest() + ".jsonl")
        try:
            terminal = terminal_response_id(response.native_trajectory, read_calls(path), rollout_id)
        except (OSError, ValueError, KeyError) as error:
            response.mask_sample = True
            response.reward = 0.0
            response.failure_kind = "agentvista:training_capture"
            response.failure_reason = str(error)
        else:
            response.terminal_response_id = terminal
        return response


if __name__ == "__main__":
    AgentVistaTrainingAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaTrainingAgent.run_webserver()
