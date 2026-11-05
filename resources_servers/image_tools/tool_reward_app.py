# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scores an image tools agent rollout under an Environment Server: the task's answer plus its tool reward.

The agent's own ``/run`` picks the verifier for each row, verifies the final answer and adds the tool reward it
accumulated. Under an Environment Server this resources server does both instead:

- It sends the final assistant message to the verifier the row's ``image_tools_base_agent_ref`` selects, from the
  agent's ``resource_servers_by_agent``. Those verifiers keep no session state, so it does not seed them.
- It recomputes the tool reward by replaying the rollout's assistant turns through the agent's image tool settings,
  with the row's own source images, and checks the replay ran as many tools as the agent reports. A rollout from
  another agent, or one the replay cannot reproduce, fails verification with a terminal 422: running the episode
  again would not change either.
- The reward is the verifier's reward plus the tool reward, at most 1.0.
"""

import asyncio
import tempfile
from typing import Any, Optional

from fastapi import HTTPException
from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AgentServerRef, ResourcesServerRef, get_first_server_config_dict
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


# The package-relative import works under pytest; as a server, Gym runs this file as a script (see app.py).
try:
    from .base import (
        IMAGE_TOOLS_ROLLOUT_KEY,
        ImageToolsEpisode,
        ImageToolsGymToolLogic,
        ImageToolsSettings,
        extract_image_refs,
    )
except ImportError:
    from base import (
        IMAGE_TOOLS_ROLLOUT_KEY,
        ImageToolsEpisode,
        ImageToolsGymToolLogic,
        ImageToolsSettings,
        extract_image_refs,
    )


class ImageToolsRewardConfig(BaseResourcesServerConfig):
    # The image tools agent whose rollouts this server scores: its image tool settings and its
    # resource_servers_by_agent are read from its config.
    agent_server: AgentServerRef


class ImageToolsRewardVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


class ImageToolsRewardVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    base_reward: float
    image_tools_aux_reward: float
    image_tools_call_count: int
    image_tools_error_count: int
    image_tools_output_paths: list[str] = []
    image_tools_generation_image_paths: list[list[str]] = []
    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


def _assistant_turns(response: NeMoGymResponse) -> list[str]:
    """Each model call's assistant text, as the agent read it: the turns are split at the observations it appended."""
    turns: list[list[str]] = [[]]
    for item in response.output:
        role = getattr(item, "role", None)
        if role == "user":
            turns.append([])
            continue
        if getattr(item, "type", None) != "message" or role != "assistant":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            turns[-1].append(content)
        elif isinstance(content, list):
            turns[-1].extend(text for part in content if isinstance(text := getattr(part, "text", None), str))
    return ["\n".join(texts).strip() for texts in turns]


def _final_assistant_response(response: NeMoGymResponse) -> NeMoGymResponse:
    for output_item in reversed(response.output):
        if getattr(output_item, "type", None) == "message" and getattr(output_item, "role", None) == "assistant":
            return response.model_copy(update={"output": [output_item]})
    return response


class ImageToolsRewardServer(SimpleResourcesServer):
    config: ImageToolsRewardConfig
    # Verification only reads the request; the delegated verifiers keep no session state either.
    checkpoint_mode = "stateless"
    _settings: ImageToolsSettings = PrivateAttr()
    _verifiers: dict[str, ResourcesServerRef] = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        agent = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        if isinstance(agent, DictConfig):
            agent = OmegaConf.to_container(agent, resolve=True)
        self._settings = ImageToolsSettings.model_validate(
            {name: agent[name] for name in ImageToolsSettings.model_fields if name in agent}
        )
        self._verifiers = {
            name: ResourcesServerRef.model_validate(dict(ref))
            for name, ref in (agent.get("resource_servers_by_agent") or {}).items()
        }

    async def verify(self, body: ImageToolsRewardVerifyRequest) -> ImageToolsRewardVerifyResponse:
        base_agent_name = (body.image_tools_base_agent_ref or {}).get("name")
        verifier = self._verifiers.get(base_agent_name)
        if verifier is None:
            raise HTTPException(
                422,
                f"No verifier for image_tools_base_agent_ref {base_agent_name!r}; "
                f"the agent's resource_servers_by_agent names {sorted(self._verifiers)}",
            )
        files = (body.response.model_extra or {}).get(IMAGE_TOOLS_ROLLOUT_KEY)
        if not isinstance(files, dict):
            raise HTTPException(
                422, f"The response has no {IMAGE_TOOLS_ROLLOUT_KEY}; this server scores image tools agent rollouts"
            )
        if files.get("agent") != self.config.agent_server.name:
            # The replay uses the settings of the agent this server names; another agent's rollout would be scored
            # with the wrong rewards.
            raise HTTPException(
                422,
                f"The rollout came from agent {files.get('agent')!r}, but this server replays "
                f"{self.config.agent_server.name!r}'s settings",
            )
        # The verifier scores the final answer, as it does for the agent's own /run.
        verify_body = body.model_dump(mode="json") | {
            "response": _final_assistant_response(body.response).model_dump(mode="json")
        }
        verify_response = await self.server_client.post(
            server_name=verifier.name, url_path="/verify", json=verify_body
        )
        await raise_for_status(verify_response)
        result = await get_response_json(verify_response)

        episode = await asyncio.to_thread(self._replay, body)
        tool_call_count = int(episode.metadata.get("tool_call_count", 0))
        reported = files.get("image_tools_output_paths") or []
        if len(reported) != tool_call_count:
            # A replay that ran other tools than the agent did would score a different rollout. The usual cause is a
            # source image this server cannot read, such as a relative path or a file only on the agent's host.
            raise HTTPException(
                422,
                f"Replaying the rollout ran {tool_call_count} image tools, but the agent reports {len(reported)}; "
                "check that this server can read the row's source images",
            )

        base_reward = float(result.get("reward", 0.0))
        return ImageToolsRewardVerifyResponse.model_validate(
            result
            | {
                "base_reward": base_reward,
                "image_tools_aux_reward": episode.aux_reward,
                # Auxiliary rewards add to the base score, capped above at 1.0. Positive bonuses can raise any score
                # below the cap; negative penalties still apply at every base score, including a correct answer's 1.0.
                "reward": min(1.0, base_reward + episode.aux_reward),
                "image_tools_call_count": tool_call_count,
                "image_tools_error_count": int(episode.metadata.get("tool_error_count", 0)),
                "image_tools_output_paths": reported,
                "image_tools_generation_image_paths": files.get("image_tools_generation_image_paths") or [],
                "image_tools_base_agent_ref": body.image_tools_base_agent_ref,
                "response": body.response.model_dump(mode="json"),
            }
        )

    def _replay(self, body: ImageToolsRewardVerifyRequest) -> ImageToolsEpisode:
        """Run the rollout's tool calls again with the agent's settings; the crops go to a directory removed after."""
        with tempfile.TemporaryDirectory(prefix="image_tools_replay_") as crop_dir:
            logic = ImageToolsGymToolLogic(self._settings.model_copy(update={"crop_dir": crop_dir}).tool_config())
            episode = ImageToolsEpisode(logic, extract_image_refs(body.responses_create_params.input))
            for assistant_text in _assistant_turns(body.response):
                if episode.step(assistant_text) is None:
                    break
            return episode


if __name__ == "__main__":
    ImageToolsRewardServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = ImageToolsRewardServer.run_webserver()  # noqa: F401
