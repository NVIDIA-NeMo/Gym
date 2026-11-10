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
"""
labbench2 VLM agent — extends simple_agent to embed media at rollout time.

At ``run()`` time, resolves ``verifier_metadata.media_dir`` references in the
incoming request, reads the corresponding image/PDF files from disk, and
either injects ``input_image`` blocks (default) or extracted PDF text as
``input_text`` when ``media_mode=text`` **and** the row's ``verifier_metadata.tag``
is ``protocolqa2`` (so mixed JSONLs can use one run: text for protocols, images
for figqa2/tableqa2). Non-protocol rows always use the image path when
``media_mode=text`` is set.

An agent session gets the same media from its ``agent_context`` (``artifacts.media`` and ``values.tag``), the
``agent`` role's context in the task row, and embeds it in the episode it creates; the agent loop stays Simple Agent's.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.agents import AgentSeedSessionRequest, resolve_agent_context_artifact
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import AgentContextArtifact
from nemo_gym.trajectory import ObservationGap, TrajectoryRecord
from resources_servers.labbench2_vlm.prepare_data import embed_media_into_row
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
    SimpleAgentSessionState,
    SimpleAgentVerifyResponse,
)


class LabbenchVLMAgentConfig(SimpleAgentConfig):
    media_base_dir: str = Field(
        description="Base directory for resolving verifier_metadata.media_dir references, relative to Gym root.",
    )
    dpi: int = Field(default=170, description="DPI for PDF page rendering.")
    media_mode: str = Field(
        default="image",
        description=(
            "'image': always render PDFs/images as input_image blocks. "
            "'text': extract PDF text only for verifier_metadata.tag protocolqa2; "
            "all other tags still use images (for mixed example/benchmark JSONLs)."
        ),
    )
    strip_images_from_output: bool = Field(
        default=True,
        description="Remove base64 input_image blocks from serialized rollout artifacts.",
    )


def _effective_media_mode(agent_media_mode: str, verifier_metadata: dict | None) -> str:
    """Map agent config to per-row embed mode: ``text`` applies only to protocolqa2."""
    if agent_media_mode == "text":
        tag = str((verifier_metadata or {}).get("tag", ""))
        if tag.startswith("protocolqa2"):
            return "text"
    return "image"


def _scrub_image_blocks(value: Any) -> tuple[Any, bool]:
    """``value`` without its input_image blocks, and whether it had any."""
    removed_image = False

    def scrub(value: Any) -> Any:
        nonlocal removed_image
        if isinstance(value, list):
            cleaned = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "input_image":
                    removed_image = True
                    continue
                cleaned.append(scrub(item))
            return cleaned
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        return value

    return scrub(value), removed_image


def _strip_image_blocks(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    """Remove input_image blocks from the serialized rollout result.

    Operates on a dict dump to avoid Pydantic model mutation/serialization issues,
    then re-validates into the response model.
    """
    data, removed_image = _scrub_image_blocks(result.model_dump(mode="json"))
    if removed_image:
        for key in ("ng_trajectory", "ng_agent_observations"):
            observations = data.get(key)
            if isinstance(observations, dict):
                observations.setdefault("gaps", []).append({"code": "multimodal_history_redacted"})
    return SimpleAgentVerifyResponse.model_validate(data)


def _strip_trajectory_images(trajectory: TrajectoryRecord) -> TrajectoryRecord:
    """The trajectory without the base64 images its conversations carry, with a gap that says so."""
    data, removed_image = _scrub_image_blocks(trajectory.model_dump(mode="json"))
    stripped = TrajectoryRecord.model_validate(data)
    if removed_image:
        stripped.gaps.append(ObservationGap(code="multimodal_history_redacted"))
    return stripped


class LabbenchVLMAgentContext(BaseModel):
    """The ``agent_context`` a labbench2 task carries: its media directory and its subtask tag.

    The tag is required: it picks text or image media for protocolqa2 under ``media_mode=text``, so a row without it
    would reach the model with a different prompt than on /run.
    """

    model_config = ConfigDict(extra="forbid")

    media: AgentContextArtifact
    tag: str


@dataclass
class LabbenchVLMSessionState(SimpleAgentSessionState):
    # The task's media directory, resolved under media_base_dir when the session is seeded.
    media_dir: Path | None = None
    tag: str = ""


class LabbenchVLMAgent(SimpleAgent):
    config: LabbenchVLMAgentConfig
    agent_context_model = LabbenchVLMAgentContext

    def _media_base_dir(self) -> Path:
        base = Path(self.config.media_base_dir)
        return base if base.is_absolute() else PARENT_DIR / base

    def _new_session_state(self, body: AgentSeedSessionRequest) -> LabbenchVLMSessionState:
        # A restore rebuilds sessions through here too, so a restored session keeps its media.
        state = super()._new_session_state(body)
        context = self.parse_agent_context(body)
        media_dir = resolve_agent_context_artifact(context.media, self._media_base_dir(), kind="directory")
        return LabbenchVLMSessionState(
            **{field.name: getattr(state, field.name) for field in fields(state)}, media_dir=media_dir, tag=context.tag
        )

    async def _create_session_episode(
        self, state: SimpleAgentSessionState, body: NeMoGymResponseCreateParamsNonStreaming, **kwargs: Any
    ) -> tuple[NeMoGymResponse, TrajectoryRecord | None, Any, Any]:
        if isinstance(state, LabbenchVLMSessionState) and state.media_dir is not None:
            # The same media for the same request, so a retried activation sends the model the same input.
            row = {
                "responses_create_params": body.model_dump(mode="json", exclude_unset=True),
                "verifier_metadata": {"media_dir": state.media_dir.name},
            }
            enriched = embed_media_into_row(
                row,
                state.media_dir.parent,
                dpi=self.config.dpi,
                media_mode=_effective_media_mode(self.config.media_mode, {"tag": state.tag}),
            )
            body = NeMoGymResponseCreateParamsNonStreaming.model_validate(enriched["responses_create_params"])
        model_response, trajectory, model_cookies, resources_cookies = await super()._create_session_episode(
            state, body, **kwargs
        )
        # A session's trajectory reaches its observations and the rollout record; keep the base64 images out of both.
        if trajectory is not None and self.config.strip_images_from_output:
            trajectory = _strip_trajectory_images(trajectory)
        return model_response, trajectory, model_cookies, resources_cookies

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        resolved_base = Path(self.config.media_base_dir)
        if not resolved_base.is_absolute():
            resolved_base = PARENT_DIR / resolved_base

        row_dict = body.model_dump(exclude_unset=True)
        meta = row_dict.get("verifier_metadata")
        embed_mode = _effective_media_mode(self.config.media_mode, meta if isinstance(meta, dict) else None)

        enriched = embed_media_into_row(
            row_dict,
            resolved_base,
            dpi=self.config.dpi,
            media_mode=embed_mode,
        )
        body = SimpleAgentRunRequest.model_validate(enriched)

        result = await super().run(request, body)

        if self.config.strip_images_from_output:
            result = _strip_image_blocks(result)

        return result


if __name__ == "__main__":
    LabbenchVLMAgent.run_webserver()
