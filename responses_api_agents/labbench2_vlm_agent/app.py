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
"""

from pathlib import Path

from fastapi import Request
from pydantic import Field

from nemo_gym import PARENT_DIR
from resources_servers.labbench2_vlm.prepare_data import embed_media_into_row
from responses_api_agents.simple_agent.app import (
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentRunRequest,
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
        description="Remove base64 input_image blocks from the rollout output to keep files small.",
    )


def _effective_media_mode(agent_media_mode: str, verifier_metadata: dict | None) -> str:
    """Map agent config to per-row embed mode: ``text`` applies only to protocolqa2."""
    if agent_media_mode == "text":
        tag = str((verifier_metadata or {}).get("tag", ""))
        if tag.startswith("protocolqa2"):
            return "text"
    return "image"


def _strip_image_blocks(result: SimpleAgentVerifyResponse) -> SimpleAgentVerifyResponse:
    """Remove input_image blocks from responses_create_params in the output.

    Operates on a dict dump to avoid Pydantic model mutation/serialization issues,
    then re-validates into the response model.
    """
    data = result.model_dump()
    for msg in data.get("responses_create_params", {}).get("input", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = [b for b in content if b.get("type") != "input_image"]
    return SimpleAgentVerifyResponse.model_validate(data)


class LabbenchVLMAgent(SimpleAgent):
    config: LabbenchVLMAgentConfig

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
