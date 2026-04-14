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
incoming request, reads the corresponding image/PDF files from disk,
base64-encodes them, and injects ``input_image`` blocks into the model request
before delegating to the standard simple_agent pipeline.
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


class LabbenchVLMAgent(SimpleAgent):
    config: LabbenchVLMAgentConfig

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        resolved_base = Path(self.config.media_base_dir)
        if not resolved_base.is_absolute():
            resolved_base = PARENT_DIR / resolved_base

        enriched = embed_media_into_row(body.model_dump(), resolved_base, dpi=self.config.dpi)
        body = SimpleAgentRunRequest.model_validate(enriched)

        return await super().run(request, body)


if __name__ == "__main__":
    LabbenchVLMAgent.run_webserver()
