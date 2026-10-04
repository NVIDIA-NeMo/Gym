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
import asyncio
from typing import Any

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.mimo_music.setup_abc2midi import ensure_abc2midi


class MimoMusicConfig(BaseResourcesServerConfig):
    pass


class MimoMusicRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MimoMusicVerifyRequest(MimoMusicRunRequest, BaseVerifyRequest):
    pass


class MimoMusicVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    abc_found: bool


def _final_text(response: Any) -> str:
    texts = []
    for item in response.output or []:
        if getattr(item, "type", None) == "message":
            texts.extend(getattr(part, "text", "") or "" for part in item.content or [])
    return "\n".join(texts)


class MimoMusicResourcesServer(SimpleResourcesServer):
    """MiMo-V2.6-RL-oss music: ABC notation scored for human-likeness of the rendered MIDI."""

    ray_enabled = False
    config: MimoMusicConfig

    def model_post_init(self, context: Any, /) -> None:
        ensure_abc2midi()

    async def verify(self, body: MimoMusicVerifyRequest) -> MimoMusicVerifyResponse:
        from resources_servers.mimo_music.scorer import compute_score, extract_abc

        text = _final_text(body.response)
        reward = await asyncio.to_thread(compute_score, "music", text)
        return MimoMusicVerifyResponse(**body.model_dump(), reward=reward, abc_found=extract_abc(text) is not None)


if __name__ == "__main__":
    MimoMusicResourcesServer.run_webserver()
