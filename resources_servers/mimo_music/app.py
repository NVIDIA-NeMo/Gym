# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from typing import Any

from pydantic import ConfigDict

from nemo_gym import failure_kinds
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
    ray_enabled = False
    config: MimoMusicConfig

    def model_post_init(self, context: Any, /) -> None:
        ensure_abc2midi()

    async def verify(self, body: MimoMusicVerifyRequest) -> MimoMusicVerifyResponse:
        from resources_servers.mimo_music.scorer import Abc2MidiMissing, compute_score, extract_abc

        text = _final_text(body.response)
        found = extract_abc(text) is not None
        try:
            reward = await asyncio.to_thread(compute_score, "music", text)
        except Abc2MidiMissing as e:
            return MimoMusicVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                abc_found=found,
                mask_sample=True,
                failure_kind=failure_kinds.VERIFIER_ERROR,
                failure_reason=str(e),
            )
        return MimoMusicVerifyResponse(**body.model_dump(), reward=reward, abc_found=found)


if __name__ == "__main__":
    MimoMusicResourcesServer.run_webserver()
