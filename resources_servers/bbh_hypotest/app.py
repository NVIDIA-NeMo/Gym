# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from fastapi import Request
from pydantic import Field

from resources_servers.aviary.hypotest_app import (
    HypotestResourcesServer as AviaryHypotestResourcesServer,
)
from resources_servers.aviary.schemas import (
    AviaryAgentVerifyRequest,
    AviaryAgentVerifyResponse,
    AviaryCloseRequest,
    AviaryCloseResponse,
)


logger = logging.getLogger(__name__)


class BBHHypotestResourcesServer(AviaryHypotestResourcesServer):
    env_id_to_result_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)

    async def verify(self, request: Request, body: AviaryAgentVerifyRequest) -> AviaryAgentVerifyResponse:
        response = await super().verify(request, body)
        env_id = body.response.env_id
        env = self.env_id_to_env.get(env_id)
        metadata = env.get_result_metadata() if env is not None else self.env_id_to_result_metadata.pop(env_id, {})
        if not metadata:
            return response

        payload = response.model_dump()
        payload.update(metadata)
        if metadata.get("rubric_model_failed"):
            instance_config = dict(payload.get("instance_config") or {})
            instance_config["mask_sample"] = True
            instance_config["agent_error_kind"] = "rubric_model"
            payload["instance_config"] = instance_config
        return AviaryAgentVerifyResponse(**payload)

    async def close(self, request: Request, body: AviaryCloseRequest) -> AviaryCloseResponse:
        env = self.env_id_to_env.get(body.env_id)
        if env is not None:
            try:
                self.env_id_to_result_metadata[body.env_id] = env.get_result_metadata()
            except Exception:
                logger.exception("Failed to collect Hypotest result metadata before close")
        return await super().close(request, body)


if __name__ == "__main__":
    BBHHypotestResourcesServer.run_webserver()
