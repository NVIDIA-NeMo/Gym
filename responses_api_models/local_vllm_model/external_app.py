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
# limitations under the License.
"""Entrypoint for a local_vllm_model server whose vLLM runs elsewhere, which needs no Ray.

Gym starts Ray for a server whose class declares `ray_enabled = True`, as `app.py` does to launch vLLM on Ray.
When `base_url` points at an external vLLM endpoint, nothing is launched,
so this entrypoint serves the same model server without Ray.
Select it with `entrypoint: external_app.py`.
It accepts every `app.py` config key,
so a config that switches between a local launch and an external endpoint
only needs to change `entrypoint` and `base_url`.
"""

from typing import Any, ClassVar

from fastapi import FastAPI
from pydantic import Field, field_validator

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.local_vllm_model.server_config import GetInnerVLLMConfigResponse, LocalVLLMModelConfig
from responses_api_models.vllm_model.app import VLLMModel


class LocalVLLMExternalModelConfig(LocalVLLMModelConfig):
    """LocalVLLMModelConfig for an external endpoint: `base_url` is required and the launch settings are optional."""

    base_url: str | list[str]
    # Only `app.py` launches vLLM.
    # These settings are accepted here so the same config block works with either entrypoint.
    vllm_serve_kwargs: dict[str, Any] = Field(default_factory=dict)
    vllm_serve_env_vars: dict[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _require_base_url(cls, base_url: str | list[str]) -> str | list[str]:
        if not base_url:
            raise ValueError(
                "external_app.py serves an external vLLM endpoint and needs `base_url`. "
                "To launch vLLM locally, use `entrypoint: app.py`."
            )
        return base_url


class LocalVLLMExternalModel(VLLMModel):
    """The local_vllm_model server for a vLLM endpoint that runs elsewhere."""

    # Gym reads this from this file's source before the server starts, so it is declared here, not inherited.
    ray_enabled = False
    non_generating_model_routes: ClassVar[frozenset[tuple[str, str]]] = VLLMModel.non_generating_model_routes | {
        ("GET", "/get_inner_vllm_config")
    }
    config: LocalVLLMExternalModelConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Lets LocalVLLMModelProxy find the endpoint, as with app.py.
        app.get("/get_inner_vllm_config")(self.get_inner_vllm_config)
        return app

    async def get_inner_vllm_config(self) -> GetInnerVLLMConfigResponse:
        return GetInnerVLLMConfigResponse(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
        )


if __name__ == "__main__":
    LocalVLLMExternalModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = LocalVLLMExternalModel.run_webserver()  # noqa: F401
