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
"""Entrypoint for a genrm_model server whose GenRM vLLM pool runs elsewhere, which needs no Ray.

Gym starts Ray for a server whose class declares `ray_enabled = True`, as `app.py` does to launch vLLM on Ray. When
`base_url` points at an external GenRM vLLM pool, nothing is launched, so this entrypoint serves the same GenRM model
server without Ray: select it with `entrypoint: external_app.py`. It accepts every `app.py` config key, so a config
that switches between a local launch and an external pool only needs to change `entrypoint` and `base_url`.
"""

from typing import Any

from fastapi import Request

from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.genrm_model.common import GenRMModelConfig, append_genrm_roles
from responses_api_models.local_vllm_model.external_app import LocalVLLMExternalModel, LocalVLLMExternalModelConfig


# LocalVLLMExternalModelConfig comes first so its required `base_url` and optional launch settings win over the
# local-launch defaults that GenRMModelConfig inherits.
class GenRMExternalModelConfig(LocalVLLMExternalModelConfig, GenRMModelConfig):
    """GenRMModelConfig for an external pool: `base_url` is required and the launch settings are optional."""


class GenRMExternalModel(LocalVLLMExternalModel):
    """The genrm_model server for a GenRM vLLM pool that runs elsewhere."""

    # Gym reads this from this file's source before the server starts, so it is declared here, not inherited.
    ray_enabled = False

    config: GenRMExternalModelConfig

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: dict[str, Any]) -> dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        return append_genrm_roles(body_dict, supports_principle_role=self.config.supports_principle_role)


if __name__ == "__main__":
    GenRMExternalModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = GenRMExternalModel.run_webserver()  # noqa: F401
