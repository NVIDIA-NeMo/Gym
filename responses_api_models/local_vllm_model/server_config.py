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
"""Configuration shared by both local_vllm_model entrypoints.

Kept apart from `app.py`, which imports Ray and vLLM, so that `external_app.py` can load it without either.
"""

import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from responses_api_models.vllm_model.app import VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    # We inherit these configs from VLLMModelConfig, but they are set to optional since they will be set later on after we spin up a model endpoint.
    base_url: str | list[str] = Field(default_factory=list)
    # Not used on local deployments
    api_key: str = "dummy"  # pragma: allowlist secret

    hf_home: str | None = None
    vllm_serve_kwargs: dict[str, Any]
    vllm_serve_env_vars: dict[str, str]

    ray_worker_py_executable: str = sys.executable

    show_vllm_engine_stats: bool = False
    debug: bool = False

    def model_post_init(self, context):
        # Default to the .cache/huggingface in this directory.
        if not self.hf_home:
            current_directory = Path.cwd()
            self.hf_home = str(current_directory / ".cache" / "huggingface")

        return super().model_post_init(context)


class GetInnerVLLMConfigResponse(BaseModel):
    base_url: list[str]
    api_key: str
    model: str
