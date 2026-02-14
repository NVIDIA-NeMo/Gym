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
GenRM Response API Model (local vLLM server).

Same GenRM behavior (response_1/response_2/principle roles) as genrm_model.remote,
but extends LocalVLLMModel: downloads the model and starts a vLLM server (e.g. via Ray).
Use when you want to run GenRM against a vLLM instance managed by this process.
"""

from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.genrm_model.remote.app import GenRMModelMixin
from responses_api_models.local_vllm_model.app import (
    LocalVLLMModel,
    LocalVLLMModelConfig,
)


class LocalGenRMModelConfig(LocalVLLMModelConfig):
    """Configuration for GenRM with a locally managed vLLM server.

    Extends LocalVLLMModelConfig with GenRM-specific options.
    """

    supports_principle_role: bool = True


class LocalGenRMModel(GenRMModelMixin, LocalVLLMModel):
    """GenRM Response API Model (local vLLM server).

    Same as GenRMModel but extends LocalVLLMModel: downloads the model, starts
    a vLLM server (e.g. via Ray), and uses GenRM message formatting for
    response_1/response_2/principle roles.
    """

    config: LocalGenRMModelConfig


if __name__ == "__main__":
    LocalGenRMModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = LocalGenRMModel.run_webserver()  # noqa: F401
