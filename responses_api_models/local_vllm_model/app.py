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
from typing import Any, Dict, Optional

from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class LocalVLLMModelConfig(VLLMModelConfig):
    vllm_spinup_command: str
    vllm_spinup_command_template_kwargs: Optional[Dict[str, Any]] = None

    def model_post_init(self, context):
        # base_url and api_key are set later in the model spinup
        self.base_url = ""
        self.api_key = ""
        return super().model_post_init(context)


class LocalVLLMModel(VLLMModel):
    config: LocalVLLMModelConfig

    def model_post_init(self, context):
        return super().model_post_init(context)


if __name__ == "__main__":
    LocalVLLMModel.run_webserver()
