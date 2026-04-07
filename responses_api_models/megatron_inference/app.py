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
from typing import List, Tuple

from pydantic import BaseModel

from nemo_gym.openai_utils import RESPONSES_TO_TRAIN
from nemo_gym.server_utils import is_nemo_gym_fastapi_worker
from responses_api_models.vllm_model.app import VLLMConverter, VLLMModel


class MegatronMetadataMixin(BaseModel):
    policy_epoch: List[List[Tuple[int, int]]]
    kv_cache_epoch: List[List[Tuple[int, int]]]
    num_evictions: List[int]


MEGATRON_RESPONSES_TO_TRAIN = {
    base: type(f"Megatron{train.__name__}", (train, MegatronMetadataMixin), {})
    for base, train in RESPONSES_TO_TRAIN.items()
}


class MegatronInferenceConverter(VLLMConverter):
    ...


class MegatronInferenceModel(VLLMModel):
    def model_post_init(self, context):
        super().model_post_init(context)
        self._converter = MegatronInferenceConverter(
            return_token_id_information=self.config.return_token_id_information,
        )


if __name__ == "__main__":
    MegatronInferenceModel.run_webserver()
elif is_nemo_gym_fastapi_worker():
    app = MegatronInferenceModel.run_webserver()  # noqa: F401
