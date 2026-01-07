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

from responses_api_models.local_vllm_model.app import LocalVLLMModelConfig


class TestApp:
    def test_sanity_vllm_import(self) -> None:
        import vllm

        assert vllm.__version__

    def test_sanity_config_init(self) -> None:
        LocalVLLMModelConfig(
            host="",
            port=0,
            entrypoint="",
            name="test name",
            model="test model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            vllm_serve_kwargs=dict(),
        )
