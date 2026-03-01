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

config_paths="responses_api_models/local_vllm_model/configs/openai/gpt-oss-20b-reasoning-high.yaml"
ng_run "+config_paths=[${config_paths}]" \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=1 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=16 \
    ++gpt-oss-20b-reasoning-high.responses_api_models.local_vllm_model.vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY=span
