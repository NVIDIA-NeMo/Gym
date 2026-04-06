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

from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponseOutputMessageForTraining

from .app import MegatronTokenIDLogProbMixin


class MegatronRunRequest(BaseRunRequest):
    """Run request that preserves environment-specific dataset fields."""
    model_config = ConfigDict(extra="allow")


class MegatronResponseOutputMessageForTraining(
    NeMoGymResponseOutputMessageForTraining, MegatronTokenIDLogProbMixin
):
    """Training output message with Megatron epoch fields declared for type checking."""
    ...


class MegatronVerifyResponse(BaseVerifyResponse):
    """Verify response from a Megatron inference backend."""
    ...
