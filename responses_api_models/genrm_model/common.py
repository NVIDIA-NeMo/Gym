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
"""GenRM request handling and configuration shared by both genrm_model entrypoints.

Kept apart from `app.py`, which imports Ray and vLLM, so that `external_app.py` can load it without either.
"""

from typing import Any

from responses_api_models.local_vllm_model.server_config import LocalVLLMModelConfig


def append_genrm_roles(body_dict: dict[str, Any], *, supports_principle_role: bool) -> dict[str, Any]:
    """Append the GenRM comparison roles that the resources server passes in `metadata`.

    The resources server passes the comparison payload in `metadata`, so the `input` field carries only the
    conversation history and the request schema stays generic. This appends, in order:

    - `metadata["principle"]` as a `"principle"` message, only when `supports_principle_role` is true;
    - `metadata["response_1"]` as a `"response_1"` message;
    - `metadata["response_2"]` as a `"response_2"` message.

    `metadata` is consumed here and not forwarded to vLLM.
    """
    metadata = body_dict.pop("metadata", None) or {}
    response_1 = metadata.get("response_1")
    response_2 = metadata.get("response_2")
    principle = metadata.get("principle")

    messages = body_dict["messages"]

    if supports_principle_role and principle:
        messages.append({"role": "principle", "content": principle})

    # Candidate responses currently support text only; the user message may contain images.
    if response_1 is not None:
        messages.append({"role": "response_1", "content": response_1})

    if response_2 is not None:
        messages.append({"role": "response_2", "content": response_2})

    return body_dict


class GenRMModelConfig(LocalVLLMModelConfig):
    """Configuration for GenRM with a locally managed vLLM server."""

    supports_principle_role: bool = True
