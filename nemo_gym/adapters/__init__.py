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
"""Gym adapter framework — interceptor-based middleware for responses_api_models.

Public API:
    install_middleware(app, interceptor_specs)  — attach pipeline to a FastAPI app
    start_adapter_proxy(upstream_url, adapters) — host pipeline as a localhost uvicorn
    AdapterPipeline                             — the async interceptor chain
    InterceptorRegistry                         — name → class resolution
"""

from nemo_gym.adapters.middleware import install_middleware
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.proxy import ProxyHandle, start_adapter_proxy
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterProxyConfig,
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
    InterceptorSpec,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
    Stage,
)


__all__ = [
    "AdapterPipeline",
    "AdapterProxyConfig",
    "AdapterRequest",
    "AdapterResponse",
    "GracefulError",
    "InterceptorContext",
    "InterceptorRegistry",
    "InterceptorSpec",
    "ProxyHandle",
    "RequestInterceptor",
    "RequestToResponseInterceptor",
    "ResponseInterceptor",
    "Stage",
    "install_middleware",
    "start_adapter_proxy",
]
