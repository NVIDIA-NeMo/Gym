# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from responses_api_agents.browser_agent.adapters.base import BaseCUAAdapter, CUAAdapterResponse
from responses_api_agents.browser_agent.adapters.openai_adapter import OpenAICUAAdapter

_ADAPTER_REGISTRY: dict[str, type[BaseCUAAdapter]] = {
    "openai": OpenAICUAAdapter,
}

try:
    from responses_api_agents.browser_agent.adapters.anthropic_adapter import AnthropicCUAAdapter

    _ADAPTER_REGISTRY["anthropic_sonnet"] = AnthropicCUAAdapter
    _ADAPTER_REGISTRY["anthropic_opus"] = AnthropicCUAAdapter
except ImportError:
    pass

try:
    from responses_api_agents.browser_agent.adapters.gemini_adapter import GeminiCUAAdapter

    _ADAPTER_REGISTRY["gemini"] = GeminiCUAAdapter
except ImportError:
    pass


class AdapterFactory:
    @staticmethod
    def create(adapter_type: str, **kwargs) -> BaseCUAAdapter:
        if adapter_type not in _ADAPTER_REGISTRY:
            available = ", ".join(sorted(_ADAPTER_REGISTRY.keys()))
            raise ValueError(f"Unknown adapter type '{adapter_type}'. Available: {available}")
        return _ADAPTER_REGISTRY[adapter_type](**kwargs)

    @staticmethod
    def available_adapters() -> list[str]:
        return sorted(_ADAPTER_REGISTRY.keys())
