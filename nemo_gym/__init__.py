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
import sys
from os import environ
from pathlib import Path


# /path/to/dir/Gym (PARENT_DIR)
# |- cache (CACHE_DIR)
# |- results (RESULTS_DIR)
# |- nemo_gym (ROOT_DIR)
# |- responses_api_models
# |- responses_api_agents
# ...
ROOT_DIR = Path(__file__).absolute().parent
PARENT_DIR = ROOT_DIR.parent

# Editable install: PARENT_DIR is the repo root (has pyproject.toml)
# Wheel install: PARENT_DIR is site-packages/ so use cwd instead
_is_editable_install = (PARENT_DIR / "pyproject.toml").exists()
WORKING_DIR = PARENT_DIR if _is_editable_install else Path.cwd()

CACHE_DIR = WORKING_DIR / "cache"
RESULTS_DIR = WORKING_DIR / "results"

sys.path.append(str(PARENT_DIR))


def _resolve_under_cwd_or_install(path) -> Path:
    """Resolve a possibly-relative path for *reading* a built-in or user-supplied file.

    Absolute paths are returned unchanged. A relative path is tried first under the current working
    directory (the user's project), then under the Gym install root (``PARENT_DIR``) where built-in
    assets live in both editable and wheel installs. This mirrors ``config_paths`` resolution, so a
    repo-relative path like ``resources_servers/<env>/data/example.jsonl`` resolves by name from any
    cwd. If neither exists the cwd candidate is returned so error messages point at the user's cwd.

    Use this for read paths only — never for write targets (e.g. metrics written next to a dataset),
    which must stay relative to the user's writable cwd rather than the install root.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    install_candidate = PARENT_DIR / p
    if install_candidate.exists():
        return install_candidate
    return cwd_candidate


# TODO: Maybe eventually we want an override for OMP_NUM_THREADS ?

# Turn off HF tokenizers paralellism
environ["TOKENIZERS_PARALLELISM"] = "false"

# Huggingface related caching directory overrides to local folders.
# Only override if not already set by the user.
if "HF_DATASETS_CACHE" not in environ:
    environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "huggingface")
if "HF_HOME" not in environ:
    environ["HF_HOME"] = str(CACHE_DIR / "huggingface")


OLD_PRINT = print


def print_always_flushes(*args, **kwargs) -> None:
    kwargs["flush"] = True
    OLD_PRINT(*args, **kwargs)


__builtins__["print"] = print_always_flushes


from nemo_gym.package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)


# Public API surface.
#
# This is the supported entry point for building environments: import the base classes and core
# types from ``nemo_gym`` directly (e.g. ``from nemo_gym import SimpleResourcesServer``) rather than
# from internal module paths like ``nemo_gym.base_resources_server``. The internal module layout is
# not a public contract and may change; this top-level surface is what we keep stable.
#
# Symbols are resolved lazily via ``__getattr__`` (PEP 562): nothing below is imported until first
# accessed. This keeps ``import nemo_gym`` cheap (it does not eagerly pull in FastAPI/pydantic/openai
# and the whole server stack) and avoids import cycles, since several of these modules import from
# this package during their own initialization.
#
# Maps public name -> submodule it lives in. The deep import paths continue to work unchanged.
_LAZY_EXPORTS: dict[str, str] = {
    # Resources server base classes and request/response models
    "BaseResourcesServer": "base_resources_server",
    "BaseResourcesServerConfig": "base_resources_server",
    "SimpleResourcesServer": "base_resources_server",
    "BaseRunRequest": "base_resources_server",
    "BaseVerifyRequest": "base_resources_server",
    "BaseVerifyResponse": "base_resources_server",
    "BaseSeedSessionRequest": "base_resources_server",
    "BaseSeedSessionResponse": "base_resources_server",
    "MCPResourcesServer": "base_resources_server",
    "MCPServerMetadata": "base_resources_server",
    "MCPSessionError": "base_resources_server",
    "gym_tool": "base_resources_server",
    # Agent base classes
    "BaseResponsesAPIAgent": "base_responses_api_agent",
    "BaseResponsesAPIAgentConfig": "base_responses_api_agent",
    "SimpleResponsesAPIAgent": "base_responses_api_agent",
    # Model base classes
    "BaseResponsesAPIModel": "base_responses_api_model",
    "BaseResponsesAPIModelConfig": "base_responses_api_model",
    "SimpleResponsesAPIModel": "base_responses_api_model",
    # Server plumbing shared across all server types
    "BaseServer": "server_utils",
    "SimpleServer": "server_utils",
    "ServerClient": "server_utils",
    "request": "server_utils",
    "raise_for_status": "server_utils",
    "get_response_json": "server_utils",
    "get_server_url": "server_utils",
    "SESSION_ID_KEY": "server_utils",
    # Config base types
    "BaseServerConfig": "config_types",
    "BaseRunServerInstanceConfig": "config_types",
    "Domain": "config_types",
    "AggregateMetrics": "config_types",
    "AggregateMetricsRequest": "config_types",
    # OpenAI-compatible schema (Responses + Chat Completions)
    "NeMoGymAsyncOpenAI": "openai_utils",
    "NeMoGymResponse": "openai_utils",
    "NeMoGymResponseCreateParamsNonStreaming": "openai_utils",
    "NeMoGymResponseInput": "openai_utils",
    "NeMoGymResponseInputItem": "openai_utils",
    "NeMoGymResponseInputText": "openai_utils",
    "NeMoGymResponseOutputItem": "openai_utils",
    "NeMoGymResponseOutputMessage": "openai_utils",
    "NeMoGymResponseOutputText": "openai_utils",
    "NeMoGymResponseOutputRefusal": "openai_utils",
    "NeMoGymResponseReasoningItem": "openai_utils",
    "NeMoGymResponseFunctionToolCall": "openai_utils",
    "NeMoGymResponseUsage": "openai_utils",
    "NeMoGymEasyInputMessage": "openai_utils",
    "NeMoGymMessage": "openai_utils",
    "NeMoGymFunctionToolParam": "openai_utils",
    "NeMoGymFunctionCallOutput": "openai_utils",
    "NeMoGymChatCompletion": "openai_utils",
    "NeMoGymChatCompletionCreateParamsNonStreaming": "openai_utils",
    "NeMoGymChatCompletionMessage": "openai_utils",
    "NeMoGymChatCompletionMessageParam": "openai_utils",
    "NeMoGymChatCompletionToolParam": "openai_utils",
}

# Eagerly-defined names (path constants + package metadata) that live directly in this module.
_EAGER_PUBLIC: tuple[str, ...] = (
    "ROOT_DIR",
    "PARENT_DIR",
    "WORKING_DIR",
    "CACHE_DIR",
    "RESULTS_DIR",
    "__version__",
    "__package_name__",
)

__all__ = sorted({*_EAGER_PUBLIC, *_LAZY_EXPORTS})


def __getattr__(name: str):
    # PEP 562 module-level __getattr__: only called for names not already in the module globals,
    # so eagerly-defined symbols and previously-resolved lazy ones never reach here.
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    # Cache on the package so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
