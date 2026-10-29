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
"""Loading and accessing the run config, and the server config types."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.config.config_types import (
        ROLLOUT_PATH_PREFIX,
        TOKEN_CAPTURE_PATH_SEGMENT,
        AgentServerRef,
        AggregateMetrics,
        AggregateMetricsRequest,
        BaseServerConfig,
        BenchmarkDatasetConfig,
        DatasetConfig,
        Domain,
        DownloadJsonlDatasetGitlabConfig,
        EnvironmentServerRef,
        ModelServerRef,
        ResourcesServerRef,
    )
    from nemo_gym.config.global_config import (
        AGENT_REF_KEY_NAME,
        ATTEMPT_INDEX_KEY_NAME,
        CACHE_DIR_KEY_NAME,
        DISALLOWED_PORTS_KEY_NAME,
        HF_TOKEN_KEY_NAME,
        OBSERVABILITY_ENABLED_KEY_NAME,
        RESULTS_DIR_KEY_NAME,
        ROLLOUT_ID_KEY_NAME,
        ROLLOUT_INDEX_KEY_NAME,
        SKILLS_REF_KEY_NAME,
        SKIP_VERIFICATION_KEY_NAME,
        SKIP_VERIFICATION_REWARD_KEY_NAME,
        TASK_INDEX_KEY_NAME,
        TASK_SOURCE_KEY_NAME,
        TOKEN_ID_CAPTURE_BLOCK,
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        OmegaConf,
        find_open_port,
        get_first_server_config_dict,
        get_global_config_dict,
        get_hf_token,
        maybe_get_global_config_dict,
    )
    from nemo_gym.config.server_metadata import ServerMetadata, visit_agent_server, visit_resources_server


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AGENT_REF_KEY_NAME": "global_config",
    "ATTEMPT_INDEX_KEY_NAME": "global_config",
    "AgentServerRef": "config_types",
    "AggregateMetrics": "config_types",
    "AggregateMetricsRequest": "config_types",
    "BaseServerConfig": "config_types",
    "BenchmarkDatasetConfig": "config_types",
    "CACHE_DIR_KEY_NAME": "global_config",
    "DISALLOWED_PORTS_KEY_NAME": "global_config",
    "DatasetConfig": "config_types",
    "Domain": "config_types",
    "DownloadJsonlDatasetGitlabConfig": "config_types",
    "EnvironmentServerRef": "config_types",
    "GlobalConfigDictParser": "global_config",
    "GlobalConfigDictParserConfig": "global_config",
    "HF_TOKEN_KEY_NAME": "global_config",
    "ModelServerRef": "config_types",
    "OBSERVABILITY_ENABLED_KEY_NAME": "global_config",
    "OmegaConf": "global_config",
    "RESULTS_DIR_KEY_NAME": "global_config",
    "ROLLOUT_ID_KEY_NAME": "global_config",
    "ROLLOUT_INDEX_KEY_NAME": "global_config",
    "ROLLOUT_PATH_PREFIX": "config_types",
    "ResourcesServerRef": "config_types",
    "SKILLS_REF_KEY_NAME": "global_config",
    "SKIP_VERIFICATION_KEY_NAME": "global_config",
    "SKIP_VERIFICATION_REWARD_KEY_NAME": "global_config",
    "ServerMetadata": "server_metadata",
    "TASK_INDEX_KEY_NAME": "global_config",
    "TASK_SOURCE_KEY_NAME": "global_config",
    "TOKEN_CAPTURE_PATH_SEGMENT": "config_types",
    "TOKEN_ID_CAPTURE_BLOCK": "global_config",
    "find_open_port": "global_config",
    "get_first_server_config_dict": "global_config",
    "get_global_config_dict": "global_config",
    "get_hf_token": "global_config",
    "maybe_get_global_config_dict": "global_config",
    "visit_agent_server": "server_metadata",
    "visit_resources_server": "server_metadata",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
