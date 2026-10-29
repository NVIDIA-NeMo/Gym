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
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.secret_utils import hide_secrets_in_container
from nemo_gym.path_utils import resolved_config_path_for


def write_resolved_config(global_config_dict: Mapping[str, Any], output_fpath: Path) -> Path:
    """Save the run's config beside its rollouts file as `<stem>_resolved_config.yaml`.

    Secret-shaped values are masked and interpolations stay as `${...}` strings, so the file is
    safe to share and records what was asked for rather than what the environment resolved to.
    It replaces any earlier attempt's file atomically.
    """
    path = resolved_config_path_for(output_fpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    container = (
        OmegaConf.to_container(global_config_dict, resolve=False)
        if isinstance(global_config_dict, DictConfig)
        else dict(global_config_dict)
    )
    masked = hide_secrets_in_container(container)
    text = OmegaConf.to_yaml(OmegaConf.create(masked))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
