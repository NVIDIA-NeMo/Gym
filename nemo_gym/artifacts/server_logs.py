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
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.config.config_types import maybe_get_server_instance_config
from nemo_gym.config.global_config import (
    NEMO_GYM_LOG_DIR_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    SERVICE_LOG_DIR_ENV,
)
from nemo_gym.utils.path_utils import server_logs_path_for


SERVER_LOGS_SCHEMA_VERSION = 1
NO_LOG_LOCATION_REASON = (
    "No server log location is known for this run: Gym servers log to a file only when nemo_gym_log_dir is set, "
    "and `gym eval submit` jobs record their service logs."
)


def server_log_locations(global_config_dict: Mapping[str, Any], environ: Mapping[str, str] = os.environ) -> dict:
    """Where this run's model and Gym server logs are, as far as this process knows; unknown locations are absent.

    `gym_servers` are the files the servers this config starts tee their output to when
    `nemo_gym_log_dir` is set (`<dir>/<server>.log`, as `gym env start` writes them); other `*.log`
    files already in that folder are listed apart as `unverified_log_dir_files`, since a shared folder
    also holds earlier runs' logs, and do not make the locations available on their own. `services` are the per-step logs of a
    `gym eval submit` job (vLLM and the other services, the driver, the collector), listed from the
    folder submit exports, for the current Slurm job.
    """
    locations: dict[str, Any] = {"schema_version": SERVER_LOGS_SCHEMA_VERSION}
    log_dir = global_config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME)
    if log_dir:
        names = [
            name
            for name, value in global_config_dict.items()
            if name not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
            and maybe_get_server_instance_config(name=name, server_type_config_dict=value)[0] is not None
        ]
        root = Path(log_dir).expanduser().resolve()
        # The servers this config starts; their files may not exist yet.
        configured = {name.replace("/", "_"): str(root / f"{name.replace('/', '_')}.log") for name in names}
        if configured:
            locations["gym_servers"] = dict(sorted(configured.items()))
        # Other logs already in the folder: a `--no-serve` run's config names no servers, while the `gym env start`
        # serving it may log there. A shared folder also holds earlier runs' logs, so these are only candidates.
        if root.is_dir():
            others = {path.stem: str(path) for path in root.glob("*.log") if path.stem not in configured}
            if others:
                locations["unverified_log_dir_files"] = dict(sorted(others.items()))
    service_dir = environ.get(SERVICE_LOG_DIR_ENV, "").strip()
    job_id = environ.get("SLURM_JOB_ID", "").strip()
    if service_dir and job_id:
        suffix = f"-{job_id}.log"
        files = sorted(Path(service_dir).glob(f"*{suffix}"))
        # A step whose log appears after this snapshot is still found by the pattern.
        locations["services"] = {
            "dir": service_dir,
            "slurm_job_id": job_id,
            "file_pattern": f"<step>-{job_id}.log",
            "files": {path.name[: -len(suffix)]: str(path) for path in files if re.fullmatch(r"[\w.-]+", path.name)},
        }
    locations["status"] = "available" if ("gym_servers" in locations or "services" in locations) else "unavailable"
    if locations["status"] == "unavailable":
        locations["reason"] = NO_LOG_LOCATION_REASON
    return locations


def write_server_log_locations(global_config_dict: Mapping[str, Any], output_fpath: Path) -> Path:
    """Save where the run's server logs are beside its rollouts file as `<stem>_server_logs.json`.

    Lets the validation gate and EvalRDL find serving evidence such as vLLM logs. Paths only; no
    log content is copied. It replaces any earlier attempt's file atomically.
    """
    path = server_logs_path_for(output_fpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(server_log_locations(global_config_dict), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
