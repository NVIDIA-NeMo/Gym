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
"""Identity checks for resuming saved evaluation results.

The manifest stores digests, not configuration values or credentials. Recovery reuses
completed rollouts; it does not checkpoint an agent's conversation or remote sandbox.
"""

import hashlib
import json
import os
import stat
import warnings
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.config_types import ConfigError


# Collection/output choices can change between invocations without changing a task.
_COLLECTION_OPTIONS = frozenset(
    {
        "resume_from_cache",
        "allow_unsafe_resume",
        "output_jsonl_fpath",
        "input_jsonl_fpath",
        "num_samples_in_parallel",
        "disable_aggregation",
        "disable_health_check",
        "health_check_workers",
        "health_check_ignored_checks",
        "upload_rollouts",
        "count_failure_classes_as_zero",
        "route_failures_to_sidecar",
    }
)
# Server addresses and credentials are resolved anew when a job restarts. Keep model
# names, prompts, generation settings and environment configuration in the digest.
_SERVER_RUNTIME_OPTIONS = frozenset(
    {
        "host",
        "port",
        "api_key",
        "openai_api_key",
        "policy_api_key",
        "base_url",
        "openai_base_url",
        "policy_base_url",
        "head_server",
        "disallowed_ports",
        "port_range_low",
        "port_range_high",
        "model_call_capture_dir",
    }
)
_SERVER_KINDS = ("responses_api_agents", "responses_api_models", "resources_servers")


def _plain(value: Any) -> Any:
    """Pydantic's Python dump may still contain nested Hydra containers."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def manifest_path_for(output: Path) -> Path:
    return output.with_name(output.stem + "_manifest.json")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _without_runtime_options(value: Any) -> Any:
    # Exclude only known configuration boundaries. Recursively dropping names such
    # as "port" would also erase meaningful tool schemas or environment parameters.
    result = {k: v for k, v in value.items() if k not in _SERVER_RUNTIME_OPTIONS and k not in _COLLECTION_OPTIONS}

    def settings_identity(settings):
        if not isinstance(settings, dict):
            return settings
        settings = {k: v for k, v in settings.items() if k not in _SERVER_RUNTIME_OPTIONS}
        if isinstance(settings.get("token_id_capture"), dict):
            settings["token_id_capture"] = {k: v for k, v in settings["token_id_capture"].items() if k != "dir"}
        return settings

    for name, block in list(result.items()):
        if not isinstance(block, dict):
            continue
        updated = dict(block)
        for kind in _SERVER_KINDS:
            if kind in block and isinstance(block[kind], dict):
                updated[kind] = {
                    implementation: settings_identity(settings) for implementation, settings in block[kind].items()
                }
        result[name] = updated
    return result


def _configuration_identity(config: dict, global_config: dict, rows: list[dict]) -> dict:
    # Resolve only the servers reachable from the materialized agents. Keep the
    # original root as interpolation context, but do not evaluate unused servers
    # (which may require credentials/environment variables unavailable here).
    raw = _plain(global_config)
    context = OmegaConf.create(raw)
    filtered = _without_runtime_options(raw)
    servers = {
        name for name, block in raw.items() if isinstance(block, dict) and any(k in block for k in _SERVER_KINDS)
    }
    pending = {row.get("agent_ref", {}).get("name") for row in rows} - {None}
    pending.update(row["task_source"] for row in rows if isinstance(row.get("task_source"), str))
    if not pending:
        pending = set(servers)
    selected = {}

    def references(value):
        if isinstance(value, dict):
            if isinstance(value.get("name"), str) and value["name"] in servers:
                pending.add(value["name"])
            for item in value.values():
                references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)

    while pending:
        name = pending.pop()
        if name in selected or name not in servers:
            continue
        context[name] = filtered[name]
        selected[name] = OmegaConf.to_container(context[name], resolve=True)
        references(selected[name])
    # Collection overrides may themselves contain DictConfig/ListConfig values.
    collection = {key: _plain(value) for key, value in config.items() if key not in _COLLECTION_OPTIONS}
    context["_collection_identity"] = collection
    capture = raw.get("token_id_capture", {})
    context["_capture_identity"] = {key: value for key, value in capture.items() if key != "dir"}
    return {
        "collection": OmegaConf.to_container(context._collection_identity, resolve=True),
        "servers": selected,
        "token_id_capture": OmegaConf.to_container(context._capture_identity, resolve=True),
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        # Exclusive creation honors the caller's umask. Rewrites retain existing
        # sharing permissions instead of inheriting NamedTemporaryFile's 0600.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            if path.exists():
                os.fchmod(file.fileno(), stat.S_IMODE(path.stat().st_mode))
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    # Selection is a run policy, not an inference from payload arrival order.
    selection_policy: Literal["latest_dispatched"] = "latest_dispatched"
    run_id: str
    source_digest: str
    materialized_digest: str
    config_digest: str
    legacy_import: bool = False
    identity_overridden: bool = False

    @classmethod
    def create(cls, source: Path, rows: list[dict], config: dict, global_config: dict) -> "RunManifest":
        return cls(
            run_id=uuid4().hex,
            source_digest=_file_digest(source),
            materialized_digest=_digest(rows),
            config_digest=_digest(_configuration_identity(config, global_config, rows)),
        )

    def write(self, path: Path) -> None:
        # Publish a complete manifest, or leave the previous one intact.
        atomic_write_json(path, self.model_dump(mode="json"))

    @classmethod
    def import_legacy(cls, rows: list[dict]) -> "RunManifest":
        return cls(
            run_id=uuid4().hex,
            source_digest="unverified",
            materialized_digest=_digest(rows),
            config_digest="unverified",
            legacy_import=True,
        )


def validate_resume(
    path: Path,
    current: RunManifest | None,
    materialized_path: Path,
    *,
    allow_unsafe: bool = False,
) -> RunManifest | None:
    """Reject unknown or incompatible saved work before any output is modified.

    ``allow_unsafe`` is an explicit compatibility escape hatch for legacy artifacts or
    deliberate identity overrides. Unsupported/corrupt manifests remain errors.
    """
    if not path.exists():
        problem = "Saved rollouts have no run manifest; their input/configuration identity cannot be verified."
        saved = None
    else:
        try:
            saved = RunManifest.model_validate_json(path.read_bytes())
            with materialized_path.open(encoding="utf-8") as file:
                saved_rows = [json.loads(line) for line in file if line.strip()]
        except (ValidationError, ValueError, OSError) as error:
            raise ConfigError(f"Cannot read recovery manifest or materialized inputs: {error}") from error
        mismatches = []
        if saved.legacy_import:
            mismatches.append("unverified legacy run identity")
        if saved.identity_overridden:
            mismatches.append("previously overridden run identity")
        if saved.materialized_digest != _digest(saved_rows):
            mismatches.append("saved materialized inputs")
        if current is None:
            mismatches.append("current input/configuration identity")
        else:
            for field, description in (
                ("source_digest", "source dataset"),
                ("materialized_digest", "current materialized inputs"),
                ("config_digest", "resolved configuration"),
            ):
                if getattr(saved, field) != getattr(current, field):
                    mismatches.append(description)
        if not mismatches:
            return saved
        problem = "Cannot resume with incompatible " + ", ".join(mismatches) + "."
    if not allow_unsafe:
        raise ConfigError(problem + " Start a fresh run or explicitly set allow_unsafe_resume=true.")
    warnings.warn(problem + " Continuing because allow_unsafe_resume=true; saved inputs will be reused.", stacklevel=2)
    return saved.model_copy(update={"identity_overridden": True}) if saved is not None else None
