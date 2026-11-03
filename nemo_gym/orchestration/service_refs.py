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
"""`service:` references in a benchmark's `run` config.

`service:NAME` as the value of a server type replaces the whole server; `service:NAME.field` fills one
value. Each service describes its own fields through an `accessors(name)` method; this module only walks
the config and substitutes them.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


SERVICE_REF_PREFIX = "service:"
# Server type -> the (url, model, key) field names that `service:NAME` fills in.
SERVER_TYPE_FIELDS = {
    "vllm_model": ("base_url", "model", "api_key"),
    "openai_model": ("openai_base_url", "openai_model", "openai_api_key"),
}
SERVICE_FIELDS = ("url", "host", "port", "model", "key")
# Placeholder for clients that require a key; neither vLLM nor a declarative service supplies one.
PLACEHOLDER_KEY = "dummy"  # pragma: allowlist secret


def service_host_env_var(service_name: str) -> str:
    """Driver env var holding a pinned service's hostname, known only once the job has nodes."""
    return f"GYM_SERVICE_{re.sub(r'[^A-Z0-9]', '_', service_name.upper())}_HOST"


@dataclass(frozen=True)
class ServiceAccessors:
    """A service's named fields as strings. A pinned service's host is only known once the job has nodes,
    so it is an OmegaConf reference to a driver env var and `host_deferred` is set."""

    url: str | list[str]
    host: str
    port: str
    model: str
    key: str = PLACEHOLDER_KEY
    host_deferred: bool = False


def resolve_service_refs(run: dict[str, Any], services: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    """`run` with its `service:` references resolved, and the services whose deferred host it used.

    The result is a new dict: the input may be shared with the caller, who may keep it as a record of
    what was submitted. `services` maps names to configs; those that serve a model have `accessors(name)`.
    """
    deferred: set[str] = set()
    return _resolve(run, services, deferred), deferred


def check_host_var_collisions(deferred: set[str]) -> None:
    """Reject two services that would be exported to the driver under the same variable."""
    by_var: dict[str, str] = {}
    for name in sorted(deferred):
        var = service_host_env_var(name)
        if var in by_var:
            raise ValueError(
                f"'{SERVICE_REF_PREFIX}{by_var[var]}' and '{SERVICE_REF_PREFIX}{name}' are both pinned services "
                f"whose hosts would be exported as {var}. Rename one so the names differ in more than punctuation."
            )
        by_var[var] = name


def _resolve(run: dict[str, Any], services: Mapping[str, Any], deferred: set[str]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in run.items():
        if isinstance(value, dict):
            resolved[key] = _resolve(value, services, deferred)
        elif isinstance(value, str) and value.startswith(SERVICE_REF_PREFIX):
            name, _, field = value[len(SERVICE_REF_PREFIX) :].partition(".")
            accessors = _accessors(name, services)
            if accessors.host_deferred:
                deferred.add(name)
            if field:
                if field not in SERVICE_FIELDS:
                    raise ValueError(f"'{value}': unknown field '{field}' (one of {', '.join(SERVICE_FIELDS)}).")
                resolved[key] = getattr(accessors, field)
            elif key in SERVER_TYPE_FIELDS:
                url_field, model_field, key_field = SERVER_TYPE_FIELDS[key]
                resolved[key] = {url_field: accessors.url, model_field: accessors.model, key_field: accessors.key}
            else:
                raise ValueError(
                    f"'{value}' is under '{key}', which is not a server type "
                    f"({', '.join(SERVER_TYPE_FIELDS)}); name a field instead, like '{value}.url'."
                )
        else:
            resolved[key] = value
    return resolved


def _accessors(name: str, services: Mapping[str, Any]) -> ServiceAccessors:
    if name not in services:
        raise ValueError(f"'{SERVICE_REF_PREFIX}{name}' does not match any service ({', '.join(sorted(services))}).")
    service = services[name]
    if not hasattr(service, "accessors"):
        raise ValueError(f"'{SERVICE_REF_PREFIX}{name}' is a {service.type} service, which does not serve a model.")
    return service.accessors(name)
