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


"""Check judge and grader credentials on the submitting host, before any GPU is allocated.

A benchmark's `prepare` and `run` mappings name a remote endpoint as a `<prefix>_base_url` key with a
`<prefix>_api_key` sibling (`judge_base_url` and `judge_api_key`, say). The key value is either a literal or a
`${oc.env:NAME}` reference to a variable the job receives through `driver.env` or `compute.forward_env`.
Each such pair gets one `GET <base_url>/models` with the key. A missing or empty key, or a 401/403 answer, fails the
submit; a network failure or any other status only warns, since the submitting host may have no route to the
endpoint. A key that cannot be read on this host (a `runtime:` value unset here, say) is reported as not probed.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import requests

from nemo_gym.config.config_types import ConfigError
from nemo_gym.orchestration.api import RUNTIME_ENV_PREFIX, SlurmComputeConfig, SubmitConfig


logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 10.0
_BASE_URL_SUFFIX = "_base_url"
_API_KEY_SUFFIX = "_api_key"
# The policy is served by this job, so its endpoint and key are not a remote credential.
_POLICY_PREFIX = "policy"
_ENV_REFERENCE = re.compile(r"^\$\{oc\.env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?P<default>,.*)?\}$")


class CredentialPreflightError(ConfigError):
    """A judge or grader credential is missing or was rejected by its endpoint."""


@dataclass(frozen=True)
class _Endpoint:
    base_url: str
    # What a message calls the credential: an environment variable name, or the config key for a literal.
    label: str
    # None when the credential cannot be read on this host.
    key: str | None
    problem: str | None = None


def _first_url(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str) and value.startswith(("http://", "https://")) and "${" not in value:
        return value
    return None


def _credential(
    benchmark: str, section: str, key: str, raw: Any, config: SubmitConfig
) -> tuple[str, str | None, str | None]:
    """The label, the key value readable on this host, and a problem when the credential is missing."""
    match = _ENV_REFERENCE.match(raw) if isinstance(raw, str) else None
    if match is None:
        label = f"{benchmark}.{section}.{key}"
        return (label, raw, None) if isinstance(raw, str) and raw else (label, None, "is empty")

    name = match.group("name")
    compute = next(iter(config.compute.values()))
    forwarded = compute.forward_env if isinstance(compute, SlurmComputeConfig) else []
    if name in config.driver.env:
        value = config.driver.env[name]
        if value.startswith(RUNTIME_ENV_PREFIX):
            value = os.environ.get(value[len(RUNTIME_ENV_PREFIX) :])
            if value is None:
                return name, None, None
    elif name in forwarded:
        value = os.environ.get(name)
    else:
        # The job never receives it. A default means the benchmark expects that; without one the job fails later.
        return name, None, None if match.group("default") is not None else "is not passed to the job"
    return (name, value, None) if value else (name, None, "is unset or empty")


def _endpoints(config: SubmitConfig) -> list[_Endpoint]:
    endpoints: dict[tuple[str, str], _Endpoint] = {}
    for benchmark, run_config in config.driver.benchmarks.items():
        for section, values in (("prepare", run_config.prepare), ("run", run_config.run)):
            for key, raw_url in values.items():
                if not key.endswith(_BASE_URL_SUFFIX):
                    continue
                prefix = key[: -len(_BASE_URL_SUFFIX)]
                key_name = prefix + _API_KEY_SUFFIX
                if prefix == _POLICY_PREFIX or key_name not in values:
                    continue
                base_url = _first_url(raw_url)
                if base_url is None:
                    continue
                label, value, problem = _credential(benchmark, section, key_name, values[key_name], config)
                endpoints.setdefault((base_url, label), _Endpoint(base_url, label, value, problem))
    return list(endpoints.values())


def _probe(endpoint: _Endpoint) -> None:
    url = endpoint.base_url.rstrip("/") + "/models"
    try:
        response = requests.get(
            url, headers={"Authorization": f"Bearer {endpoint.key}"}, timeout=PROBE_TIMEOUT_SECONDS
        )
    except requests.RequestException as error:
        logger.warning(
            "Credential preflight could not reach %s (%s); %s was not checked.",
            endpoint.base_url,
            type(error).__name__,
            endpoint.label,
        )
        return
    if response.status_code in (401, 403):
        raise CredentialPreflightError(
            f"Endpoint {endpoint.base_url} rejected {endpoint.label} (HTTP {response.status_code}). "
            "Set a valid credential, or pass --skip-credential-preflight to submit anyway."
        )
    if not response.ok:
        logger.warning(
            "Credential preflight: %s answered HTTP %s to GET /models; %s was not checked.",
            endpoint.base_url,
            response.status_code,
            endpoint.label,
        )


def check_credentials(config: SubmitConfig) -> None:
    """Raise CredentialPreflightError for a missing or rejected credential of a remote endpoint of `config`."""
    endpoints = _endpoints(config)
    for endpoint in endpoints:
        if endpoint.problem is not None:
            raise CredentialPreflightError(
                f"Credential {endpoint.label} for endpoint {endpoint.base_url} {endpoint.problem}. "
                "Set it, or pass --skip-credential-preflight to submit anyway."
            )
    for endpoint in endpoints:
        if endpoint.key is None:
            logger.info(
                "Credential preflight: %s for %s not probed (not readable here).", endpoint.label, endpoint.base_url
            )
        else:
            _probe(endpoint)
