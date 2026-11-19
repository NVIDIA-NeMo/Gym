# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""The server, run and worker labels on every sandbox, and closing every sandbox that carries given labels.

``AsyncSandbox.start`` adds the labels to every sandbox it creates.
With them, a live worker can stop the sandboxes of a worker that died, by its routing ID.
Otherwise they would run until the provider's TTL.
"""

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from nemo_gym.core.worker_resources import worker_routing_id
from nemo_gym.sandbox.attribution import resolve_run_id, sanitize_label_value
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxProvider, SupportsSandboxCloseByMetadata


LOGGER = logging.getLogger(__name__)

SERVER_LABEL = "nemo-gym-server"
RUN_LABEL = "nemo-gym-run"
WORKER_LABEL = "nemo-gym-worker"

# The gym CLI sets it on every server process to the server instance's name.
# It is nemo_gym.config.global_config.NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
# not imported here to keep the sandbox package free of the config package.
_SERVER_NAME_ENV_VAR = "NEMO_GYM_CONFIG_PATH"
_CLOSE_CONCURRENCY = 16


def worker_labels() -> dict[str, str]:
    """The labels of a sandbox this worker creates: run and worker always, server when the gym CLI started it.

    Values are valid Kubernetes label values already, so every provider stores them unchanged.
    A later ``close_by_metadata`` with the same values therefore matches.
    """
    labels = {RUN_LABEL: sanitize_label_value(resolve_run_id()), WORKER_LABEL: worker_routing_id()}
    server = os.environ.get(_SERVER_NAME_ENV_VAR, "").strip()
    if server:
        labels[SERVER_LABEL] = sanitize_label_value(server)
    return labels


@dataclass(frozen=True)
class CloseByMetadataResult:
    """What ``close_by_metadata`` did, by sandbox ID.

    Attributes:
        closed: Sandboxes whose provider ``close`` returned.
            OpenShell's close hands the delete to a worker thread and returns.
            There it means the delete was handed off.
            A delete that later fails is only logged.
        refused: Sandboxes the provider listed whose labels no longer matched when re-read, left running.
        failed: Sandboxes whose re-read or close raised; a caller that owes their release retries the call.
        gone: Sandboxes the provider listed that no longer existed when re-read.
    """

    closed: tuple[str, ...] = ()
    refused: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    gone: tuple[str, ...] = ()


async def close_by_metadata(provider: SandboxProvider, labels: Mapping[str, str]) -> CloseByMetadataResult:
    """Close every sandbox of ``provider`` that carries all of ``labels``.

    Before each close it re-reads the sandbox's labels from the provider.
    It leaves the sandbox running unless every label still matches.
    A provider's listing may be stale or its filter advisory, and a stop is not undone.

    Every sandbox that carries all of ``labels`` is closed.
    Labels that name no worker, such as a run or a server alone, therefore also stop the sandboxes of live workers.
    To end one dead worker's sandboxes, include ``WORKER_LABEL``.

    Raises:
        ValueError: ``labels`` is empty or has an empty value; that would select sandboxes nobody named.
        NotImplementedError: The provider cannot find sandboxes by label.
    """
    wanted = dict(labels)
    if not wanted or not all(wanted.values()):
        raise ValueError(f"close_by_metadata needs at least one label and no empty value, got {wanted!r}")
    if not isinstance(provider, SupportsSandboxCloseByMetadata):
        provider_name = getattr(provider, "name", type(provider).__name__)
        raise NotImplementedError(f"Sandbox provider {provider_name!r} cannot close sandboxes by label")

    handles = await provider.list_by_metadata(wanted)
    limit = asyncio.Semaphore(_CLOSE_CONCURRENCY)
    closed: list[str] = []
    refused: list[str] = []
    failed: list[str] = []
    gone: list[str] = []

    async def close_if_still_labeled(handle: SandboxHandle) -> None:
        async with limit:
            try:
                current = await provider.metadata_of(handle)
                if current is None:
                    gone.append(handle.sandbox_id)
                    return
                if any(current.get(key) != value for key, value in wanted.items()):
                    LOGGER.warning(
                        "Not closing sandbox %s: its labels changed since it was listed (wanted %s, found %s)",
                        handle.sandbox_id,
                        wanted,
                        current,
                    )
                    refused.append(handle.sandbox_id)
                    return
                await provider.close(handle)
            except Exception as error:
                LOGGER.warning("Could not close sandbox %s by label %s: %r", handle.sandbox_id, wanted, error)
                failed.append(handle.sandbox_id)
                return
            closed.append(handle.sandbox_id)

    await asyncio.gather(*(close_if_still_labeled(handle) for handle in handles))
    return CloseByMetadataResult(closed=tuple(closed), refused=tuple(refused), failed=tuple(failed), gone=tuple(gone))
