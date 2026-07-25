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
"""Gym model server that serves model calls through a Switchyard routing proxy.

Switchyard decides, per request, which upstream model should carry the work. Exposing it as a
Responses API model rather than wiring it into a single agent means every benchmark Gym already
supports can be run against a router without touching harness code::

    gym eval run --benchmark <name> --model-type switchyard_model

The dependency direction is one-way: Gym knows Switchyard, Switchyard does not know Gym. This
server owns the mapping between Gym concepts and Switchyard's generic proxy contract -- notably
Gym's rollout-attempt id becomes Switchyard's opaque session id, so proxy-side routing decisions
and costs join back to the rollout that produced them.
"""

import asyncio
import atexit
import contextvars
import logging
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from typing import Any, Dict, Optional

from pydantic import Field, model_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    find_open_port,
    get_global_config_dict,
)
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


logger = logging.getLogger(__name__)


# Gym's capture middleware strips the /ng-rollout/<id> correlation prefix before routing, so by the
# time a handler runs the rollout id is gone. _RolloutSessionMiddleware runs outside it and
# republishes the id here for the duration of the request.
_ROLLOUT_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "switchyard_model_rollout_id", default=None
)

_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[^/]+)(?P<rest>/.*)$")


class _RolloutSessionMiddleware:
    """Pure-ASGI middleware that publishes the rollout id on a ContextVar.

    Installed last so it sits outside Gym's capture middleware and still sees the correlation
    prefix. The path is forwarded untouched -- stripping it stays the capture middleware's job.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        match = _ROLLOUT_PATH_RE.match(scope.get("path", ""))
        token = _ROLLOUT_ID.set(match.group("rollout_id") if match else None)
        try:
            await self._app(scope, receive, send)
        finally:
            _ROLLOUT_ID.reset(token)


class SwitchyardModelConfig(BaseResponsesAPIModelConfig):
    """Configuration for serving Gym model calls through a Switchyard proxy.

    By default Gym launches the proxy itself from ``routing_profiles``, so a routed eval is one
    command and the user never manages a proxy. Setting ``switchyard_base_url`` instead attaches to
    a proxy someone else runs -- useful when an eval needs to pin a specific Switchyard build, or
    when several servers should share one instance.
    """

    # Set to attach to an already-running proxy instead of launching one.
    switchyard_base_url: Optional[str] = None
    switchyard_api_key: str = "dummy"  # pragma: allowlist secret

    # The route name to request. Switchyard maps it to a concrete provider model, so this is a
    # routing label rather than a model id -- which target actually served the call comes back on
    # the response.
    switchyard_model: str

    # Used when launching. routing_profiles is the routing config Switchyard serves.
    routing_profiles: Optional[str] = None
    # 0.0.0.0 so the proxy is reachable off-box; this server always talks to it over loopback.
    proxy_host: str = "0.0.0.0"
    proxy_port: Optional[int] = None
    proxy_startup_timeout_s: float = 120.0
    switchyard_executable: str = "switchyard"

    # Header Switchyard reads as its opaque session id. Gym sends its rollout-attempt id so
    # per-call routing decisions can be joined back to the rollout after the run.
    session_id_header: str = "proxy_x_session_id"
    forward_session_id: bool = True

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    default_headers: Dict[str, str] = Field(default_factory=dict)

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        description=(
            "Cap on in-flight upstream requests from this server (per-process asyncio.Semaphore). None = unlimited."
        ),
    )

    @property
    def launches_proxy(self) -> bool:
        return self.switchyard_base_url is None

    @model_validator(mode="after")
    def validate_target(self) -> "SwitchyardModelConfig":
        if self.launches_proxy and not self.routing_profiles:
            raise ValueError(
                "switchyard_model needs either routing_profiles (Gym launches the proxy) or "
                "switchyard_base_url (attach to a proxy you run yourself)"
            )
        return self


class SwitchyardModel(SimpleResponsesAPIModel):
    config: SwitchyardModelConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = (
            asyncio.Semaphore(self.config.max_concurrent_requests)
            if self.config.max_concurrent_requests is not None
            else nullcontext()
        )
        # When launching, the address does not exist yet -- setup_webserver builds the client once
        # the proxy is serving.
        if not self.config.launches_proxy:
            self._build_client(self.config.switchyard_base_url)
        return super().model_post_init(context)

    def _build_client(self, base_url: str) -> None:
        self._client = NeMoGymAsyncOpenAI(
            base_url=base_url,
            api_key=self.config.switchyard_api_key,
            default_headers=self.config.default_headers,
        )

    def setup_webserver(self):
        # Launch here rather than in model_post_init so the proxy only starts when this server is
        # actually going to serve -- matching local_vllm_model, and keeping config validation and
        # tests free of side effects.
        if self.config.launches_proxy:
            print("Starting Switchyard proxy...")
            self._build_client(self.start_proxy())

        app = super().setup_webserver()
        # Added last, so it wraps the capture middleware and still sees the correlation prefix.
        app.add_middleware(_RolloutSessionMiddleware)
        return app

    # --- Proxy lifecycle ---

    def start_proxy(self) -> str:
        """Start `switchyard ... serve` and block until it answers /health.

        Switchyard is not a dependency of this package -- launching drives its CLI, so the
        executable has to be on PATH. Check that up front rather than surfacing it as an opaque
        FileNotFoundError from Popen.
        """
        if shutil.which(self.config.switchyard_executable) is None:
            raise RuntimeError(
                f"switchyard_model needs the {self.config.switchyard_executable!r} executable on PATH to "
                "launch a proxy. Install it with `pip install 'nemo-switchyard[server]'`, or set "
                "switchyard_base_url to attach to a proxy you run yourself."
            )

        port = self.config.proxy_port or find_open_port(
            disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME]
        )
        command = [
            self.config.switchyard_executable,
            "--routing-profiles",
            str(self.config.routing_profiles),
            "--",
            "serve",
            "--host",
            self.config.proxy_host,
            "--port",
            str(port),
        ]
        logger.info("Starting Switchyard proxy: %s", " ".join(command))
        process = subprocess.Popen(command)
        self._proxy_process = process
        atexit.register(self.stop_proxy)

        # The proxy binds a wildcard address so it is reachable off-box, but a wildcard is not a
        # destination -- this server always reaches its own child over loopback.
        host = "127.0.0.1" if self.config.proxy_host in ("0.0.0.0", "::") else self.config.proxy_host
        root_url = f"http://{host}:{port}"
        self.wait_for_proxy(root_url, process)
        return f"{root_url}/v1"

    def wait_for_proxy(self, root_url: str, process: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.config.proxy_startup_timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Switchyard proxy exited during startup with code {process.returncode}")
            try:
                with urllib.request.urlopen(f"{root_url}/health", timeout=5) as response:
                    if response.status == 200:
                        logger.info("Switchyard proxy is up at %s", root_url)
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)

        self.stop_proxy()
        raise TimeoutError(f"Switchyard proxy did not become healthy within {self.config.proxy_startup_timeout_s}s")

    def stop_proxy(self) -> None:
        process = getattr(self, "_proxy_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()

    # --- Model calls ---

    def client_for_request(self) -> NeMoGymAsyncOpenAI:
        """Return the upstream client, carrying this rollout's session id when there is one.

        The client's headers are fixed at construction, so a request that must be correlated gets
        a copy with the session header merged in. The copy is a plain Pydantic model over Gym's
        shared aiohttp client, so this costs an object allocation, not a connection.
        """
        rollout_id = _ROLLOUT_ID.get()
        if rollout_id is None or not self.config.forward_session_id:
            return self._client

        headers = {**self._client.default_headers, self.config.session_id_header: rollout_id}
        return self._client.model_copy(update={"default_headers": headers})

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.switchyard_model
        async with self._semaphore:
            response_dict = await self.client_for_request().create_response(**body_dict)
        return NeMoGymResponse.model_validate(response_dict)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        body_dict = self.config.extra_body | body.model_dump(exclude_unset=True)
        body_dict["model"] = self.config.switchyard_model
        async with self._semaphore:
            response_dict = await self.client_for_request().create_chat_completion(**body_dict)
        return NeMoGymChatCompletion.model_validate(response_dict)


if __name__ == "__main__":
    SwitchyardModel.run_webserver()
