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
import contextvars
import hashlib
import importlib.metadata
import json
import logging
import re
import tomllib
from collections.abc import Mapping
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field, model_validator

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    DISALLOWED_PORTS_KEY_NAME,
    OBSERVABILITY_ENABLED_KEY_NAME,
    TOKEN_ID_CAPTURE_BLOCK,
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
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint, request


logger = logging.getLogger(__name__)


# Gym's capture middleware strips the /ng-rollout/<id> correlation prefix before routing, so by the
# time a handler runs the rollout id is gone. _RolloutSessionMiddleware runs outside it and
# republishes the id here for the duration of the request.
_ROLLOUT_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "switchyard_model_rollout_id", default=None
)

# Charset kept in step with nemo_gym.rollout_correlation.RolloutContextMiddleware and the
# _validate_rollout_id check on the capture path. It matters more here than it does there: this id
# leaves the process as an HTTP header value, so anything outside the contract's alphabet is
# dropped rather than forwarded.
_ROLLOUT_PATH_RE = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[A-Za-z0-9][A-Za-z0-9._-]*)/.*$")


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

    By default Gym hosts the proxy itself from ``deployment``, so a routed eval is one command and
    the user never manages a proxy. Setting ``switchyard_base_url`` instead attaches to a proxy
    someone else runs -- useful when an eval needs to pin a specific Switchyard build, or when
    several servers should share one instance (routing strategies with session or agent affinity
    are stateful, so replicas each hosting their own proxy would not route like one shared proxy).
    """

    # Set to attach to an already-running proxy instead of hosting one.
    switchyard_base_url: Optional[str] = None
    switchyard_api_key: str = "dummy"  # pragma: allowlist secret

    # The route name to request. Switchyard maps it to a concrete provider model, so this is a
    # routing label rather than a model id -- which target actually served the call comes back on
    # the response.
    switchyard_model: str

    # Used when hosting. A native Switchyard TOML deployment: llm_clients, targets, and routes,
    # validated by the server when it loads. This is the routing condition an eval runs under --
    # an explicit, diffable artifact, which is what makes routed runs comparable.
    deployment: Optional[str] = None
    # The hosted proxy binds loopback only; this server is the network-facing piece.
    proxy_port: Optional[int] = None

    # Headers Switchyard reads as its opaque session id. Gym sends its rollout-attempt id so
    # per-call routing decisions can be joined back to the rollout after the run.
    # x-switchyard-session-id is the name the native server parses into routing metadata (session
    # affinity, request logs, spans). A list because attach-mode proxies can key other subsystems
    # on other names -- e.g. proxy_x_session_id for switchyard-server durable routing logs -- but
    # add names knowingly: Switchyard strips only the headers it recognizes, so an unrecognized
    # name is forwarded verbatim to the upstream provider.
    session_id_headers: List[str] = Field(default_factory=lambda: ["x-switchyard-session-id"])
    forward_session_id: bool = True

    # Directory that receives this run's routing-condition record: a manifest written at startup
    # (route, mode, deployment hash and archived contents, nemo-switchyard version) and a
    # /v1/stats snapshot written at shutdown. Point it at the run's output directory, one per
    # run -- the manifest is what makes a routed run identifiable and reproducible after the
    # proxy is gone. None disables both writes.
    condition_dir: Optional[str] = None

    # Caller-supplied provenance for an attached proxy, copied verbatim into the condition
    # manifest. Gym cannot see what an external proxy runs, so whoever operates it supplies the
    # identity -- e.g. {"switchyard_commit": ..., "deployment_sha256": ...}. Ignored when hosting:
    # the manifest already identifies the hosted proxy exactly.
    proxy_provenance: Dict[str, Any] = Field(default_factory=dict)

    extra_body: Dict[str, Any] = Field(default_factory=dict)
    default_headers: Dict[str, str] = Field(default_factory=dict)

    max_concurrent_requests: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Cap on in-flight upstream requests from this server (per-process asyncio.Semaphore). None = unlimited."
        ),
    )

    @property
    def launches_proxy(self) -> bool:
        return self.switchyard_base_url is None

    @model_validator(mode="after")
    def validate_target(self) -> "SwitchyardModelConfig":
        if self.launches_proxy and not self.deployment:
            raise ValueError(
                "switchyard_model needs either deployment (Gym hosts the proxy from a native "
                "Switchyard TOML deployment) or switchyard_base_url (attach to a proxy you run yourself)"
            )
        if self.launches_proxy and (self.num_workers or 1) > 1:
            # Each uvicorn worker is its own process, so each would host its own proxy: stateful
            # routing splits across instances, /v1/stats fragments, and a fixed proxy_port
            # collides. One shared proxy is the coherent shape for parallel workers.
            raise ValueError(
                "switchyard_model cannot host the proxy with num_workers > 1: every worker would "
                "start its own proxy, splitting session affinity and stats. Run one proxy yourself "
                "and attach all workers to it with switchyard_base_url."
            )
        if self.switchyard_base_url and self.deployment:
            # Both fields default from environment variables, so this is easy to hit by accident.
            # Attaching wins, but say so -- silently ignoring a deployment the user supplied would
            # look like their routing config was in effect when it was not.
            logger.warning(
                "switchyard_model: both switchyard_base_url and deployment are set. Attaching to %s; "
                "the deployment %s is served by that proxy, not by Gym, and is otherwise ignored.",
                self.switchyard_base_url,
                self.deployment,
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
        app = super().setup_webserver()
        # Added last, so it wraps the capture middleware and still sees the correlation prefix.
        app.add_middleware(_RolloutSessionMiddleware)
        self.setup_proxy_lifespan(app)
        return app

    def setup_proxy_lifespan(self, app) -> None:
        """Host the proxy for exactly as long as the app serves.

        Startup and shutdown must share a thread: the native server's PyO3 binding is
        thread-affine (unsendable), so the thread that constructs it is the only one allowed to
        close it. Running both ends inside the lifespan guarantees that on any host -- uvicorn
        runs the lifespan on the main thread, test clients on a worker thread, and either way
        construction and close land together. It also keeps config validation, app assembly, and
        tests free of side effects; the proxy exists only while the app serves.
        """
        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            if self.config.launches_proxy:
                print("Starting Switchyard proxy...")
                self._build_client(self.start_proxy())
            # finally, not a trailing statement -- and entered the moment the proxy serves:
            # anything that fails between here and a finished startup (the manifest write, the
            # inner lifespan) must still close the proxy, or it outlives the server that never
            # finished starting.
            try:
                self.check_rollout_correlation()
                self.write_condition_manifest()
                async with main_app_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                # Stats before stop: the snapshot needs the proxy still answering.
                await self.snapshot_proxy_stats()
                self.stop_proxy()

        app.router.lifespan_context = lifespan_wrapper

    # --- Proxy lifecycle ---

    def start_proxy(self) -> str:
        """Host the native Switchyard server in-process and return its base URL.

        The constructor loads and validates the TOML deployment, binds loopback, and returns
        already serving -- a bad deployment or an unbindable port is an exception here, not a
        timeout later, so there is no health polling and no subprocess to reap.

        nemo-switchyard is a dependency of this server, so the import normally succeeds. Catch
        anyway -- a hand-built environment can leave it missing, and an explicit message beats a
        bare ModuleNotFoundError.
        """
        try:
            from switchyard_rust.server import Server
        except ImportError as error:
            raise RuntimeError(
                "switchyard_model could not import switchyard_rust, so it cannot host a proxy. It ships "
                "with this server's nemo-switchyard dependency; if you are running a custom environment, "
                "install it with `pip install nemo-switchyard==0.2.0` or set switchyard_base_url to attach "
                "to a proxy you run yourself."
            ) from error

        port = self.config.proxy_port or find_open_port(
            disallowed_ports=get_global_config_dict()[DISALLOWED_PORTS_KEY_NAME]
        )
        logger.info("Hosting Switchyard proxy from deployment %s on port %d", self.config.deployment, port)
        self._proxy_server = Server(str(self.config.deployment), port=port)
        logger.info("Switchyard proxy is up at %s", self._proxy_server.base_url)
        return f"{self._proxy_server.base_url}/v1"

    def stop_proxy(self) -> None:
        server = getattr(self, "_proxy_server", None)
        if server is None:
            return
        self._proxy_server = None
        server.close()

    # --- Routing-condition record ---

    def proxy_root_url(self) -> Optional[str]:
        """The proxy's root URL (no /v1), in whichever mode is active."""
        server = getattr(self, "_proxy_server", None)
        if server is not None:
            return server.base_url
        if self.config.switchyard_base_url:
            return self.config.switchyard_base_url.rstrip("/").removesuffix("/v1")
        return None

    def check_rollout_correlation(self) -> None:
        """Fail or warn at startup when the rollout id cannot reach this server.

        The id arrives on the ``/ng-rollout/<id>/`` URL prefix, which agents add only when
        model-call observability or training-token capture is enabled. Without one of those,
        forward_session_id has nothing to forward -- Switchyard sees no session and the documented
        correlation and session-affinity behavior silently does not happen. Asking for forwarding
        explicitly makes that an error; leaving the default on makes it a warning, because plain
        evals that never look at sessions should not have to care.
        """
        if not self.config.forward_session_id:
            return
        global_config = getattr(self.server_client, "global_config_dict", None)
        if not isinstance(global_config, Mapping):
            return
        token_capture_block = global_config.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        correlation_active = bool(global_config.get(OBSERVABILITY_ENABLED_KEY_NAME, False)) or (
            isinstance(token_capture_block, Mapping) and bool(token_capture_block.get("enabled", False))
        )
        if correlation_active:
            return
        message = (
            "switchyard_model: forward_session_id is on, but neither observability_enabled nor "
            "token_id_capture is -- so requests carry no /ng-rollout prefix, and no session id "
            "will reach Switchyard. Enable ++observability_enabled=true (or token capture) to "
            "correlate rollouts, or set forward_session_id=false to accept uncorrelated routing."
        )
        if "forward_session_id" in self.config.model_fields_set:
            raise RuntimeError(message)
        logger.warning(message)

    def write_condition_manifest(self) -> None:
        """Record the routing condition this run serves under, if condition_dir is set.

        The manifest is the run's provenance: which route, against which deployment (hash and
        archived contents), on which proxy. Two runs are comparable when their manifests differ
        only in the route or deployment, and a hosted run is reproducible from the archive alone.
        Failing to write it must not take down serving -- provenance is worth a warning, not an
        outage -- so every step, including reading the deployment, stays inside the handler.
        """
        if not self.config.condition_dir:
            return
        try:
            deployment_sha256: Optional[str] = None
            deployment_toml: Optional[str] = None
            if self.config.launches_proxy:
                deployment_bytes = Path(str(self.config.deployment)).read_bytes()
                deployment_sha256 = hashlib.sha256(deployment_bytes).hexdigest()
                deployment_toml = self._redacted_deployment_toml(deployment_bytes.decode(errors="replace"))

            manifest = {
                "route": self.config.switchyard_model,
                "mode": "hosted" if self.config.launches_proxy else "attached",
                "proxy_root_url": self.proxy_root_url(),
                # Only the hosted proxy runs the local wheel. An attached proxy is someone else's
                # build -- reporting the local version there would name code that never served a
                # request, so identity comes from caller-supplied proxy_provenance instead.
                "nemo_switchyard_version": self._nemo_switchyard_version() if self.config.launches_proxy else None,
                "proxy_provenance": dict(self.config.proxy_provenance) or None,
                "deployment_path": str(self.config.deployment) if self.config.deployment else None,
                "deployment_sha256": deployment_sha256,
                "deployment_toml": deployment_toml,
                "session_id_headers": list(self.config.session_id_headers),
                "written_at": datetime.now(timezone.utc).isoformat(),
            }
            condition_dir = Path(self.config.condition_dir)
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "switchyard-condition.json").write_text(json.dumps(manifest, indent=2) + "\n")
        except Exception:
            logger.warning("switchyard_model: could not write the condition manifest", exc_info=True)

    def _redacted_deployment_toml(self, text: str) -> Optional[str]:
        """Archive-safe copy of the deployment, or None when safety cannot be shown.

        The native schema carries credentials by environment-variable name (api_key_env), which
        is safe to archive -- but extra_headers values are sent to providers verbatim, so an
        Authorization header there is a live secret. Parse the TOML, collect every extra_headers
        value and every api_key-named value wherever they appear, and blank those exact strings
        in the raw text. A deployment that does not parse cannot be searched for secrets, so it
        is not archived at all; the hash still identifies it.
        """
        try:
            parsed = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            logger.warning(
                "switchyard_model: deployment is not parseable TOML, so it cannot be checked for "
                "secrets and will not be archived in the condition manifest"
            )
            return None

        secrets: List[str] = []

        def collect(node: Any, under_extra_headers: bool = False) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, str) and (under_extra_headers or key == "api_key"):
                        secrets.append(value)
                    else:
                        collect(value, under_extra_headers or key == "extra_headers")
            elif isinstance(node, list):
                for item in node:
                    collect(item, under_extra_headers)

        collect(parsed)
        for secret in secrets:
            if secret:
                text = text.replace(secret, "<redacted>")
        return text

    async def snapshot_proxy_stats(self) -> None:
        """Persist the proxy's /v1/stats before it goes away, if condition_dir is set.

        A hosted proxy's counters die with the process, so this is the only chance to keep the
        proxy-side view of the run -- per-target requests, errors, tokens, latency -- next to the
        eval outputs it explains. Best-effort for the same reason as the manifest, and hard-capped
        well inside Gym's shutdown window: this runs on the event loop during lifespan shutdown,
        and Gym escalates to SIGKILL about a second after asking servers to stop.
        """
        root_url = self.proxy_root_url()
        if not self.config.condition_dir or root_url is None:
            return
        try:
            stats = await asyncio.wait_for(self._fetch_proxy_stats(f"{root_url}/v1/stats"), timeout=0.5)
            snapshot = {
                # A proxy counts from its own start, not this run's. Hosted, the two coincide;
                # attached, a shared proxy's counters span every run and neighbor it has served,
                # so the file says what it holds instead of implying run-level accounting.
                "scope": "proxy-lifetime aggregate"
                if not self.config.launches_proxy
                else "this run (proxy hosted for exactly this run)",
                "mode": "hosted" if self.config.launches_proxy else "attached",
                "written_at": datetime.now(timezone.utc).isoformat(),
                "stats": stats,
            }
            condition_dir = Path(self.config.condition_dir)
            condition_dir.mkdir(parents=True, exist_ok=True)
            (condition_dir / "switchyard-stats.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        except Exception:
            logger.warning("switchyard_model: could not snapshot proxy stats", exc_info=True)

    async def _fetch_proxy_stats(self, url: str) -> Any:
        """GET /v1/stats through Gym's shared aiohttp client, as the proxy's configured caller.

        The model path authenticates with switchyard_api_key and default_headers; an attached
        proxy that requires them would serve model calls and still 401 an anonymous stats read,
        so the snapshot authenticates the same way.
        """
        response = await request(
            "GET",
            url,
            headers={**self.config.default_headers, "Authorization": f"Bearer {self.config.switchyard_api_key}"},
            timeout=aiohttp.ClientTimeout(total=0.5),
        )
        async with response:
            response.raise_for_status()
            return await response.json()

    def _nemo_switchyard_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("nemo-switchyard")
        except importlib.metadata.PackageNotFoundError:
            return None

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

        headers = {**self._client.default_headers}
        for name in self.config.session_id_headers:
            headers[name] = rollout_id
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
elif is_nemo_gym_fastapi_entrypoint(__file__):  # pragma: no cover - process-entry glue
    # Multi-worker uvicorn imports this module by string and looks up ``app``. Config validation
    # has already rejected hosted mode with num_workers > 1, so every worker built here attaches
    # to one shared external proxy.
    app = SwitchyardModel.run_webserver()  # noqa: F401
