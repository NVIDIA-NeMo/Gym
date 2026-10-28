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
"""`ServerClient`, which finds Gym servers through the head server and calls them, and server URLs."""

import json
import logging
import re
from importlib import import_module
from os import getenv
from typing import Any, Literal, Optional, Union, Unpack

import requests
from aiohttp import (
    ClientResponse,
)
from aiohttp.client import _RequestOptions
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, PrivateAttr
from requests.exceptions import ConnectionError

from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.client.http_client import (
    GlobalAIOHTTPAsyncClientConfig,
    is_global_aiohttp_client_setup,
    request,
    set_global_aiohttp_client,
)
from nemo_gym.config.config_types import (
    ROLLOUT_PATH_PREFIX,
    TOKEN_CAPTURE_PATH_SEGMENT,
    TRACE_CONTEXT_PATH_SEGMENT,
    BaseServerConfig,
    HeadServerUnreachableError,
)
from nemo_gym.config.global_config import (
    HEAD_SERVER_KEY_NAME,
    OBSERVABILITY_ENABLED_KEY_NAME,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.core.episode import (
    VERIFIER_INVOCATION_HEADER,
    current_rollout_id,
    current_verifier_invocation_id,
    maybe_rollout_id_from_run_body,
)
from nemo_gym.telemetry.setup import is_trace_exporter_active
from nemo_gym.utils.server_process import (
    _has_injected_global_config_env,
)


logger = logging.getLogger(__name__)


NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME = "NEMO_GYM_MODEL_SERVER_NAME"


NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME = "NEMO_GYM_MODEL_SERVER_BASE_URL"


ServerStatus = Union[Literal["success"], Literal["connection_error"], Literal["timeout"], Literal["unknown_error"]]


def _connectable_client_host(host: str) -> str:
    """Return a connectable client host for a server bind address."""

    if host == "0.0.0.0":
        return "127.0.0.1"
    if host in {"::", "[::]"}:
        return "[::1]"
    return host


class ServerClient(BaseModel):
    head_server_config: BaseServerConfig

    global_config_dict: DictConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Resolved base URLs, cached by server name.
    _server_base_urls: dict[str, str] = PrivateAttr(default_factory=dict)
    _checkpoint_enabled: Optional[bool] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        # Validate the ``checkpoint:`` block once, when the client is built at startup, rather than on every request.
        self.checkpoint_enabled()

    def checkpoint_enabled(self) -> bool:
        """Return whether the global ``checkpoint:`` block enables partial-rollout checkpointing."""
        if self._checkpoint_enabled is None:
            self._checkpoint_enabled = checkpoint_settings(self.global_config_dict) is not None
        return self._checkpoint_enabled

    @staticmethod
    def _client_host(host: str) -> str:
        """Return a connectable host for a server bind address.

        Wildcard addresses are valid for binding a server but are not valid
        advertised client targets. In particular, proxy-aware HTTP clients can
        route ``0.0.0.0`` through an external proxy instead of reaching the
        local server.
        """
        return _connectable_client_host(host)

    def assistant_message_header(self, model_server_name: str) -> bytes | None:
        """Read the optional header property of harnesses using this model server.

        The property lives on the harness package so model workers need not import
        an agent's app or install its runtime dependencies.
        """
        headers = set()
        for instance in self.global_config_dict.values():
            if not isinstance(instance, (dict, DictConfig)):
                continue
            for harness, config in instance.get("responses_api_agents", {}).items():
                model = config.get("model_server") or {}
                if model.get("name") != model_server_name or model.get("type") != "responses_api_models":
                    continue
                package = import_module(f"responses_api_agents.{harness}")
                header = getattr(package, "_assistant_message_header", None)
                if header is not None:
                    headers.add(header.lower())
        if len(headers) > 1:
            raise ValueError(f"Harnesses using model server {model_server_name!r} declare different assistant headers")
        return next(iter(headers), None)

    @classmethod
    def load_head_server_config(cls) -> BaseServerConfig:
        global_config_dict = get_global_config_dict()
        server_config_dict = global_config_dict[HEAD_SERVER_KEY_NAME]
        config = BaseServerConfig.model_validate(server_config_dict)
        return config

    @classmethod
    def load_from_global_config(cls, head_server_config: Optional[BaseServerConfig] = None) -> "ServerClient":
        """Build a client from the fully resolved global config.

        Gym-launched server processes reuse the config injected by their parent.
        Other processes fetch the config from the head server.
        """
        if head_server_config is None:
            head_server_config = cls.load_head_server_config()

        if _has_injected_global_config_env():
            return cls(
                head_server_config=head_server_config,
                global_config_dict=get_global_config_dict(),
            )

        # It's critical we use requests here instead of the global httpx client since a FastAPI server may be run downstream of this function call.
        head_server_host = _connectable_client_host(head_server_config.host)
        head_server_url = f"http://{head_server_host}:{head_server_config.port}"
        try:
            response = requests.get(
                f"{head_server_url}/global_config_dict_yaml",
            )
        except ConnectionError as e:
            # requests' ConnectionError also covers proxy and name-resolution failures; keep the real reason
            # (with its traceback) for --verbose, since the ConfigError below is printed without its cause.
            logger.debug(
                "Could not fetch the global config from the head server at %s", head_server_url, exc_info=True
            )
            # A ConfigError so the CLI prints just this message (no traceback); the cause stays chained.
            raise HeadServerUnreachableError(
                f"Could not connect to the head server at {head_server_url}. Is the head server running? "
                "Start it with: `gym env start`. If it is already running on a different host or port, pass "
                "`++head_server.host=<host>` / `++head_server.port=<port>` so this command can find it."
            ) from e

        global_config_dict_yaml = response.content.decode()
        global_config_dict = OmegaConf.create(json.loads(global_config_dict_yaml))

        return cls(head_server_config=head_server_config, global_config_dict=global_config_dict)

    async def request(
        self,
        server_name: str,
        url_path: str,
        method: str,
        *,
        _control: bool = False,
        _resend_after_disconnect: bool = True,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """Call another server. ``_control`` sends the call over the reserved control pool."""
        model_server_name = getenv(NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME)
        model_server_base_url = getenv(NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME)
        if model_server_base_url and server_name == model_server_name:
            # Subprocess agents do not inherit the current rollout context.
            # The launcher provides a model URL that already contains the rollout prefix.
            # Use that URL instead of rebuilding it from global server configuration.
            base_url = model_server_base_url.rstrip("/")
        else:
            base_url = self._resolve_base_url(server_name)

        json_obj = kwargs.get("json")
        if "json" in kwargs:
            if isinstance(json_obj, BaseModel):
                json_obj = json_obj.model_dump(exclude_unset=True)
                kwargs["json"] = json_obj

        # Checkpointing needs every model and resources call attributed to its rollout attempt,
        # so it carries the rollout prefix even when observability is off.
        attribute_rollout = (
            self.global_config_dict.get(OBSERVABILITY_ENABLED_KEY_NAME, False) or self.checkpoint_enabled()
        )
        server_entry = self.global_config_dict.get(server_name)
        rollout_id = current_rollout_id()
        if attribute_rollout and server_entry is not None and "resources_servers" in server_entry:
            if url_path == "/verify":
                rollout_id = rollout_id or maybe_rollout_id_from_run_body(json_obj)
            if rollout_id is not None and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/"):
                url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"

        if (
            rollout_id is not None
            and attribute_rollout
            and server_entry is not None
            and "responses_api_models" in server_entry
            and url_path.partition("?")[0] in {"/v1/responses", "/v1/chat/completions", "/v1/messages"}
            and not url_path.startswith(f"/{ROLLOUT_PATH_PREFIX}/")
        ):
            url_path = f"{rollout_path_prefix(rollout_id)}{url_path}"
            verifier_invocation_id = current_verifier_invocation_id()
            if verifier_invocation_id is not None:
                kwargs["headers"] = {
                    **(kwargs.get("headers") or {}),
                    VERIFIER_INVOCATION_HEADER: verifier_invocation_id,
                }

        return await request(
            method=method,
            url=f"{base_url}{url_path}",
            _internal=True,
            _server_name=server_name,
            _control=_control,
            # A /run starts an episode attempt, so with checkpointing it is never sent twice.
            # A dropped connection usually means the server's worker died mid-episode.
            # Sending it again would run the attempt a second time,
            # and the first run's sessions on other servers would never be freed: the second run replies,
            # so the controller never retires the attempt.
            _resend_after_disconnect=_resend_after_disconnect
            and not (self.checkpoint_enabled() and url_path == "/run"),
            **kwargs,
        )

    async def get(
        self,
        server_name: str,
        url_path: str,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """
        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="GET",
            **kwargs,
        )

    async def post(
        self,
        server_name: str,
        url_path: str,
        **kwargs: Unpack[_RequestOptions],
    ) -> ClientResponse:
        """
        Args:
            server_name: str
                The name of the server you are trying to call.
            url_path: str
                The URL path in the server you are trying to call e.g. "/v1/responses".

        """
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="POST",
            **kwargs,
        )

    def poll_for_status(self, server_name: str, *, timeout_seconds: float = 5) -> ServerStatus:  # pragma: no cover
        if server_name == HEAD_SERVER_KEY_NAME:
            server_config_dict = self.global_config_dict[HEAD_SERVER_KEY_NAME]
        else:
            server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)

        try:
            requests.get(self._build_server_base_url(server_config_dict), timeout=timeout_seconds)
            # We don't check the status code since there may not be a route at /
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"
        except requests.exceptions.Timeout:
            return "timeout"
        except Exception:
            return "unknown_error"

    def _resolve_base_url(self, server_name: str) -> str:
        cached = self._server_base_urls.get(server_name)
        if cached is not None:
            return cached
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)
        self._server_base_urls[server_name] = base_url
        return base_url

    def _build_server_base_url(self, server_config_dict: OmegaConf) -> str:
        # Use the module helper so this method remains safe when borrowed by a
        # duck-typed or mocked ServerClient.
        host = _connectable_client_host(server_config_dict.host)
        return f"http://{host}:{server_config_dict.port}"


SESSION_ID_KEY = "session_id"


# A W3C ``traceparent``, version 00, the only form the propagator writes.
_TRACEPARENT_PATTERN = r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}"


def get_server_url(server_name: str) -> str:
    global_config_dict = get_global_config_dict()

    model_server_config = get_first_server_config_dict(
        global_config_dict,
        server_name,
    )

    return f"http://{model_server_config['host']}:{model_server_config['port']}"


def rollout_path_prefix(rollout_id: Optional[str], *, token_capture: bool = False) -> str:
    """Return the leading model-server path prefix for a rollout, if available."""
    if not rollout_id:
        return ""
    capture_segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if token_capture else ""
    return f"/{ROLLOUT_PATH_PREFIX}/{rollout_id}{capture_segment}"


def trace_context_path_segment() -> str:
    """Return ``/ng-trace/<traceparent>`` for the current span, or ``""`` while traces are off or no span is current.

    ``TraceContextPathMiddleware`` on the receiving Gym server turns it back into the header.
    """
    if not is_trace_exporter_active():
        return ""
    from nemo_gym.telemetry.contrib import inject_trace_context

    carrier: dict[str, str] = {}
    inject_trace_context(carrier)
    traceparent = carrier.get("traceparent", "")
    if not re.fullmatch(_TRACEPARENT_PATTERN, traceparent):
        return ""
    return f"/{TRACE_CONTEXT_PATH_SEGMENT}/{traceparent}"


def apply_rollout_prefix(
    base_url: str, rollout_id: Optional[str], *, token_capture: bool = False, trace_context: bool = True
) -> str:
    """Append the rollout prefix and, while traces export, the current span's trace context to a model-server root URL.

    Every caller hands the result to a client Gym does not instrument (a harness in a sandbox, a
    third-party client library), which is why the trace context travels in the URL here; Gym's own
    clients send the ``traceparent`` header instead and use ``url_path_for_run``.

    The span named is the one current when the URL is built, so pass ``trace_context=False`` for a
    URL that is built once and reused across rollouts (a cached client): otherwise every later
    rollout's calls would be parented to the first one's span.
    """
    prefix = rollout_path_prefix(rollout_id, token_capture=token_capture) if rollout_id else ""
    if trace_context:
        prefix += trace_context_path_segment()
    if not prefix:
        return base_url
    return base_url.rstrip("/") + prefix


def setup_server_client(head_server_config: Optional[BaseServerConfig] = None) -> ServerClient:  # pragma: no cover
    server_client = ServerClient.load_from_global_config(head_server_config)

    # We set this rollout global aiohttp client to use the same max connections as the underlying head server global config.
    if not is_global_aiohttp_client_setup():
        set_global_aiohttp_client(cfg=GlobalAIOHTTPAsyncClientConfig.model_validate(server_client.global_config_dict))

    return server_client
