# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DesktopEnv-compatible client for Gym's remote OSWorld Resources Server."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

import requests


LOG = logging.getLogger("nemo_gym.osworld_agent.remote_environment")


class OSWorldResourcesServerError(RuntimeError):
    pass


class _ObservationController:
    """Read-only subset used by agent scaffolds that inspect ``env.controller``."""

    def __init__(self, env: "RemoteDesktopEnv") -> None:
        self._env = env

    def get_screenshot(self) -> bytes:
        return self._env._get_obs().get("screenshot") or b""  # noqa: SLF001

    def get_accessibility_tree(self) -> Any:
        return self._env._get_obs().get("accessibility_tree")  # noqa: SLF001

    def get_terminal_output(self) -> Any:
        return self._env._get_obs().get("terminal")  # noqa: SLF001


class RemoteDesktopEnv:
    """Proxy the DesktopEnv methods used by the Gym OSWorld rollout loop.

    The ``requests.Session`` retains Gym's signed session cookie from
    ``/seed_session``. Every later request therefore reaches the same live
    DesktopEnv and the same remote Docker worker.
    """

    supports_recording = False
    vm_ip = None

    def __init__(
        self,
        *,
        resources_server_url: str,
        auth_token: str = "",
        request_timeout: float = 900.0,
        connect_timeout: float = 10.0,
        request_retries: int = 3,
        provider_name: str = "remote_docker",
        action_space: str = "pyautogui",
        screen_size: tuple[int, int] = (1920, 1080),
        headless: bool = True,
        require_a11y_tree: bool = False,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        client_password: str = "password",  # pragma: allowlist secret
        enable_proxy: bool = False,
        **_kwargs: Any,
    ) -> None:
        if not resources_server_url.strip():
            raise ValueError("resources_server_url must not be empty")
        if provider_name != "remote_docker":
            raise ValueError(
                "remote OSWorld Resources Server currently requires provider_name='remote_docker'"
            )
        if os_type.lower() != "ubuntu":
            raise ValueError("remote OSWorld Resources Server currently supports Ubuntu only")
        self.resources_server_url = resources_server_url.rstrip("/")
        self.auth_token = auth_token
        self.request_timeout = max(1.0, float(request_timeout))
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.request_retries = max(1, int(request_retries))
        self.provider_name = provider_name
        self.action_space = action_space
        self.screen_size = (int(screen_size[0]), int(screen_size[1]))
        self.headless = bool(headless)
        self.require_a11y_tree = bool(require_a11y_tree)
        self.require_terminal = bool(require_terminal)
        self.client_password = client_password
        self.enable_proxy = bool(enable_proxy)
        self.controller = _ObservationController(self)
        self.session_id: Optional[str] = None
        self.worker: Optional[str] = None
        self._seeded = False
        self._closed = False
        self._last_observation: Dict[str, Any] = {}
        self._operation_index = 0
        self._session = requests.Session()

    def reset(
        self,
        task_config: Optional[Dict[str, Any]] = None,
        seed=None,
        options=None,
    ) -> Dict[str, Any]:
        del seed, options
        if task_config is None:
            raise ValueError("RemoteDesktopEnv.reset requires task_config")
        if self._closed:
            raise OSWorldResourcesServerError("cannot reset a closed remote environment")
        if not self._seeded:
            payload = {
                "task_config": task_config,
                "environment": {
                    "action_space": self.action_space,
                    "screen_width": self.screen_size[0],
                    "screen_height": self.screen_size[1],
                    "headless": self.headless,
                    "require_a11y_tree": self.require_a11y_tree,
                    "require_terminal": self.require_terminal,
                    "client_password": self.client_password,
                    "enable_proxy": self.enable_proxy,
                },
            }
            response = self._request("POST", "/seed_session", json_body=payload)
            self._seeded = True
            self.session_id = str(response.get("session_id") or "") or None
            self.worker = str(response.get("worker") or "") or None
        else:
            response = self._request(
                "POST",
                "/reset",
                json_body={"task_config": task_config},
            )
        self._last_observation = self._decode_observation(response.get("observation") or {})
        return self._last_observation

    def _get_obs(self) -> Dict[str, Any]:
        self._require_seeded()
        response = self._request("GET", "/observe")
        self._last_observation = self._decode_observation(response)
        return self._last_observation

    def step(self, action: Any, pause: float = 2) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._require_seeded()
        operation_id = f"{self.session_id or 'session'}-{self._operation_index}-{uuid.uuid4().hex}"
        self._operation_index += 1
        response = self._request(
            "POST",
            "/step",
            json_body={
                "operation_id": operation_id,
                "action": action,
                "pause": float(pause),
            },
        )
        self._last_observation = self._decode_observation(response.get("observation") or {})
        info = response.get("info")
        return (
            self._last_observation,
            float(response.get("reward") or 0.0),
            bool(response.get("done")),
            info if isinstance(info, dict) else {"value": info},
        )

    def evaluate(self, *_args: Any, **_kwargs: Any) -> float:
        self._require_seeded()
        response = self._request("POST", "/evaluate", json_body={})
        return float(response.get("score") or 0.0)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._seeded:
                self._request("POST", "/close", json_body={}, allow_not_found=True)
        finally:
            self._closed = True
            self._session.close()

    def _require_seeded(self) -> None:
        if not self._seeded:
            raise OSWorldResourcesServerError("remote environment has not been seeded")
        if self._closed:
            raise OSWorldResourcesServerError("remote environment is closed")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        allow_not_found: bool = False,
    ) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        url = f"{self.resources_server_url}{path}"
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.request_retries + 1):
            started = time.monotonic()
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json_body,
                    headers=headers,
                    timeout=(self.connect_timeout, self.request_timeout),
                )
                elapsed = time.monotonic() - started
                self._log_transport(
                    method=method,
                    path=path,
                    status_code=response.status_code,
                    elapsed_seconds=elapsed,
                    response_bytes=len(response.content),
                    attempt=attempt,
                )
                if allow_not_found and response.status_code == 404:
                    return {}
                if response.status_code in {502, 503, 504} and attempt < self.request_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
                    continue
                if not response.ok:
                    try:
                        detail = response.json().get("detail")
                    except Exception:  # noqa: BLE001
                        detail = response.text[:1000]
                    raise OSWorldResourcesServerError(
                        f"{method} {path} returned HTTP {response.status_code}: {detail}"
                    )
                if not response.content:
                    return {}
                decoded = response.json()
                return decoded if isinstance(decoded, dict) else {"value": decoded}
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                self._log_transport(
                    method=method,
                    path=path,
                    status_code=None,
                    elapsed_seconds=time.monotonic() - started,
                    response_bytes=0,
                    attempt=attempt,
                    error=repr(exc),
                )
                if attempt < self.request_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
                    continue
                break
        raise OSWorldResourcesServerError(
            f"{method} {path} failed after {self.request_retries} attempt(s): {last_error}"
        ) from last_error

    @staticmethod
    def _decode_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
        encoded = payload.get("screenshot_b64") or ""
        try:
            screenshot = base64.b64decode(encoded, validate=True) if encoded else b""
        except (ValueError, TypeError) as exc:
            raise OSWorldResourcesServerError("Resources Server returned invalid screenshot base64") from exc
        return {
            "screenshot": screenshot,
            "accessibility_tree": payload.get("accessibility_tree"),
            "terminal": payload.get("terminal"),
            "instruction": payload.get("instruction"),
        }

    def _log_transport(
        self,
        *,
        method: str,
        path: str,
        status_code: Optional[int],
        elapsed_seconds: float,
        response_bytes: int,
        attempt: int,
        error: Optional[str] = None,
    ) -> None:
        log_path = os.environ.get("OSWORLD_RESOURCES_IO_LOG", "").strip()
        if not log_path:
            return
        event = {
            "schema_version": 1,
            "timestamp_unix_ns": time.time_ns(),
            "event": "osworld_resources_http",
            "method": method,
            "path": path,
            "status_code": status_code,
            "elapsed_seconds": elapsed_seconds,
            "response_bytes": response_bytes,
            "attempt": attempt,
            "session_id": self.session_id,
            "worker": self.worker,
            "error": error,
        }
        try:
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        except OSError:
            LOG.exception("Failed to append Resources Server transport log")
