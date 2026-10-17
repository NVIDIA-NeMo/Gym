# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One browser session per process, reachable over HTTP.

A headed PyAutoGUI driver acts on its process's X display, so one process can
serve one session at a time. A browser instance is such a process with its own
display: it hosts an unmodified web backend (driver plus evaluator) behind six
endpoints, and the resources server reaches it through ``RemoteWebBackend``.
The resources server can then hold as many sessions as it has instances, while
every action still runs through the same driver code as a local session.

The server is single-threaded on purpose: Playwright's synchronous API is
thread-affine, so every backend call runs on the main thread.
"""

from __future__ import annotations

import argparse
import hmac
import importlib
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.browser_session import BrowserSessionHandle
from nemo_gym.web.models import WebAction, WebObservation, WebStepResult, WebTask, WebVerifierResult


LOG = logging.getLogger("nemo_gym.web.browser_instance")

TRANSPORT = "browser_instance"
TOKEN_HEADER = "X-Browser-Instance-Token"
TOKEN_ENV = "NEMO_GYM_BROWSER_INSTANCE_TOKEN"
# Instances are reached directly; an HTTP(S)_PROXY meant for the browsed sites
# must never sit between the resources server and its instances.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class BrowserInstanceUnavailableError(RuntimeError):
    """The instance could not be reached or answered outside its protocol."""


def _load(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected 'module:attribute', got {path!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _object_path(value: Any) -> str:
    return f"{value.__module__}:{value.__qualname__}"


class RemoteWebBackend:
    """Forward the web backend protocol to one leased browser instance.

    Calls are synchronous and run on the session's operation-runner thread,
    like the local backend they replace. The instance builds the real backend
    on ``reset`` from the factory and configuration named here, so the
    resources server and the instance always agree on driver and evaluator.
    """

    def __init__(
        self,
        *,
        handle: BrowserSessionHandle,
        config: Any,
        session_id: str,
        backend_factory: str,
    ) -> None:
        if handle.transport != TRANSPORT or not handle.endpoint:
            raise ValueError(f"RemoteWebBackend requires a {TRANSPORT!r} lease with an endpoint")
        self._endpoint = handle.endpoint.rstrip("/")
        self._token = str(handle.metadata.get("token") or "")
        self._timeout = float(handle.metadata.get("request_timeout_seconds") or 600.0)
        self._config = config
        self._session_id = session_id
        self._backend_factory = backend_factory

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        body = self._call(
            "/reset",
            {
                "session_id": self._session_id,
                "backend_factory": self._backend_factory,
                "config_class": _object_path(type(self._config)),
                "config": self._config.model_dump(mode="json"),
                "task": task.model_dump(mode="json"),
            },
        )
        return WebObservation.model_validate(body["observation"]), dict(body["info"])

    def observe(self) -> WebObservation:
        return WebObservation.model_validate(self._call("/observe", {})["observation"])

    def step(self, action: WebAction) -> WebStepResult:
        return WebStepResult.model_validate(self._call("/step", {"action": action.model_dump(mode="json")})["result"])

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult:
        return WebVerifierResult.model_validate(self._call("/evaluate", {"final_answer": final_answer})["result"])

    def close(self) -> None:
        # A close that fails cannot leave a live browser behind: the instance
        # exits after a failed close, and the pool restarts any instance that
        # is unreachable or still busy when its lease comes back. Raising here
        # would only keep the session's capacity slot occupied by a retry that
        # can no longer reach the same instance.
        try:
            self._call("/close", {})
        except (BrowserInstanceUnavailableError, RuntimeError) as exc:
            LOG.warning(
                "event=browser_instance_close_failed session=%s endpoint=%s error=%s",
                self._session_id,
                self._endpoint,
                exc,
            )

    def _call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._endpoint + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", TOKEN_HEADER: self._token},
            method="POST",
        )
        try:
            with DIRECT.open(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read())
            except ValueError:
                error = {}
            error_type = str(error.get("error_type") or f"HTTP{exc.code}")
            message = str(error.get("message") or exc.reason)
            # ValueError is the backends' signal for a deterministic task or
            # environment precondition; keep it so the session manager can
            # classify it the same way as for a local backend.
            if error_type == "ValueError":
                raise ValueError(message) from None
            if exc.code in {401, 409}:
                raise BrowserInstanceUnavailableError(f"{path}: {error_type}: {message}") from None
            raise RuntimeError(f"browser instance {path} failed: {error_type}: {message}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            raise BrowserInstanceUnavailableError(f"browser instance {path} unreachable: {exc}") from exc


class _InstanceState:
    def __init__(self, token: str) -> None:
        self.token = token
        self.backend: Any = None
        self.session_id: str | None = None
        self.exit_code: int | None = None


def _make_handler(state: _InstanceState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - BaseHTTPRequestHandler API
            LOG.debug("http %s", format % args)

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._send(404, {"error_type": "NotFound", "message": self.path})
                return
            self._send(
                200,
                {
                    "ok": True,
                    "busy": state.backend is not None,
                    "session_id": state.session_id,
                    "display": os.environ.get("DISPLAY", ""),
                    "pid": os.getpid(),
                },
            )

        def do_POST(self) -> None:
            if not hmac.compare_digest(self.headers.get(TOKEN_HEADER, ""), state.token):
                self._send(401, {"error_type": "Unauthorized", "message": "bad instance token"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, self._dispatch(self.path, payload))
            except _Conflict as exc:
                self._send(409, {"error_type": "Conflict", "message": str(exc)})
            except Exception as exc:  # noqa: BLE001 - the HTTP boundary reports every backend failure.
                LOG.exception("event=browser_instance_call_failed path=%s session=%s", self.path, state.session_id)
                self._send(500, {"error_type": type(exc).__name__, "message": str(exc)})

        def _dispatch(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/reset":
                return self._reset(payload)
            if path == "/close":
                return self._close()
            if state.backend is None:
                raise _Conflict("instance has no active session")
            if path == "/observe":
                return {"observation": state.backend.observe().model_dump(mode="json")}
            if path == "/step":
                result = state.backend.step(WebAction.model_validate(payload["action"]))
                return {"result": result.model_dump(mode="json")}
            if path == "/evaluate":
                result = state.backend.evaluate(payload.get("final_answer"))
                return {"result": result.model_dump(mode="json")}
            raise _Conflict(f"unknown endpoint {path}")

        def _reset(self, payload: dict[str, Any]) -> dict[str, Any]:
            session_id = str(payload["session_id"])
            if state.backend is not None and state.session_id != session_id:
                raise _Conflict(f"instance is serving session {state.session_id!r}")
            config = _load(payload["config_class"]).model_validate(payload["config"])
            task = WebTask.model_validate(payload["task"])
            if state.backend is None:
                display = os.environ.get("DISPLAY", "")
                lease = BrowserSessionHandle(
                    session_id=session_id,
                    provider_name="local_process",
                    transport="local_process",
                    endpoint=display or None,
                    metadata={"display_env": "DISPLAY", "display": display},
                    owner_pid=os.getpid(),
                )
                artifacts = WebArtifactStore(
                    config.resolved_artifact_dir(), inline_screenshots=config.inline_screenshots
                )
                state.backend = _load(payload["backend_factory"])(config, session_id, artifacts, lease)
                state.session_id = session_id
            try:
                observation, info = state.backend.reset(task)
            except BaseException:
                self._discard_backend()
                raise
            return {"observation": observation.model_dump(mode="json"), "info": info}

        def _close(self) -> dict[str, Any]:
            if state.backend is None:
                return {"closed": True}
            try:
                state.backend.close()
            except BaseException:
                # Browser handles may still be alive; this process can no
                # longer be trusted with a new session. The pool respawns it.
                state.exit_code = 1
                raise
            finally:
                state.backend = None
                state.session_id = None
            return {"closed": True}

        def _discard_backend(self) -> None:
            try:
                state.backend.close()
            except BaseException:  # noqa: BLE001 - the original failure is what the caller needs.
                state.exit_code = 1
            state.backend = None
            state.session_id = None

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class _Conflict(RuntimeError):
    pass


def _watch_parent(parent_pid: int) -> None:
    """Exit when the pool that spawned this instance is gone."""

    while True:
        if os.getppid() != parent_pid:
            LOG.error("event=browser_instance_parent_lost parent_pid=%d", parent_pid)
            os._exit(3)
        time.sleep(2.0)


def serve(*, host: str, port: int, token: str, parent_pid: int | None = None) -> int:
    state = _InstanceState(token)
    server = HTTPServer((host, port), _make_handler(state))
    if parent_pid is not None:
        threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()

    def stop(signum: int, _frame: Any) -> None:
        LOG.info("event=browser_instance_signal signal=%d", signum)
        state.exit_code = 0 if state.exit_code is None else state.exit_code
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, stop)
    LOG.info(
        "event=browser_instance_ready port=%d display=%s pid=%d", port, os.environ.get("DISPLAY", ""), os.getpid()
    )
    try:
        while state.exit_code is None:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        if state.backend is not None:
            try:
                state.backend.close()
            except BaseException:  # noqa: BLE001
                LOG.exception("event=browser_instance_shutdown_close_failed session=%s", state.session_id)
        server.server_close()
    return state.exit_code or 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    token = os.environ.pop(TOKEN_ENV, "")
    if not token:
        print(f"{TOKEN_ENV} is required", file=sys.stderr)
        return 2
    return serve(host=args.host, port=args.port, token=token, parent_pid=args.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
