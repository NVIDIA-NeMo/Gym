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
"""Capture-proxy sidecar entrypoint.

Runs the capture (+ optional Anthropic→OpenAI translation) proxy in a sidecar
container sharing the sandbox's network namespace: the agent reaches it on
``localhost`` (no reverse tunnel) while the real upstream key stays only in this
container's env. Captures land on a volume shared with the agent container.

Reuses :func:`start_capture_proxy`; the sidecar only adds process lifecycle and an
env-var config contract:

    NEMO_GYM_SIDECAR_PORT                 (required) localhost port to listen on
    NEMO_GYM_SIDECAR_UPSTREAM             (required) model base URL to forward to
    NEMO_GYM_SIDECAR_SESSION_ID           (required) rollout/session key
    NEMO_GYM_SIDECAR_OUT_DIR              capture dir on the shared volume
    NEMO_GYM_SIDECAR_API_KEY              real upstream key (isolated here)
    NEMO_GYM_SIDECAR_TRANSLATE_ANTHROPIC  "1"/"true" to translate Anthropic↔OpenAI
    NEMO_GYM_SIDECAR_MODEL                translate model override
    NEMO_GYM_SIDECAR_INJECT_JSON          JSON extra-body to inject upstream
    NEMO_GYM_SIDECAR_HOST                 bind host (default 127.0.0.1)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading

from nemo_gym.adapters.sandbox_capture import start_capture_proxy


LOG = logging.getLogger("nemo_gym.sidecar")

HEALTH_PATH = "/_proxy_health"


def _bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"sidecar: required env var {name} is missing")
    return value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    inject_raw = os.environ.get("NEMO_GYM_SIDECAR_INJECT_JSON") or "{}"
    try:
        inject = json.loads(inject_raw)
    except json.JSONDecodeError:
        LOG.warning("ignoring invalid NEMO_GYM_SIDECAR_INJECT_JSON: %r", inject_raw)
        inject = {}

    proxy = start_capture_proxy(
        model_base_url=_require("NEMO_GYM_SIDECAR_UPSTREAM"),
        session_id=_require("NEMO_GYM_SIDECAR_SESSION_ID"),
        store_dir=os.environ.get("NEMO_GYM_SIDECAR_OUT_DIR", "/nemo-capture"),
        host=os.environ.get("NEMO_GYM_SIDECAR_HOST", "127.0.0.1"),
        port=int(_require("NEMO_GYM_SIDECAR_PORT")),
        inject_extra_body=inject,
        upstream_api_key=os.environ.get("NEMO_GYM_SIDECAR_API_KEY") or None,
        translate_anthropic=_bool(os.environ.get("NEMO_GYM_SIDECAR_TRANSLATE_ANTHROPIC")),
        translate_model_override=os.environ.get("NEMO_GYM_SIDECAR_MODEL") or None,
    )
    LOG.info("sidecar capture proxy ready on %s", proxy.handle.url)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        stop.wait()
    finally:
        proxy.stop()
        LOG.info("sidecar capture proxy stopped")


if __name__ == "__main__":
    main()
