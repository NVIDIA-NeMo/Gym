# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-task OSWorld setup/evaluate subprocess entry (imports the pinned ``osworld`` fork).

The resources server shells out to this script (with its own venv's ``sys.executable``)
because the fork's remote-provider addressing is env-var-global
(``OSWORLD_CONTROL_SERVER_URL`` / ``OSWORLD_REMOTE_ADDR``): concurrent rollouts against
different sandboxes cannot share a process. Each invocation is one phase of one task:

- ``--phase setup``    -> ``DesktopEnv(provider_name="remote").reset(task_config)`` — the
  official reset path (setup steps with official semantics, incl. proxied/CDP handling).
- ``--phase evaluate`` -> ``_set_task_info`` + ``action_history`` + ``env.evaluate()`` — the
  official getter->metric dispatch (evaluate-at-max-steps semantics live in the caller,
  which always invokes this phase).

Addressing modes:
- ``--control-url http://<pod-ip>:5000`` (direct): host:port-expressible; the remote
  provider's ``OSWORLD_REMOTE_ADDR`` carries all ports (5000/9222/vnc/vlc) directly.
- ``--control-url http://<domain>/sandboxes/<id>/proxy/5000 --proxied``: path-based proxy;
  local forwarders (``local_forwarder.py``) give the upstream code plain host:port targets
  and inject any required route ``--headers-json``.

Contract (stdout): exactly one line ``__NEMO_GYM_OSWORLD__ {"ok": bool, "score": float|null,
"error": str|null}``.
"""

import argparse
import json
import os
import sys
import time
import traceback


SENTINEL = "__NEMO_GYM_OSWORLD__"

VNC_PORT = 8006
VLC_PORT = 8080


def _emit(ok: bool, score: float | None = None, error: str | None = None) -> None:
    sys.stdout.write(f"{SENTINEL} {json.dumps({'ok': ok, 'score': score, 'error': error})}\n")
    sys.stdout.flush()


def _configure_remote_addressing(control_url: str, proxied: bool, headers: dict[str, str]) -> None:
    """Point the fork's remote provider at this task's sandbox (before importing desktop_env)."""
    control_url = control_url.rstrip("/")
    if not proxied:
        # http://<host>:5000 -> host; all guest ports are directly reachable on that host.
        hostport = control_url.split("://", 1)[-1]
        host, _, port = hostport.partition(":")
        server_port = port or "5000"
        os.environ["OSWORLD_CONTROL_SERVER_URL"] = control_url
        os.environ["OSWORLD_REMOTE_ADDR"] = f"{host}:{server_port}:9222:{VNC_PORT}:{VLC_PORT}"
        return

    from resources_servers.osworld.local_forwarder import start_forwarder

    # http://<domain>/sandboxes/<id>/proxy/5000 -> proxy root, then one forwarder per port.
    if not control_url.endswith("/5000"):
        raise ValueError(f"proxied control URL must end with /5000, got: {control_url}")
    proxy_root = control_url[: -len("/5000")]
    _, port_5000 = start_forwarder(f"{proxy_root}/5000", headers)
    _, port_9222 = start_forwarder(f"{proxy_root}/9222", headers)
    # VLC's HTTP interface (:8080) is queried directly by the vlc getters
    # (vlc_playing_info -> http://<ip>:<vlc_port>/requests/status.xml) — forward it too,
    # or those evaluations dial a dead local port and crash (vlc-gate-v1: 2x evaluate_exception).
    _, port_8080 = start_forwarder(f"{proxy_root}/8080", headers)
    os.environ["OSWORLD_CONTROL_SERVER_URL"] = f"http://127.0.0.1:{port_5000}"
    os.environ["OSWORLD_REMOTE_ADDR"] = f"127.0.0.1:{port_5000}:{port_9222}:{VNC_PORT}:{port_8080}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["setup", "evaluate"])
    parser.add_argument("--config", required=True, help="Path to the task config JSON (verifier_metadata)")
    parser.add_argument("--control-url", required=True, help="Guest :5000 control base URL")
    parser.add_argument("--proxied", action="store_true", help="control-url goes through a path-based proxy")
    parser.add_argument("--headers-json", default="{}", help="Route headers required by the proxy")
    parser.add_argument("--cache", required=True, help="Setup download cache dir")
    parser.add_argument("--action-history", default="[]", help="JSON list; evaluate phase only")
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--client-password", default="password")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            task_config = json.load(f)
        headers = json.loads(args.headers_json) or {}
        action_history = json.loads(args.action_history)
        if not isinstance(action_history, list):
            action_history = []
    except Exception as e:  # noqa: BLE001 - bad inputs are a clean failure, not a crash
        _emit(False, error=f"bad inputs: {e!r}")
        return 0

    try:
        _configure_remote_addressing(args.control_url, args.proxied, headers)
        os.makedirs(args.cache, exist_ok=True)

        from desktop_env.desktop_env import DesktopEnv

        env = DesktopEnv(
            provider_name="remote",
            action_space="pyautogui",
            cache_dir=args.cache,
            screen_size=(args.screen_w, args.screen_h),
            client_password=args.client_password,
            require_a11y_tree=False,
            require_terminal=False,
        )

        if args.phase == "setup":
            env.reset(task_config=task_config)
            _emit(True)
        else:
            env._set_task_info(task_config)
            # _set_task_info points env.cache_dir at <base>/<task_id>, but the SetupController
            # (which executes evaluator postconfig steps and writes their `stdout:` captures)
            # still holds the base dir from __init__ — reset() is what normally syncs them, and
            # the evaluate phase never calls reset(). Without this, every cache_file getter
            # whose file comes from a postconfig capture raises FileNotFoundError
            # (= the thunderbird "evaluate_exception" infra crashes).
            env.setup_controller.reset_cache_dir(env.cache_dir)
            env.action_history = action_history
            # The guest control API occasionally answers a transient 500 mid-evaluation
            # (observed as "Internal Server Error for url: .../platform"); one such blip
            # otherwise scores a solved task 0 with `evaluate_exception`. Retry briefly.
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    score = env.evaluate()
                    break
                except Exception as exc:  # noqa: BLE001 - inspect and re-raise if not transient
                    if "Internal Server Error" not in str(exc) or attempt == 2:
                        raise
                    last_exc = exc
                    sys.stderr.write(f"evaluate transient 5xx (attempt {attempt + 1}/3): {exc}\n")
                    time.sleep(10)
            _emit(True, score=float(score))
    except Exception:  # noqa: BLE001 - report, don't crash: the caller decides the consequence
        sys.stderr.write(f"eval_task ({args.phase}) failed:\n" + traceback.format_exc())
        _emit(False, score=0.0 if args.phase == "evaluate" else None, error=f"{args.phase}_exception")
    return 0


if __name__ == "__main__":
    sys.exit(main())
