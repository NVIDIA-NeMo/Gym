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

"""Run one Hermes rollout across its managed-runtime boundary.

Hermes's official installer creates a Python 3.11 environment with Hermes's
own dependency versions, while the Gym agent server runs Python 3.13 with
Gym's dependency constraints. Importing upstream ``AIAgent`` in the server
process would mix those incompatible environments. The server therefore
passes a rollout to this entrypoint as JSON, executes it with the
Hermes-installed interpreter, and reads the complete result back as JSON.
"""

import json
import logging
import signal
import sys
import traceback
from pathlib import Path
from typing import Any


def _write_response(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")


def _load_ai_agent():
    from run_agent import AIAgent  # pyright: ignore[reportMissingImports]

    return AIAgent


def _prepare_mcp_tools() -> None:
    """Discover configured MCP tools before AIAgent snapshots the tool registry."""
    from hermes_cli.mcp_startup import ensure_mcp_discovery_before_agent_build

    ensure_mcp_discovery_before_agent_build(
        logger=logging.getLogger(__name__),
        single_query=True,
        thread_name="nemo-gym-mcp-discovery",
    )


def run(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if request.get("mcp_enabled"):
        _prepare_mcp_tools()
    agent = _load_ai_agent()(
        base_url=request["base_url"],
        api_key=request["api_key"],
        model=request["model"],
        max_iterations=request["max_iterations"],
        max_tokens=request.get("max_tokens"),
        enabled_toolsets=request.get("enabled_toolsets"),
        disabled_toolsets=request.get("disabled_toolsets"),
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
        request_overrides=request.get("request_overrides") or {},
        save_trajectories=False,
    )
    # Gym owns rollout persistence. Hermes 0.20.5 replaced the public
    # persist_session constructor flag with this runtime switch.
    agent._persist_disabled = True
    # Gym consumes one completed result, and the prior adapter disabled
    # streaming. Hermes 0.20.5 replaced that constructor option with this flag.
    agent._disable_streaming = True
    observer = None
    observation_failure = None
    if request.get("capture_observations"):
        try:
            try:
                from .raw_observability import RawHermesObserver
            except ImportError:
                from raw_observability import RawHermesObserver  # type: ignore[no-redef]

            observer = RawHermesObserver().instrument(agent)
        except Exception as exc:
            observation_failure = {
                "source": "hermes",
                "invocations": [],
                "tools": [],
                "compactions": [],
                "gaps": [
                    {
                        "code": "observation_capture_failed",
                        "invocation_id": None,
                        "detail": type(exc).__name__,
                    }
                ],
            }

    def interrupt(_signum, _frame) -> None:
        agent.interrupt("timeout")

    signal.signal(signal.SIGTERM, interrupt)

    try:
        result = agent.run_conversation(
            request["user_message"],
            request.get("system_message"),
            request.get("history") or [],
        )
    finally:
        try:
            agent.close()
        except Exception:
            pass
    try:
        observations = observer.finish(result) if observer is not None else observation_failure
    except Exception as exc:
        observations = {
            "source": "hermes",
            "invocations": [],
            "tools": [],
            "compactions": [],
            "gaps": [
                {
                    "code": "observation_capture_failed",
                    "invocation_id": None,
                    "detail": type(exc).__name__,
                }
            ],
        }
    return result, observations


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: runner.py REQUEST_JSON RESPONSE_JSON", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result, observations = run(request)
        _write_response(
            response_path,
            {
                "result": result,
                "observations": observations,
            },
        )
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        _write_response(response_path, {"error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
