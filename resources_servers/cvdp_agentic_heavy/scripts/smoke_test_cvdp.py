#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test CVDP conversion artifacts and the verifier container path.

This script does not call a model endpoint. It is intended as a fast customer
preflight before running a real NeMo-Gym rollout:

1. Optionally validate a converted CVDP JSONL file for leakage-sensitive fields.
2. Run a tiny synthetic harness that should pass.
3. Optionally run the first official converted row as a no-op attempt. That row
   usually fails, but proving the hidden harness runs is the useful signal.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient  # noqa: E402
from resources_servers.cvdp_agentic_heavy.app import (  # noqa: E402
    AgenticHeavySeedRequest,
    CVDPAgenticHeavyConfig,
    CVDPAgenticHeavyResourcesServer,
    CVDPAgenticHeavyVerifyRequest,
)


COMMERCIAL_EDA_RE = re.compile(r"__VERIF_EDA_IMAGE__|xrun|xcelium|licnetwork", re.IGNORECASE)


class Request:
    def __init__(self, session_id: str):
        self.session = {SESSION_ID_KEY: session_id}


def _verify_body(meta: dict[str, Any]) -> CVDPAgenticHeavyVerifyRequest:
    return CVDPAgenticHeavyVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "smoke"}]},
            "response": {
                "id": "resp_smoke",
                "created_at": 1000000,
                "model": "smoke-model",
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_smoke",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
            "verifier_metadata": meta,
        }
    )


def _select_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if shutil.which("apptainer"):
        return "apptainer"
    if shutil.which("docker"):
        return "docker"
    raise RuntimeError("Neither apptainer nor docker was found on PATH.")


def _check_backend(backend: str) -> None:
    executable = "apptainer" if backend == "apptainer" else "docker"
    if not shutil.which(executable):
        raise RuntimeError(f"{executable} is required for --backend {backend}, but it was not found on PATH.")


def _make_server(backend: str, timeout: int) -> CVDPAgenticHeavyResourcesServer:
    config = CVDPAgenticHeavyConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cvdp_agentic_heavy",
        execution_backend=backend,
        container_timeout=timeout,
        tool_timeout=timeout,
        num_processes=1,
    )
    return CVDPAgenticHeavyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def _run_one(
    server: CVDPAgenticHeavyResourcesServer,
    session_id: str,
    meta: dict[str, Any],
) -> tuple[float, int | None, list[dict[str, Any]]]:
    await server.seed_session(Request(session_id), AgenticHeavySeedRequest(verifier_metadata=meta))
    result = await server.verify(Request(session_id), _verify_body(meta))
    return result.reward, result.container_exit_code, result.container_services or []


def _synthetic_meta() -> dict[str, Any]:
    return {
        "task_id": "cvdp_synthetic_pass",
        "categories": ["smoke", "easy"],
        "difficulty": "easy",
        "context_files": {"pass.txt": "ok\n"},
        "harness_files": {
            "docker-compose.yml": (
                "services:\n"
                "  sanity:\n"
                "    image: __OSS_SIM_IMAGE__\n"
                "    command: sh -lc \"test -f /code/pass.txt && grep ok /code/pass.txt\"\n"
            )
        },
        "origin": {"source": "smoke_test_cvdp.py"},
    }


def validate_jsonl(path: Path) -> tuple[int, int, int]:
    rows = 0
    rows_with_patch = 0
    commercial_rows = 0
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            meta = row.get("verifier_metadata", {})
            if "patch" in meta:
                rows_with_patch += 1
            compose = (meta.get("harness_files") or {}).get("docker-compose.yml", "")
            if COMMERCIAL_EDA_RE.search(compose):
                commercial_rows += 1
    return rows, rows_with_patch, commercial_rows


def load_first_row(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"No rows found in {path}")


async def async_main(args: argparse.Namespace) -> int:
    backend = _select_backend(args.backend)
    _check_backend(backend)
    print(f"backend={backend}")

    if args.converted_jsonl:
        path = Path(args.converted_jsonl)
        rows, rows_with_patch, commercial_rows = validate_jsonl(path)
        print(f"converted_jsonl={path}")
        print(f"rows={rows}")
        print(f"rows_with_patch={rows_with_patch}")
        print(f"commercial_eda_rows={commercial_rows}")
        if rows == 0:
            raise RuntimeError("Converted JSONL contains no usable rows.")
        if rows_with_patch and not args.allow_solution_metadata:
            raise RuntimeError("Converted JSONL contains patch metadata. Re-run conversion without --include-solution-metadata.")
        if commercial_rows:
            raise RuntimeError("Converted JSONL still contains commercial EDA/Xcelium harness services.")

    server = _make_server(backend, args.timeout)

    reward, exit_code, services = await _run_one(server, "synthetic-pass", _synthetic_meta())
    print(f"synthetic_reward={reward}")
    print(f"synthetic_exit_code={exit_code}")
    print(f"synthetic_services={[(s.get('service'), s.get('exit_code')) for s in services]}")
    if reward != 1.0 or exit_code != 0:
        raise RuntimeError("Synthetic verifier smoke did not pass.")

    if args.converted_jsonl and args.run_official_noop:
        row = load_first_row(Path(args.converted_jsonl))
        meta = row["verifier_metadata"]
        reward, exit_code, services = await _run_one(server, "official-noop", meta)
        print(f"official_task={meta.get('task_id')}")
        print(f"official_noop_reward={reward}")
        print(f"official_noop_exit_code={exit_code}")
        print(f"official_services={[(s.get('service'), s.get('exit_code')) for s in services]}")
        if not services:
            raise RuntimeError("Official CVDP row did not run any verifier services.")

    print("smoke_test=passed")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "apptainer", "docker"], default="auto")
    parser.add_argument("--timeout", type=int, default=180, help="Container timeout in seconds.")
    parser.add_argument("--converted-jsonl", default="", help="Optional converted CVDP JSONL to validate.")
    parser.add_argument(
        "--run-official-noop",
        action="store_true",
        help="Run the first converted official row without applying a fix. The reward may be 0.",
    )
    parser.add_argument(
        "--allow-solution-metadata",
        action="store_true",
        help="Allow patch metadata when validating a JSONL file. Keep this off for rollout datasets.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
