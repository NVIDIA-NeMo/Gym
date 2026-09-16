# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a complete NOOA rollout inside a Gym-managed sandbox."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from responses_api_agents.nooa_agent.app import NOOAAgentRunRequest
from responses_api_agents.nooa_agent.direct_client import DirectServerClient
from responses_api_agents.nooa_agent.runner import EmbeddedNOOARunner, NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.sandbox_protocol import SandboxRunArtifact, SandboxRunRequest, write_artifact


async def run_worker(request_path: Path, result_path: Path, checkpoint_path: Path) -> int:
    request = SandboxRunRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    row = NOOAAgentRunRequest.model_validate(request.row)
    client = DirectServerClient(request.endpoints)

    def checkpoint(result: NOOARunResult) -> None:
        write_artifact(checkpoint_path, SandboxRunArtifact.from_result(result, complete=False))

    runner = EmbeddedNOOARunner(
        invocation=request.invocation,
        server_client=client,
        model_server_name=request.model_server_name,
        resources_server_name=request.resources_server_name,
        max_steps=request.max_steps,
        on_checkpoint=checkpoint,
    )
    run_request = NOOARunRequest(
        row=row,
        model_url_path=request.model_url_path,
        model_cookies=dict(request.model_cookies),
        resource_cookies=dict(request.resource_cookies),
        task_id=request.task_id,
        rollout_id=request.rollout_id,
    )
    try:
        result = await runner.run(run_request)
    except NOOARunFailure as error:
        artifact = SandboxRunArtifact.from_result(error.result, complete=True, error=error.__cause__ or error)
        write_artifact(checkpoint_path, artifact)
        write_artifact(result_path, artifact)
        return 1
    except BaseException as error:
        artifact = SandboxRunArtifact(error_type=type(error).__name__, error=str(error), complete=True)
        write_artifact(checkpoint_path, artifact)
        write_artifact(result_path, artifact)
        return 1

    artifact = SandboxRunArtifact.from_result(result, complete=True)
    write_artifact(checkpoint_path, artifact)
    write_artifact(result_path, artifact)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(run_worker(args.request, args.result, args.checkpoint))


if __name__ == "__main__":
    raise SystemExit(main())
