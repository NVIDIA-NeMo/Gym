# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unmodified benchmark task with the upstream Harbor CLI."""

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def container_endpoint(endpoint: str) -> str:
    """Reach a host-local Factory proxy through the rootless network gateway."""
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Model endpoint must be an HTTP(S) URL")
    if parts.username or parts.password:
        raise ValueError("Model endpoint must not embed credentials")
    if parts.hostname in {"localhost", "127.0.0.1", "::1"}:
        authority = "10.0.2.2" + (f":{parts.port}" if parts.port else "")
        parts = parts._replace(netloc=authority)
    return urlunsplit(parts).rstrip("/")


def job_config(
    *,
    task_path: Path,
    output: Path,
    runtime_root: Path,
    model: str,
    endpoint: str,
    context_tokens: int,
    output_tokens: int,
    backend: str = "podman",
) -> dict:
    """Create a single-attempt Harbor job using OpenCode's compatible provider."""
    if not 0 < output_tokens < context_tokens:
        raise ValueError("Output token limit must be positive and smaller than context")
    if backend not in {"podman", "remote"}:
        raise ValueError("Backend must be podman or remote")
    environment_class = (
        "remote_environment:RemoteEnvironment" if backend == "remote" else "podman_environment:PodmanEnvironment"
    )
    if backend == "remote" and urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote task backend requires a network-reachable Factory proxy address")
    return {
        "job_name": "trial",
        "jobs_dir": str(output / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "environment": {
            "import_path": "responses_api_agents.agentic_vbench_agent." + environment_class,
            "delete": True,
            "kwargs": {"runtime_root": str(runtime_root)},
        },
        "tasks": [{"path": str(task_path)}],
        "agents": [
            {
                "name": "opencode",
                "model_name": f"openai/{model}",
                "kwargs": {
                    "version": "1.14.39",
                    "opencode_config": {
                        "provider": {
                            "openai": {
                                "npm": "@ai-sdk/openai-compatible",
                                "options": {
                                    "baseURL": endpoint.rstrip("/")
                                    if backend == "remote"
                                    else container_endpoint(endpoint),
                                    "apiKey": "unused",
                                },
                                "models": {
                                    model: {
                                        "name": model,
                                        "limit": {"context": context_tokens, "output": output_tokens},
                                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                                    }
                                },
                            }
                        },
                    },
                },
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--credentials-file", type=Path)
    args = parser.parse_args()
    if importlib.metadata.version("harbor") != "0.6.6":
        raise RuntimeError("AgenticVBench requires upstream harbor==0.6.6")
    from harbor.models.job.config import JobConfig

    config = JobConfig.model_validate(
        job_config(
            task_path=args.task_path.resolve(strict=True),
            output=args.output.resolve(strict=True),
            runtime_root=args.runtime_root.resolve(),
            model=args.model,
            endpoint=args.endpoint,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
            backend=os.environ.get("AGENTIC_VBENCH_BACKEND", "podman"),
        )
    )
    config_path = args.output / "harbor_job.json"
    with config_path.open("x") as handle:
        handle.write(config.model_dump_json(indent=2) + "\n")
    command = [str(Path(sys.executable).with_name("harbor")), "run", "--config", str(config_path)]
    if args.credentials_file:
        command.extend(["--env-file", str(args.credentials_file.resolve(strict=True))])
    env = dict(os.environ)
    gym_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [gym_root, env.get("PYTHONPATH")]))
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
