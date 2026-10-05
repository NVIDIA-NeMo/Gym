# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unmodified benchmark task with the upstream Harbor CLI."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


OPENCODE_OUTPUT_CAP = "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"


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
    max_turns: int | None = None,
    verifier_timeout_multiplier: float = 1.0,
) -> dict:
    """Create a single-attempt Harbor job using OpenCode's compatible provider.

    ``max_turns`` caps OpenCode's agentic iterations (its ``build`` agent ``steps``);
    after the cap OpenCode must give a final text-only response.
    ``verifier_timeout_multiplier`` scales only the task's verifier timeout: some
    single-threaded judges need longer than their 1800 s on this hardware. The agent's
    budget and the scores are unaffected.
    """
    if not 0 < output_tokens < context_tokens:
        raise ValueError("Output token limit must be positive and smaller than context")
    if backend not in {"podman", "remote"}:
        raise ValueError("Backend must be podman or remote")
    environment_class = (
        "remote_environment:RemoteEnvironment" if backend == "remote" else "podman_environment:PodmanEnvironment"
    )
    if backend == "remote" and urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Remote task backend requires a network-reachable Factory proxy address")
    if max_turns is not None and max_turns < 1:
        raise ValueError("Maximum agent turns must be positive")
    if verifier_timeout_multiplier < 1.0:
        raise ValueError("Verifier timeout multiplier must not shorten the task's own budget")
    opencode_config = {
        "provider": {
            "openai": {
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": endpoint.rstrip("/") if backend == "remote" else container_endpoint(endpoint),
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
    }
    if max_turns is not None:
        opencode_config["agent"] = {"build": {"steps": max_turns}}
    return {
        "job_name": "trial",
        "jobs_dir": str(output / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "verifier_timeout_multiplier": verifier_timeout_multiplier,
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
                "kwargs": {"version": "1.14.39", "opencode_config": opencode_config},
                # OpenCode reserves min(model output limit, this cap; default 32000) when deciding
                # to compact. The Factory adapter sends max_tokens equal to the output limit, so the
                # reservation must match it or prompts overflow the context before compaction.
                # Harbor redacts literal values of *TOKEN* keys when it writes the job config, so
                # pass a template that Harbor resolves from the environment main() provides.
                "env": {OPENCODE_OUTPUT_CAP: f"${{{OPENCODE_OUTPUT_CAP}}}"},
            }
        ],
    }


JUDGE_PROTOCOLS = {
    "nvinference-hybrid": {
        "AVB_JUDGE_BASE_URL": "https://inference-api.nvidia.com",
        "AVB_JUDGE_CLAUDE_MODEL": "us/aws/anthropic/eccn-claude-opus-4-7",
        "AVB_JUDGE_GEMINI_MODEL": "us/gcp/google/eccn-gemini-3.5-flash",
        "AVB_JUDGE_GEMINI_THINKING": "low",
    }
}
NATIVE_JUDGE_ENV = {"ANTHROPIC_API_KEY", "GEMINI_API_KEY"}
# Informational scan in the repurpose verifier; inaccessible /proc entries in
# rootless containers must not abort the verifier under pipefail.
TYPING_DIAGNOSTIC = "find / -name 'typing_extensions*' 2>/dev/null | head -20"


def stage_judge(task_path: Path, *, staging: Path, output: Path, protocol: str) -> Path:
    """Copy a natively judged task with its judge routed through NVIDIA Inference Hub.

    Tasks without native judge credentials are returned unchanged. The Hub key is
    exposed only to the verifier, never to the agent's container environment.
    """
    if protocol not in JUDGE_PROTOCOLS:
        raise ValueError(f"Unknown judge protocol: {protocol}")
    text = (task_path / "task.toml").read_text()
    task = tomllib.loads(text)
    if set(task.get("environment", {}).get("env", {})) != NATIVE_JUDGE_ENV:
        return task_path
    if len(task["steps"]) != 1 or task["steps"][0].get("verifier", {}).get("env"):
        raise ValueError(f"Unexpected judged task layout: {task_path}")
    staged = staging / task_path.name
    shutil.copytree(task_path, staged)
    start, end = text.index("[environment.env]"), text.index("[[steps]]")
    settings = JUDGE_PROTOCOLS[protocol]
    text = text[:start] + text[end:] + '\n[steps.verifier.env]\nNVINFERENCE_API_KEY = "${NVINFERENCE_API_KEY}"\n'
    text += "".join(f'{key} = "{value}"\n' for key, value in settings.items())
    (staged / "task.toml").write_text(text)
    tests = staged / "steps" / task["steps"][0]["name"] / "tests"
    verifier = tests / "test.sh"
    verifier.write_text(verifier.read_text().replace(TYPING_DIAGNOSTIC, TYPING_DIAGNOSTIC + " || true"))
    (tests / "judge.py").rename(tests / "avb-original-judge.py")
    shutil.copyfile(Path(__file__).with_name("hub_judge.py"), tests / "judge.py")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (output / "judge_protocol.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "settings": settings,
                "source_task": str(task_path),
                "original_sha256": {
                    name: digest(task_path / relative)
                    for name, relative in {
                        "task.toml": "task.toml",
                        "test.sh": verifier.relative_to(staged),
                        "judge.py": (tests / "judge.py").relative_to(staged),
                    }.items()
                },
                "staged_sha256": {
                    "task.toml": digest(staged / "task.toml"),
                    "test.sh": digest(verifier),
                    "judge.py": digest(tests / "judge.py"),
                    "avb-original-judge.py": digest(tests / "avb-original-judge.py"),
                },
            },
            indent=2,
        )
        + "\n"
    )
    return staged


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

    task_path = args.task_path.resolve(strict=True)
    protocol = os.environ.get("AGENTIC_VBENCH_JUDGE_PROTOCOL", "")
    if protocol:
        from dotenv import dotenv_values

        credentials = dotenv_values(args.credentials_file) if args.credentials_file else {}
        if not (credentials.get("NVINFERENCE_API_KEY") or os.environ.get("NVINFERENCE_API_KEY")):
            raise ValueError(f"Judge protocol {protocol} requires NVINFERENCE_API_KEY in the credentials file")
        task_path = stage_judge(
            task_path, staging=args.runtime_root.resolve() / "judge-task", output=args.output, protocol=protocol
        )

    config = JobConfig.model_validate(
        job_config(
            task_path=task_path,
            output=args.output.resolve(strict=True),
            runtime_root=args.runtime_root.resolve(),
            model=args.model,
            endpoint=args.endpoint,
            context_tokens=args.context_tokens,
            output_tokens=args.output_tokens,
            backend=os.environ.get("AGENTIC_VBENCH_BACKEND", "podman"),
            max_turns=int(os.environ["AGENTIC_VBENCH_MAX_TURNS"])
            if os.environ.get("AGENTIC_VBENCH_MAX_TURNS")
            else None,
            verifier_timeout_multiplier=float(os.environ.get("AGENTIC_VBENCH_VERIFIER_TIMEOUT_MULTIPLIER", "1")),
        )
    )
    config_path = args.output / "harbor_job.json"
    with config_path.open("x") as handle:
        handle.write(config.model_dump_json(indent=2) + "\n")
    # Tasks that declare [environment.env] variables (e.g. HF_TOKEN) make Harbor ask
    # for confirmation. Confirm non-interactively and close stdin so no prompt can block.
    command = [str(Path(sys.executable).with_name("harbor")), "run", "--config", str(config_path), "--yes"]
    if args.credentials_file:
        command.extend(["--env-file", str(args.credentials_file.resolve(strict=True))])
    env = dict(os.environ)
    env[OPENCODE_OUTPUT_CAP] = str(args.output_tokens)
    gym_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [gym_root, env.get("PYTHONPATH")]))
    run_harbor(command, env=env, jobs_dir=args.output.resolve() / "jobs")


DOCKER_CHECK_FAILURE = "Docker daemon is not running"
HARBOR_START_ATTEMPTS = 4
HARBOR_START_BACKOFF_SECONDS = 30.0


def run_harbor(command: list[str], *, env: dict, jobs_dir: Path, sleep=time.sleep) -> None:
    """Run Harbor, retrying only its start-up Docker probe failure.

    Harbor checks the daemon with a single ``docker info`` under a 10 s timeout before
    any trial starts; on a loaded remote backend that probe times out and Harbor exits
    with ``Docker daemon is not running`` although the backend is healthy. The retry is
    limited to that message while no trial directory exists, so a trial is never run twice.
    """
    for attempt in range(1, HARBOR_START_ATTEMPTS + 1):
        result = subprocess.run(
            command, env=env, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace"
        )
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()
        if result.returncode == 0:
            return
        trial_started = jobs_dir.is_dir() and any(jobs_dir.glob("*/*__*"))
        if DOCKER_CHECK_FAILURE not in result.stderr or trial_started or attempt == HARBOR_START_ATTEMPTS:
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f"Harbor Docker probe failed before any trial started; retry {attempt}/{HARBOR_START_ATTEMPTS - 1}")
        sleep(HARBOR_START_BACKOFF_SECONDS * attempt)


if __name__ == "__main__":
    main()
