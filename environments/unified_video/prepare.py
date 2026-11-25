# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare ClawEval tasks for session harnesses; other benchmarks use their stock Gym preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import time
import tomllib
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

import yaml

from nemo_gym.core import ModelCallSampling
from nemo_gym.data.tasks.harbor import HarborTask
from responses_api_agents.claweval_agent.core import inventory as claw_inventory
from responses_api_agents.claweval_agent.core import stage_harbor_task


CLAW = "claweval_agent"
CLAW_HARNESS_TASKSETS: dict[str, str] = {
    "opencode": "claweval_opencode",
    "codex": "claweval_codex",
    "hermes": "claweval_hermes",
}
BACKEND_CLIENT_ENV = "client.env"
TRAIN_DATASET_LICENSES = {
    CLAW: "NVIDIA Internal Use Only, Do Not Distribute",
}
CLAW_MEDIA_OUTPUT_BYTES = 1024**3
CLAW_MEDIA_SERVER = {"name": "claw", "transport": "streamable-http", "url": "http://127.0.0.1:17654/mcp"}


class _PreparedTasks(Protocol):
    tasks: list[HarborTask]
    folder: Path


def backend_environment(
    backend_dir: Path, wait_seconds: float = 1800.0, *, poll_seconds: float = 15.0
) -> dict[str, str]:
    """Wait for the selected Docker backend's client settings and use its local TLS directory."""
    backend_dir = backend_dir.resolve()
    path = backend_dir / BACKEND_CLIENT_ENV
    deadline = time.monotonic() + wait_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"No Docker backend client settings at {path} after {wait_seconds:.0f}s")
        time.sleep(poll_seconds)
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").strip().partition("=")
        if not separator or not key.isidentifier():
            raise ValueError(f"Malformed backend client setting in {path}")
        parts = shlex.split(value)
        values[key] = parts[0] if parts else ""
    if not values.get("DOCKER_HOST"):
        raise ValueError(f"{path} does not define DOCKER_HOST")
    return {
        "DOCKER_HOST": values["DOCKER_HOST"],
        "DOCKER_TLS_VERIFY": values.get("DOCKER_TLS_VERIFY", "1"),
        "DOCKER_CERT_PATH": str(backend_dir),
        "DOCKER_BUILDKIT": values.get("DOCKER_BUILDKIT", "0"),
    }


def prompt_text(row: dict) -> str:
    """Read the same single, verbatim user prompt admitted by the existing adapters."""
    messages = row["responses_create_params"]["input"]
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Expected exactly one user prompt")
    content = messages[0].get("content")
    if isinstance(content, list):
        if len(content) != 1 or content[0].get("type") != "input_text":
            raise ValueError("Initial media must be staged in the task sandbox")
        content = content[0].get("text")
    if not isinstance(content, str) or not content:
        raise ValueError("Expected a nonempty text prompt")
    return content


def set_toml_values(path: Path, section: str, values: dict) -> None:
    """Update scalar deployment settings without rewriting task/verifier definitions."""
    text = path.read_text()
    header = re.compile(r"(?m)^\[" + re.escape(section) + r"\]\s*$")
    match = header.search(text)
    if match is None:
        text += f"\n[{section}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items())
    else:
        end = re.search(r"(?m)^\[", text[match.end() :])
        stop = match.end() + end.start() if end else len(text)
        body = text[match.end() : stop]
        for key, value in values.items():
            line = f"{key} = {json.dumps(value)}"
            pattern = re.compile(r"(?m)^" + re.escape(key) + r"\s*=.*$")
            body = pattern.sub(lambda _: line, body) if pattern.search(body) else body + "\n" + line + "\n"
        text = text[: match.end()] + body + text[stop:]
    tomllib.loads(text)  # Refuse malformed TOML before touching the staged task.
    path.write_text(text)


def _task_metadata(task: Path, metadata: dict, *, source: str) -> None:
    # Keep native Dockerfile, task workdir and step upload/setup semantics unchanged.
    path = task / "task_data.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError("Task data must be an object")
    additions = {"video_source": source, "verifier_metadata": metadata}
    if any(key in existing and existing[key] != value for key, value in additions.items()):
        raise ValueError("Task metadata conflicts with the video provenance")
    path.write_text(json.dumps(existing | additions) + "\n")


def stage_claw(row: dict, task: dict, *, source: Path, destination: Path, environment_image: str | None) -> Path:
    """Retain the native Claw grader and task-local media tools for session harnesses."""
    metadata = row["verifier_metadata"]
    if any(metadata.get(k) != task[k] for k in ("task_variant", "family", "task_package_sha256")):
        raise ValueError(f"Mismatched Claw metadata: {metadata.get('task_id')}")
    if not metadata.get("source_id", "").startswith(f"{task['task_variant']}__"):
        raise ValueError("Claw source_id and task variant disagree")
    prompt = prompt_text(row)
    if hashlib.sha256(prompt.encode()).hexdigest() != metadata["prompt_sha256"]:
        raise ValueError("Claw prompt hash mismatch")
    stage_harbor_task(task, prompt, destination, claw_eval_source=source, environment_image=environment_image)
    _task_metadata(destination, metadata, source=CLAW)
    # Full 256-frame 1280x720 PNG replies can exceed the standalone default
    # of 256 MiB. Keep the decoded pixels and frame allowance unchanged.
    set_toml_values(
        destination / "task.toml", "environment.env", {"CLAW_MEDIA_MAX_OUTPUT_BYTES": str(CLAW_MEDIA_OUTPUT_BYTES)}
    )
    tools = destination / "steps/solve/workdir/.claw_tools"
    module_root = Path(__file__).resolve().parents[2] / "responses_api_agents/claweval_agent"
    for filename in ("mcp_http.py", "start_media_http.py"):
        shutil.copy2(module_root / filename, tools / filename)
    setup = destination / "steps/solve/workdir/setup.sh"
    if setup.exists():
        raise ValueError("Refuse to replace a task-owned setup.sh")
    setup.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "CLAW_MEDIA_MAX_OUTPUT_BYTES=1073741824 python3 .claw_tools/start_media_http.py\n"
    )
    # The task owns its MCP declaration. A harness receives an explicit required
    # grant; unsupported harnesses refuse it before any model call.
    with (destination / "task.toml").open("a") as stream:
        stream.write(
            '\n[[environment.mcp_servers]]\nname = "claw"\ntransport = "streamable-http"\n'
            'url = "http://127.0.0.1:17654/mcp"\n'
        )
    return destination


@contextmanager
def client_environment(values: dict[str, str]) -> Iterator[None]:
    """Select a Docker endpoint for serial preparation subprocesses, then restore it."""
    previous = {key: os.environ.get(key) for key in (*values, "DOCKER_CONTEXT")}
    os.environ.pop("DOCKER_CONTEXT", None)
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def training_dataset_registration(source: str, rows: Path) -> dict[str, object]:
    """Preserve the shipped dataset's metadata when changing its harness."""
    if source != CLAW:
        raise ValueError("This preparer handles ClawEval; use stock Harbor preparation for Agentic-VBench")
    return {
        "name": "train",
        "type": "train",
        "jsonl_fpath": str(rows),
        "num_repeats": 1,
        "license": TRAIN_DATASET_LICENSES[source],
    }


def component_config(
    source: str,
    *,
    prepared: _PreparedTasks,
    rows: Path,
    context_tokens: int,
    output_tokens: int,
    max_steps: int,
) -> dict[str, object]:
    """Wire one taskset to session harnesses using the configured Docker client."""
    from nemo_gym.data.tasks.harbor import taskset_mapping

    if source != CLAW:
        raise ValueError("This preparer handles ClawEval; use stock Harbor preparation for Agentic-VBench")
    environment_name = "claweval_environment_server"
    provider_ref = "claw_sandbox"
    config = {
        provider_ref: {
            "docker": {
                "create": {
                    "apply_resource_limits": False,
                    "use_init": False,
                },
                "exec": {"concurrency": 48},
            }
        },
        source: {
            "responses_api_agents": {
                "opencode_agent": {
                    "entrypoint": "app.py",
                    "token_id_capture": True,
                    "num_workers": 1,
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "model": "${policy_model_name}",
                    "context_window": context_tokens,
                    "max_output_tokens": output_tokens,
                    "image_input": True,
                    "timeout": 7200,
                    "concurrency": 12,
                    "opencode_config": {
                        "compaction": {"auto": False, "prune": False},
                        "agent": {"build": {"steps": max_steps}},
                        "permission": {"*": "allow", "task": "deny", "webfetch": "deny", "websearch": "deny"},
                        "tools": {"task": False},
                    },
                    "datasets": [training_dataset_registration(source, rows)],
                }
            }
        },
        environment_name: {
            "environment_servers": {
                "unified_video_harbor": {
                    "entrypoint": "app.py",
                    "agent_server": {"type": "responses_api_agents", "name": source},
                    "harness_agents": {
                        "opencode": {"type": "responses_api_agents", "name": source},
                        "codex": {"type": "responses_api_agents", "name": source + "_codex"},
                        "hermes": {"type": "responses_api_agents", "name": source + "_hermes"},
                    },
                    "sandbox_provider": provider_ref,
                    "tasksets": {
                        taskset: taskset_mapping(prepared.tasks, prepared.folder)
                        for taskset in (CLAW, *CLAW_HARNESS_TASKSETS.values())
                    },
                    "task_mcp_grants": "required",
                    "cleanup_timeout_seconds": 180,
                    "default_episode_timeout_seconds": 21600,
                    "harness_setup_allowance_s": 600,
                    "max_concurrent_episodes": 12,
                    "queue_timeout_seconds": 21600,
                    "artifacts_dir": str(rows.parent / "claw" / "verifier"),
                    "secrets": {
                        "CLAW_EVAL_JUDGE_API_KEY": "${oc.env:INFERENCE_API_KEY}",
                        "CLAW_EVAL_JUDGE_BASE_URL": "${oc.env:CLAW_EVAL_JUDGE_BASE_URL,https://inference-api.nvidia.com/v1}",
                        "CLAW_EVAL_JUDGE_MODEL": "${oc.env:CLAW_EVAL_JUDGE_MODEL,gcp/google/gemini-3-flash-preview}",
                    },
                }
            }
        },
    }
    common = {
        "training_model_controls": True,
        "token_id_capture": True,
        "entrypoint": "app.py",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "model": "${policy_model_name}",
        "resources_server": None,
        "concurrency": 12,
        "num_workers": 1,
        "datasets": [],
    }
    config[source + "_codex"] = {
        "responses_api_agents": {
            "codex_agent": common
            | {
                "timeout": 7200,
                "model_context_window": context_tokens,
                "codex_version": "0.144.4",
            }
        }
    }
    config[source + "_hermes"] = {
        "responses_api_agents": {
            "hermes_agent": common
            | {
                "sandbox_runner_timeout_seconds": 7200,
                "max_turns": max_steps,
                "max_tokens": output_tokens,
                "enabled_toolsets": ["terminal"],
                "compression_enabled": False,
                "context_length": context_tokens,
                "chat_template_kwargs_enabled": True,
            }
        }
    }
    return config


def _docker_binary(value: str) -> Path:
    selected = Path(value)
    if selected.name != "docker":
        raise ValueError("docker_binary must name a docker executable")
    binary = selected.parent.resolve() / selected.name
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"docker_binary must name an existing executable: {binary}")
    return binary


def write_launcher(output: Path, *, backend: Path, docker_binary: str) -> Path:
    """Start the prepared servers with Docker's standard process environment."""
    binary = _docker_binary(docker_binary)
    values = backend_environment(backend) | {
        "PATH": str(binary.parent) + os.pathsep + os.environ.get("PATH", ""),
        "DOCKER_CONFIG": str(binary.parent),
    }
    launcher = output / "start.sh"
    launcher.write_text(
        "#!/bin/sh\nset -eu\nunset DOCKER_CONTEXT\n"
        + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items())
        + "exec gym env start --config "
        + shlex.quote(str((output / "gym_config.yaml").resolve()))
        + ' "$@"\n'
    )
    launcher.chmod(0o700)
    return launcher


def _row_identity(row: dict) -> str:
    if row["task_source"] != CLAW:
        raise ValueError("Expected a ClawEval task")
    metadata = row["verifier_metadata"]
    task_id = metadata["task_id"]
    if (
        not isinstance(task_id, str)
        or not task_id
        or task_id.startswith(".")
        or any(character in task_id for character in ("/", "\\", "\0"))
    ):
        raise ValueError("Task ID must name one visible task directory")
    return task_id


def prepare(
    *,
    training_data: Path,
    output: Path,
    claw_roots: dict[str, str],
    claw_source: Path,
    claw_backend: Path | None = None,
    docker_binary: str | None = None,
    environment_image: str | None = None,
    context_tokens: int = 262144,
    output_tokens: int = 8192,
    max_steps: int = 32,
    stage_only: bool = False,
    reuse_staged: bool = False,
) -> Path:
    """Stage each ClawEval task once, preserving all original dataset sampling weights.

    Agentic-VBench uses its stock Harbor entry and DSS materialization recipe. This preparer does not
    interpret its rows, stage its tasks, build its images, or replace its judge.
    """
    output = output.resolve()
    rows = [json.loads(line) for line in training_data.read_text().splitlines() if line.strip()]
    if any(
        row.get("task_source") == "agentic_vbench_agent" and not isinstance(row.get("task_id"), dict) for row in rows
    ):
        raise ValueError("Agentic-VBench uses stock Harbor preparation and materialized task rows")
    selected = [row for row in rows if row.get("task_source") == CLAW]
    if not selected:
        raise ValueError("Training data has no ClawEval rows")
    identities = [_row_identity(row) for row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate staged ClawEval task")
    for row in selected:
        params = row["responses_create_params"]
        if not isinstance(params, dict) or params.keys() - {"input", *ModelCallSampling.model_fields}:
            raise ValueError("Claw rows may set only the prompt and model sampling fields")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=reuse_staged)
    input_sha256 = hashlib.sha256(training_data.read_bytes()).hexdigest()
    claw_media_source = Path(__file__).resolve().parents[2] / "responses_api_agents/claweval_agent/mcp_server.py"
    stage_contract = {
        "input_sha256": input_sha256,
        "environment_image": environment_image,
        "claw_media_output_bytes": CLAW_MEDIA_OUTPUT_BYTES,
        "claw_media_runtime_sha256": hashlib.sha256(claw_media_source.read_bytes()).hexdigest(),
        "claw_media_server": CLAW_MEDIA_SERVER,
        "claw_media_transport_sources": {
            name: hashlib.sha256(claw_media_source.with_name(name).read_bytes()).hexdigest()
            for name in ("mcp_http.py", "start_media_http.py")
        },
    }
    folder = output / CLAW
    if reuse_staged:
        previous = json.loads((output / "staging.json").read_text())
        if previous["contract"] != stage_contract:
            raise ValueError("Staged input or task preparation settings changed")
        for row in selected:
            task_folder = folder / row["verifier_metadata"]["task_id"]
            runtime = task_folder / "steps/solve/workdir/.claw_tools/mcp_server.py"
            document = tomllib.loads((task_folder / "task.toml").read_text())
            if hashlib.sha256(runtime.read_bytes()).hexdigest() != stage_contract[
                "claw_media_runtime_sha256"
            ] or document["environment"].get("env", {}).get("CLAW_MEDIA_MAX_OUTPUT_BYTES") != str(
                CLAW_MEDIA_OUTPUT_BYTES
            ):
                raise ValueError(f"Staged Claw media runtime or byte allowance changed: {task_folder.name}")
            for name, digest in stage_contract["claw_media_transport_sources"].items():
                if hashlib.sha256(runtime.with_name(name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Staged Claw media transport changed: {task_folder.name}")
            if document["environment"].get("mcp_servers", []) != [CLAW_MEDIA_SERVER]:
                raise ValueError(f"Staged Claw media launch changed: {task_folder.name}")
    else:
        tasks = claw_inventory(claw_roots)
        folder.mkdir(parents=True)
        (folder / "dataset.toml").write_text("[gym.defaults]\n")
        for row in selected:
            metadata = row["verifier_metadata"]
            key = f"{metadata['task_variant']}::{metadata['task_id']}"
            task = tasks.get(key)
            if task is None:
                raise ValueError(f"Unknown ClawEval task {key}")
            target = folder / metadata["task_id"]
            if target.exists():
                raise ValueError(f"Duplicate staged ClawEval task {metadata['task_id']}")
            stage_claw(row, task, source=claw_source, destination=target, environment_image=environment_image)
        (output / "staging.json").write_text(
            json.dumps(
                {
                    "contract": stage_contract,
                    "input": str(training_data.resolve()),
                    "rows": dict(Counter(row["task_source"] for row in rows)),
                },
                indent=2,
            )
            + "\n"
        )
    if stage_only:
        return output / "staging.json"
    if claw_backend is None or docker_binary is None:
        raise ValueError("Building staged tasks requires the Claw backend directory and --docker-binary")
    docker_binary = str(_docker_binary(docker_binary))
    # Build dependencies are unnecessary for the task-staging-only path.
    from nemo_gym.data.tasks.harbor import BuildOptions, materialize_task, prepare_target

    values = backend_environment(claw_backend) | {
        "PATH": str(Path(docker_binary).parent) + os.pathsep + os.environ.get("PATH", ""),
        "DOCKER_CONFIG": str(Path(docker_binary).parent),
    }
    with client_environment(values):
        prepared = prepare_target(
            str(folder),
            output_root=output / "materialized",
            sandbox_provider="docker",
            compose_platform="linux/arm64",
            build=BuildOptions(tool="docker", platform="linux/arm64"),
        )
    if prepared.skipped or {task.task_id for task in prepared.tasks} != set(identities):
        raise RuntimeError(f"Preparation skipped selected ClawEval tasks: {prepared.skipped}")
    bindings: dict[str, HarborTask] = {}
    for task in prepared.tasks:
        metadata = task.task_data["verifier_metadata"]
        if (
            task.task_data.get("video_source") != CLAW
            or _row_identity({"task_source": CLAW, "verifier_metadata": metadata}) != task.task_id
        ):
            raise ValueError("Prepared task identity differs from its original metadata")
        bindings[task.task_id] = task
    converted = []
    for row in rows:
        if row.get("task_source") != CLAW:
            converted.append(row)
            continue
        task = bindings[_row_identity(row)]
        params = row["responses_create_params"]
        materialized = materialize_task(task, CLAW).model_dump(mode="json", exclude_unset=True)
        task_input = materialized["task_input"]
        turns = task_input.get("turns", [])
        if len(turns) != 1 or not isinstance(turns[0].get("responses_create_params"), dict):
            raise ValueError("Claw tasks require exactly one native turn")
        turns[0]["responses_create_params"].update(
            {key: value for key, value in params.items() if key in ModelCallSampling.model_fields}
        )
        converted.append(row | materialized)
    row_path = output / "train.jsonl"
    row_path.write_text("".join(json.dumps(row) + "\n" for row in converted))
    config = {
        "cache_dir": str((output / "cache/gym").resolve()),
        "results_dir": str((output / "results").resolve()),
        "environment_routing_mode": "agent",
        "environment_server_routes": {
            taskset: "claweval_environment_server" for taskset in (CLAW, *CLAW_HARNESS_TASKSETS.values())
        },
    }
    config.update(
        component_config(
            CLAW,
            prepared=prepared,
            rows=row_path,
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            max_steps=max_steps,
        )
    )
    (output / "gym_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    launcher = write_launcher(output, backend=claw_backend, docker_binary=docker_binary)
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "input": str(training_data.resolve()),
                "input_sha256": input_sha256,
                "rows": dict(Counter(row["task_source"] for row in converted)),
                "source_rows": dict(Counter(row["task_source"] for row in rows)),
                "native_tasksets": [CLAW],
                "launcher": str(launcher),
                "output_sha256": hashlib.sha256(row_path.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return row_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("training-data", "output", "claw-source"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--claw-roots", type=json.loads, required=True, help="JSON variant-to-task-root mapping")
    parser.add_argument("--claw-backend", type=Path)
    parser.add_argument("--docker-binary")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--reuse-staged", action="store_true")
    parser.add_argument("--environment-image")
    parser.add_argument("--context-tokens", type=int, default=262144)
    parser.add_argument("--output-tokens", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=32)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
