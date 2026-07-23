#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the default OSWorld benchmark workflow.

The committed five-task JSONL is immediately runnable. This entrypoint checks
that data and creates a benchmark-local ``env.yaml`` so the next two commands
can be invoked without Hydra arguments::

    gym env start
    gym eval run --no-serve

Referenced task inputs and evaluator files are prefetched through the official
Hugging Face cache, then materialized in OSWorld's per-task cache layout. The
VM preparation and supervisor wrappers live under ``benchmarks/osworld/tools``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmarks.osworld.assets import DEFAULT_SETUP_CACHE, ensure_osworld_assets


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_CONFIG = BENCHMARK_DIR / "config.yaml"
DEFAULT_INPUT = BENCHMARK_DIR / "data" / "example.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "osworld" / "rollouts.jsonl"
DEFAULT_ENV = BENCHMARK_DIR / "env.yaml"

BASE_AGENT_CONFIG = REPO_ROOT / "responses_api_agents" / "osworld_agent" / "configs" / "osworld_agent.yaml"
OPENAI_MODEL_CONFIG = REPO_ROOT / "responses_api_models" / "openai_model" / "configs" / "openai_model.yaml"
POINTER_AGENT_CONFIG = BENCHMARK_DIR / "configs" / "osworld_agent_pointer.yaml"
NANO_OMNI_AGENT_CONFIG = BENCHMARK_DIR / "configs" / "osworld_agent_omni_mini.yaml"
NANO_OMNI_MODEL_CONFIG = BENCHMARK_DIR / "configs" / "vllm_model_omni_mini.yaml"
GYM_SANDBOX_CONFIG = BENCHMARK_DIR / "configs" / "osworld_sandbox.yaml"
OSWORLD_PROVIDER_CONFIG = BENCHMARK_DIR / "configs" / "osworld_docker_pinned.yaml"

PROFILE_CONFIGS: dict[str, tuple[Path, ...]] = {
    "default": (DEFAULT_CONFIG,),
    "pointer": (BASE_AGENT_CONFIG, POINTER_AGENT_CONFIG, OPENAI_MODEL_CONFIG),
    "nano_omni": (BASE_AGENT_CONFIG, NANO_OMNI_AGENT_CONFIG, NANO_OMNI_MODEL_CONFIG),
}
BACKEND_CONFIGS = {
    "gym_sandbox": GYM_SANDBOX_CONFIG,
    "osworld_provider": OSWORLD_PROVIDER_CONFIG,
}

PINNED_OSWORLD_IMAGE = (
    "docker://happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9"
)


def prepare(input_jsonl: Path = DEFAULT_INPUT) -> Path:
    """Validate and return an OSWorld JSONL suitable for rollout collection."""

    input_jsonl = input_jsonl.resolve()
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"OSWorld input JSONL does not exist: {input_jsonl}")

    row_count = 0
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_jsonl}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"OSWorld row {line_number} must be a JSON object")
            metadata = row.get("verifier_metadata")
            if not isinstance(metadata, dict) or not isinstance(metadata.get("osworld_task"), dict):
                raise ValueError(f"OSWorld row {line_number} must contain verifier_metadata.osworld_task")
            row_count += 1

    if row_count == 0:
        raise ValueError(f"OSWorld input JSONL is empty: {input_jsonl}")
    print(f"Validated {row_count} OSWorld task(s): {input_jsonl}")
    return input_jsonl


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def select_config_paths(
    *,
    profile: str,
    execution_backend: str,
    explicit_configs: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    """Resolve the complete Gym composition prepared for one OSWorld run."""

    try:
        profile_configs = PROFILE_CONFIGS[profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported OSWorld profile: {profile!r}") from exc
    try:
        backend_config = BACKEND_CONFIGS[execution_backend]
    except KeyError as exc:
        raise ValueError(f"Unsupported OSWorld execution backend: {execution_backend!r}") from exc

    selected = tuple(explicit_configs) if explicit_configs else profile_configs
    config_paths = tuple(Path(path).resolve() for path in (*selected, backend_config))
    missing = [path for path in config_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OSWorld Gym config does not exist: {missing[0]}")
    return tuple(dict.fromkeys(config_paths))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_task_shard(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    num_shards: int,
    shard_index: int,
) -> Path:
    """Write one deterministic, disjoint round-robin shard and its manifest.

    Sharding follows the zero-based index of each non-empty input row.  This
    preserves source order inside every shard while distributing adjacent
    tasks across workers.  The emitted manifest makes it possible to verify
    that independently prepared workers used the same source input and shard
    definition.
    """

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")

    input_jsonl = input_jsonl.expanduser().resolve()
    output_jsonl = output_jsonl.expanduser().resolve()
    if input_jsonl == output_jsonl:
        raise ValueError("shard output must differ from the source input")

    selected_rows: list[str] = []
    selected_task_ids: list[str] = []
    total_tasks = 0
    with input_jsonl.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if total_tasks % num_shards == shard_index:
                selected_rows.append(raw_line.rstrip("\r\n") + "\n")
                metadata = row.get("verifier_metadata") or {}
                task_id = metadata.get("task_id") or (metadata.get("osworld_task") or {}).get("id")
                selected_task_ids.append(str(task_id or f"row-{total_tasks}"))
            total_tasks += 1

    if not selected_rows:
        raise ValueError(f"shard {shard_index}/{num_shards} is empty for an input containing {total_tasks} task(s)")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(selected_rows), encoding="utf-8")
    task_ids_digest = hashlib.sha256("\n".join(selected_task_ids).encode()).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "osworld-task-shard",
        "source_path": str(input_jsonl),
        "source_sha256": _sha256_file(input_jsonl),
        "selection": "nonempty-row-index-modulo",
        "num_shards": num_shards,
        "shard_index": shard_index,
        "total_tasks": total_tasks,
        "shard_tasks": len(selected_rows),
        "task_ids_sha256": task_ids_digest,
        "task_ids": selected_task_ids,
    }
    manifest_path = output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote OSWorld shard {shard_index + 1}/{num_shards} "
        f"with {len(selected_rows)}/{total_tasks} task(s): {output_jsonl}"
    )
    return output_jsonl


def write_vm_snapshot_manifest(
    manifest_path: Path,
    *,
    vm_path: Path,
    execution_backend: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Pin the immutable qcow2 base used by both local provider paths.

    This is a disk/base-image snapshot identity. Gym Sandbox deliberately does
    not claim live RAM/device-state snapshots: reset closes the Sandbox and
    creates a fresh VM from this read-only base.
    """

    resolved_vm = vm_path.expanduser().resolve()
    if not resolved_vm.is_file() or not os.access(resolved_vm, os.R_OK):
        raise FileNotFoundError(f"OSWorld qcow2 is not readable: {resolved_vm}")
    actual_sha256 = _sha256_file(resolved_vm)
    normalized_expected = expected_sha256.removeprefix("sha256:").lower() if expected_sha256 else None
    if normalized_expected and actual_sha256 != normalized_expected:
        raise ValueError(f"OSWorld qcow2 SHA-256 mismatch: expected {normalized_expected}, got {actual_sha256}")

    manifest = {
        "schema_version": 1,
        "kind": "osworld-qcow2-base-snapshot",
        "snapshot_id": f"sha256:{actual_sha256}",
        "path": str(resolved_vm),
        "size_bytes": resolved_vm.stat().st_size,
        "sha256": actual_sha256,
        "execution_backend": execution_backend,
        "container_image": PINNED_OSWORLD_IMAGE,
        "mount_mode": "read-only" if execution_backend == "gym_sandbox" else "provider-managed",
        "reset_semantics": "close-and-recreate-from-base",
        "live_ram_snapshot_supported": False,
    }
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Pinned VM snapshot {manifest['snapshot_id']}: {manifest_path}")
    return manifest


def write_env(
    env_path: Path,
    *,
    config_path: Path | None = None,
    config_paths: Sequence[Path] | None = None,
    input_jsonl: Path,
    output_jsonl: Path,
    policy_base_url: str,
    policy_api_key: str,
    policy_model_name: str,
    setup_cache_dir: Path = DEFAULT_SETUP_CACHE,
    asset_input_jsonl: Path | None = None,
    num_samples_in_parallel: int = 1,
    max_output_tokens: int = 1500,
    temperature: float = 1.0,
    top_p: float | None = None,
    execution_backend: str = "osworld_provider",
    vm_path: Path | None = None,
    head_host: str = "127.0.0.1",
    head_port: int = 11000,
    server_venv_root: Path | None = None,
    max_steps: int | None = None,
    force: bool = False,
) -> bool:
    """Create a private env.yaml; return False when an existing file is kept."""

    env_path = env_path.resolve()
    if env_path.exists() and not force:
        print(f"Keeping existing configuration: {env_path}")
        return False

    env_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_jsonl.resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    selected_configs = tuple(config_paths or (() if config_path is None else (config_path,)))
    if not selected_configs:
        raise ValueError("write_env requires config_path or config_paths")
    if not head_host.strip():
        raise ValueError("head_host must not be empty")
    if not 1 <= head_port <= 65535:
        raise ValueError("head_port must be between 1 and 65535")
    if num_samples_in_parallel < 1:
        raise ValueError("num_samples_in_parallel must be >= 1")
    if execution_backend not in BACKEND_CONFIGS:
        raise ValueError(f"Unsupported OSWorld execution backend: {execution_backend!r}")
    resolved_vm_path = vm_path.expanduser().resolve() if vm_path else None
    if execution_backend == "gym_sandbox" and resolved_vm_path is None:
        raise ValueError("gym_sandbox execution requires an explicit vm_path")
    contents = "\n".join(
        [
            "# Generated by benchmarks/osworld/prepare.py. This file is gitignored.",
            "config_paths:",
            *(f"  - {_yaml_string(path.resolve())}" for path in selected_configs),
            "head_server:",
            f"  host: {_yaml_string(head_host)}",
            f"  port: {head_port}",
            *([] if server_venv_root is None else [f"uv_venv_dir: {_yaml_string(server_venv_root.resolve())}"]),
            "agent_name: osworld_simple_agent",
            f"input_jsonl_fpath: {_yaml_string(input_jsonl.resolve())}",
            f"output_jsonl_fpath: {_yaml_string(output_jsonl)}",
            "num_repeats: 1",
            f"num_samples_in_parallel: {num_samples_in_parallel}",
            "upload_rollouts_to_wandb: false",
            "responses_create_params:",
            f"  max_output_tokens: {max_output_tokens}",
            f"  temperature: {temperature}",
            *([] if top_p is None else [f"  top_p: {top_p}"]),
            f"policy_base_url: {_yaml_string(policy_base_url)}",
            f"policy_api_key: {_yaml_string(policy_api_key)}  # pragma: allowlist secret",
            f"policy_model_name: {_yaml_string(policy_model_name)}",
            "osworld_simple_agent:",
            "  responses_api_agents:",
            "    osworld_agent:",
            f"      concurrency: {num_samples_in_parallel}",
            f"      setup_cache_dir: {_yaml_string(setup_cache_dir.resolve())}",
            f"      asset_input_jsonl: {_yaml_string((asset_input_jsonl or input_jsonl).resolve())}",
            *([] if resolved_vm_path is None else [f"      vm_path: {_yaml_string(resolved_vm_path)}"]),
            *([] if max_steps is None else [f"      max_steps: {max_steps}"]),
            "",
        ]
    )

    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if force else os.O_EXCL)
    descriptor = os.open(env_path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(contents)
    os.chmod(env_path, 0o600)
    print(f"Wrote private configuration: {env_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="OSWorld input JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Rollout output JSONL")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split non-empty input rows into this many deterministic round-robin shards",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard selected by this worker (requires --num-shards)",
    )
    parser.add_argument(
        "--shard-output",
        type=Path,
        default=None,
        help="Generated shard JSONL (default: input-shard-NN-of-NN.jsonl beside --output)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Explicit Gym config; repeat to compose configs instead of the selected profile",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONFIGS),
        default="default",
        help="Model/agent composition written to env.yaml",
    )
    parser.add_argument(
        "--execution-backend",
        choices=tuple(BACKEND_CONFIGS),
        default="osworld_provider",
        help="VM lifecycle owner; both choices still execute through Gym env",
    )
    parser.add_argument(
        "--vm-path",
        type=Path,
        default=None,
        help="Explicit qcow2 base; required for gym_sandbox and recommended for reproducible native runs",
    )
    parser.add_argument(
        "--expected-vm-sha256",
        default=None,
        help="Fail preparation unless the qcow2 base has this SHA-256",
    )
    parser.add_argument(
        "--snapshot-manifest",
        type=Path,
        default=None,
        help="VM snapshot identity JSON (default: vm-snapshot.json beside env.yaml)",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="Generated local env.yaml")
    parser.add_argument("--no-env", action="store_true", help="Validate data without creating env.yaml")
    parser.add_argument("--skip-assets", action="store_true", help="Validate data without prefetching task assets")
    parser.add_argument("--force-env", action="store_true", help="Replace an existing generated env.yaml")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent eval requests and OSWorld DesktopEnv instances written to env.yaml",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1500, help="Per-step model output limit")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional rollout step cap override")
    parser.add_argument("--temperature", type=float, default=1.0, help="Model sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Optional nucleus-sampling top-p")
    parser.add_argument("--head-host", default="127.0.0.1", help="Gym head-server bind/connect host")
    parser.add_argument("--head-port", type=int, default=11000, help="Gym head-server port")
    parser.add_argument(
        "--server-venv-root",
        type=Path,
        default=None,
        help="Optional persistent directory for Gym-managed server environments",
    )
    parser.add_argument(
        "--setup-cache-dir",
        type=Path,
        default=DEFAULT_SETUP_CACHE,
        help="Shared cache populated for OSWorld setup and evaluator files",
    )
    parser.add_argument(
        "--asset-proxy-url",
        default=os.environ.get("OSWORLD_ASSET_PROXY_URL"),
        help="Optional HTTP proxy used only when the normal HF download fails",
    )
    parser.add_argument(
        "--policy-base-url",
        default=os.environ.get("POLICY_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--policy-api-key",
        default=os.environ.get("POLICY_API_KEY", "local-vllm"),  # pragma: allowlist secret
    )
    parser.add_argument(
        "--policy-model-name",
        default=os.environ.get("POLICY_MODEL_NAME", "osworld-policy"),
    )
    args = parser.parse_args()

    input_jsonl = prepare(args.input)
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error(f"--shard-index must be in [0, {args.num_shards})")
    if args.num_shards > 1 or args.shard_output is not None:
        shard_output = args.shard_output or (
            args.output.expanduser().resolve().parent
            / f"input-shard-{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
        )
        input_jsonl = write_task_shard(
            input_jsonl,
            shard_output,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    if not args.skip_assets:
        summary = ensure_osworld_assets(
            input_jsonl,
            args.setup_cache_dir,
            token=os.environ.get("HF_TOKEN"),
            proxy_url=args.asset_proxy_url,
        )
        print(
            f"Prepared {summary.asset_count} remote asset(s) for {summary.task_count} task(s) "
            f"in {summary.cache_dir} ({summary.materialized_count} new cache entry/entries)"
        )
    config_paths = select_config_paths(
        profile=args.profile,
        execution_backend=args.execution_backend,
        explicit_configs=args.config,
    )
    vm_path = args.vm_path.expanduser().resolve() if args.vm_path else None
    if args.execution_backend == "gym_sandbox" and vm_path is None:
        parser.error("--execution-backend gym_sandbox requires --vm-path")
    if args.expected_vm_sha256 and vm_path is None:
        parser.error("--expected-vm-sha256 requires --vm-path")
    if vm_path is not None:
        manifest_path = args.snapshot_manifest or args.env_file.expanduser().resolve().with_name("vm-snapshot.json")
        write_vm_snapshot_manifest(
            manifest_path,
            vm_path=vm_path,
            execution_backend=args.execution_backend,
            expected_sha256=args.expected_vm_sha256,
        )
    if not args.no_env:
        write_env(
            args.env_file,
            config_paths=config_paths,
            input_jsonl=input_jsonl,
            output_jsonl=args.output,
            policy_base_url=args.policy_base_url,
            policy_api_key=args.policy_api_key,
            policy_model_name=args.policy_model_name,
            setup_cache_dir=args.setup_cache_dir,
            asset_input_jsonl=input_jsonl,
            num_samples_in_parallel=args.concurrency,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            execution_backend=args.execution_backend,
            vm_path=vm_path,
            head_host=args.head_host,
            head_port=args.head_port,
            server_venv_root=args.server_venv_root,
            max_steps=args.max_steps,
            force=args.force_env,
        )

    print("\nNext steps:")
    print(f"  cd {args.env_file.expanduser().resolve().parent}")
    print("  gym env start")
    print("  gym eval run --no-serve")


if __name__ == "__main__":
    main()
