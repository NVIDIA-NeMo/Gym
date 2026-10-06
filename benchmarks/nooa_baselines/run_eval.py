# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch one recorded NOOA benchmark phase with Gym's normal lifecycle."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from benchmarks.nooa_baselines.serving import validated_model
from nemo_gym.cli._venv_setup import SETUP_COMPLETE_MARKER
from nemo_gym.cli.setup_command import get_venv_path
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


BENCHMARKS = {
    "swe": (
        "benchmarks/swebench/pro/nooa_baseline.yaml",
        "swebench-pro-nooa-731.jsonl",
        "swebench_pro_nooa",
        42000,
        16,
    ),
    "tb": (
        "benchmarks/terminal_bench_2_1/nooa.yaml",
        "terminal-bench-2.1-nooa-89.jsonl",
        "terminal_bench_2_1_nooa",
        43000,
        8,
    ),
    "gdp": ("benchmarks/gdpval/nooa.yaml", "gdpval-nooa-220.jsonl", "gdpval_nooa", 44000, 8),
}


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Read physical LF-delimited objects without splitting Unicode inside strings."""
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"Incomplete JSONL record at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield row


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    """Create and fsync an immutable receipt before subsequent reporting can fail."""
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def outcome_metadata(row: dict[str, object]) -> dict[str, object]:
    """Do not retain multi-gigabyte trajectories just to reconcile task coverage."""
    keys = {
        "_ng_task_id",
        "_ng_task_index",
        "_ng_attempt_index",
        "_ng_rollout_index",
        "_ng_failure_class",
        "failure_kind",
        "evaluation_completed",
        "mask_sample",
        "execute_only",
        "generation_manifest",
    }
    return {key: value for key, value in row.items() if key in keys}


def snapshot_source(gym: Path, destination: Path) -> Path:
    """Record tracked source bytes from the selected checkout, including local edits.

    Install the intended commit first: ignored and untracked files are deliberately
    excluded, so personal runtime files and credentials never enter this receipt.
    """
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=gym).decode().split("\0")
    files = {
        name: hashlib.sha256((gym / name).read_bytes()).hexdigest()
        for name in sorted(paths)
        if name and (gym / name).is_file()
    }
    payload = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=gym, text=True).strip(),
        "tracked_files_sha256": files,
    }
    write_receipt(destination, payload)
    return destination


def pipeline_result_valid(row: dict, benchmark: str) -> bool:
    """A model failure can be valid; an infrastructure failure cannot pass the canary."""
    if row.get("_ng_failure_class") or row.get("failure_kind"):
        return False
    if benchmark == "gdp":
        manifest = row.get("generation_manifest")
        return bool(row.get("execute_only") and manifest and Path(manifest).is_file())
    return row.get("evaluation_completed") is True and not row.get("mask_sample", False)


def _task_identity(value: object) -> tuple[str, str]:
    if not isinstance(value, dict) or set(value) != {"taskset", "task_id"}:
        raise ValueError("Expected a native task identity")
    if not all(isinstance(value[key], str) and value[key] for key in ("taskset", "task_id")):
        raise ValueError("Task identity fields must be nonempty strings")
    return value["taskset"], value["task_id"]


def _identities(values: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"taskset": taskset, "task_id": task_id} for taskset, task_id in sorted(values)]


def reconcile_coverage(
    inputs: list[dict[str, object]],
    results: list[dict[str, object]],
    failures: list[dict[str, object]],
    benchmark: str,
) -> dict[str, object]:
    """Account for native identities without converting failed attempts into scores."""
    expected = [_task_identity(row.get("task_id")) for row in inputs]
    expected_set = set(expected)
    if not expected or len(expected_set) != len(expected):
        raise ValueError("Expected a nonempty input with unique native task identities")
    expected_by_index: dict[int, tuple[str, str]] = {}
    for position, (row, identity) in enumerate(zip(inputs, expected, strict=True)):
        index = row.get("_ng_task_index", position)
        if type(index) is not int or index < 0 or index in expected_by_index:
            raise ValueError("Input task indices must be unique nonnegative integers")
        expected_by_index[index] = identity
    result_ids: list[tuple[str, str]] = []
    valid_ids: set[tuple[str, str]] = set()
    failed_ids: set[tuple[str, str]] = set()
    attempts: Counter[tuple[tuple[str, str], int]] = Counter()
    failed_attempts, invalid_rows, indexed_failures = [], [], []
    for source, rows in (("results", results), ("failures", failures)):
        for line, row in enumerate(rows, 1):
            try:
                index = row.get("_ng_task_index")
                if "_ng_task_id" in row:
                    identity = _task_identity(row["_ng_task_id"])
                elif source == "failures" and type(index) is int and index in expected_by_index:
                    # Transport failures may precede the Environment Server identity response.
                    identity = expected_by_index[index]
                    indexed_failures.append(line)
                else:
                    raise ValueError("missing_native_identity")
                if index is not None and (
                    type(index) is not int or index not in expected_by_index or expected_by_index[index] != identity
                ):
                    raise ValueError("conflicting_input_index")
                attempt = row.get("_ng_attempt_index", 0)
                if type(attempt) is not int or attempt < 0:
                    raise ValueError("invalid_attempt_index")
                if source == "failures" and not row.get("_ng_failure_class"):
                    raise ValueError("missing_failure_class")
            except (ValueError, TypeError, AttributeError):
                invalid_rows.append({"source": source, "line": line})
                continue
            attempts[(identity, attempt)] += 1
            if source == "results":
                result_ids.append(identity)
                if pipeline_result_valid(row, benchmark):
                    valid_ids.add(identity)
            else:
                failed_ids.add(identity)
                failed_attempts.append(
                    {"task_id": _identities([identity])[0], "attempt_index": attempt, "sidecar_line": line}
                )
    observed = set(result_ids) | failed_ids
    missing, unexpected = expected_set - observed, observed - expected_set
    duplicates = [
        {"task_id": _identities([identity])[0], "attempt_index": attempt, "records": count}
        for (identity, attempt), count in sorted(attempts.items())
        if count > 1
    ]
    duplicate_results = {identity for identity, count in Counter(result_ids).items() if count > 1}
    return {
        "expected_tasks": len(expected),
        "completed_result_ids": _identities(valid_ids & expected_set),
        "valid_results": len(valid_ids & expected_set),
        "valid_result_kind": "generation" if benchmark == "gdp" else "grade",
        "failed_attempt_ids": failed_attempts,
        "failed_attempts": len(failed_attempts),
        "ungraded_result_ids": _identities((set(result_ids) - valid_ids) & expected_set),
        "ungraded_task_ids": _identities((observed - valid_ids) & expected_set),
        "ungraded_tasks": len((observed - valid_ids) & expected_set),
        "unique_attempted": len(observed & expected_set),
        "missing_ids": _identities(missing),
        "unexpected_ids": _identities(unexpected),
        "duplicates": duplicates,
        "duplicate_result_ids": _identities(duplicate_results),
        "invalid_rows": invalid_rows,
        "failure_identity_from_input_index_lines": indexed_failures,
        "coverage_complete": not (missing or unexpected or duplicates or duplicate_results or invalid_rows),
    }


def prepare_e2e_config(config: dict[str, object], *, input_path: Path) -> tuple[dict[str, object], Path]:
    """Stage exact native rows for Gym's existing prepared-data/e2e lifecycle.

    Gym derives this path from its output and split. Flatten recipe includes
    using Gym's own merge order so inherited direct-input settings can be
    omitted without changing the CLI or silently re-preparing a different set.
    """
    _, inherited = GlobalConfigDictParser().load_extra_config_paths(config["config_paths"])
    merged = OmegaConf.merge(*inherited, config)
    merged.pop("config_paths", None)
    merged.pop("input_jsonl_fpath", None)
    merged["split"] = "benchmark"
    merged["reuse_existing_data_preparation"] = True
    prepared = Path(merged["output_jsonl_fpath"]).with_suffix("") / "preprocessed_datasets/benchmark.jsonl"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    original = input_path.read_bytes()
    if prepared.exists():
        if prepared.read_bytes() != original:
            raise ValueError(f"Prepared benchmark differs from the immutable native input: {prepared}")
    else:
        with prepared.open("xb") as stream:
            stream.write(original)
    return OmegaConf.to_container(merged, resolve=False), prepared


def prepare_full_resume(
    config: dict[str, object], *, input_path: Path, phase_dir: Path, benchmark: str
) -> dict[str, object]:
    """Expand the canary schedule without changing completed rows or their indices.

    Gym resume intentionally trusts its materialized schedule. Increasing limit
    alone cannot expand it. Use Gym's preprocessing, verify the old prefix, and
    preserve its exact bytes before installing the complete native schedule.
    """
    settings = {key: value for key, value in config.items() if key in RolloutCollectionConfig.model_fields}
    settings.update(input_jsonl_fpath=str(input_path), limit=None)
    collection = RolloutCollectionConfig.model_validate(settings)
    cache = collection.materialized_jsonl_fpath
    if not cache.is_file() or not Path(collection.output_jsonl_fpath).is_file():
        raise ValueError("Full resume requires the preserved canary output and materialized input schedule")
    original = cache.read_bytes()
    previous = list(read_jsonl(cache))
    complete = RolloutCollectionHelper()._preprocess_rows_from_config(collection)
    if not complete or any("task_input" not in row for row in complete):
        raise ValueError("Baseline full resume requires nonempty native task inputs")
    if not previous or previous != complete[: len(previous)]:
        raise ValueError("Canary materialized inputs are not an unchanged prefix of the full native schedule")
    expected = {(row["_ng_task_index"], row["_ng_rollout_index"]): _task_identity(row["task_id"]) for row in complete}
    if len(expected) != len(complete):
        raise ValueError("Full native schedule contains duplicate rollout indices")
    canary_valid = False
    first_key = (complete[0]["_ng_task_index"], complete[0]["_ng_rollout_index"])
    for artifact in (Path(collection.output_jsonl_fpath), failures_path_for(Path(collection.output_jsonl_fpath))):
        if not artifact.exists():
            continue
        for row in read_jsonl(artifact):
            key = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
            if key not in expected or ("_ng_task_id" in row and _task_identity(row["_ng_task_id"]) != expected[key]):
                raise ValueError("Existing rollout identity conflicts with the complete native schedule")
            if artifact == Path(collection.output_jsonl_fpath) and key == first_key:
                canary_valid = "_ng_task_id" in row and pipeline_result_valid(row, benchmark)
    if not canary_valid:
        raise ValueError("Full resume requires the valid native canary result at its original index")
    expanded = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in complete)
    receipt = {
        "path": str(cache),
        "previous_rows": len(previous),
        "full_rows": len(complete),
        "previous_sha256": hashlib.sha256(original).hexdigest(),
        "full_sha256": hashlib.sha256(expanded).hexdigest(),
    }
    if previous != complete:
        preserved = phase_dir / "materialized-inputs-before-full.jsonl"
        with preserved.open("xb") as stream:
            stream.write(original)
        temporary = phase_dir / "materialized-inputs-full.jsonl.tmp"
        with temporary.open("xb") as stream:
            stream.write(expanded)
        temporary.replace(cache)
        receipt["preserved_previous_path"] = str(preserved)
    else:
        receipt["full_sha256"] = receipt["previous_sha256"]
    return receipt


def server_venv_readiness(config: dict[str, object]) -> dict[str, bool]:
    """Reuse only complete environments for every server in this benchmark."""
    parser = GlobalConfigDictParser()
    composed = OmegaConf.create(config)
    parser._recursively_swap_keys(composed)
    # Inspect only server blocks; child-process environment secrets are not yet
    # installed in this parent process and must not be resolved here.
    server_blocks = OmegaConf.create(
        {key: value for key, value in composed.items_ex(resolve=False) if isinstance(value, DictConfig)}
    )
    paths = {
        get_venv_path(Path(server.SERVER_TYPE, next(iter(getattr(server, server.SERVER_TYPE)))), composed)
        for server in parser.filter_for_server_instance_configs(server_blocks)
    }
    if not paths:
        raise ValueError("No server environments were found in the benchmark configuration")
    return {
        str(path): all((path / name).is_file() for name in (SETUP_COMPLETE_MARKER, "bin/python", "bin/activate"))
        for path in sorted(paths)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--gym-source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gym-executable", default=shutil.which("gym") or "gym")
    parser.add_argument(
        "--input", type=Path, help="Prepared native input; defaults to RUN_ROOT/data/<benchmark>.jsonl"
    )
    parser.add_argument(
        "--source-manifest", type=Path, help="Preserved source-bundle manifest; otherwise snapshot tracked files"
    )
    parser.add_argument("--config", type=Path, action="append", default=[], help="Caller Hydra overlay, repeatable")
    parser.add_argument(
        "--prepare-apptainer", action="store_true", help="GDP only: prepare private Apptainer session config"
    )
    parser.add_argument("--port", type=int, help="Head port, with the next99 ports reserved for this benchmark")
    parser.add_argument(
        "--prefetch", action="store_true", help="Install/revalidate server environments serially before launch"
    )
    parser.add_argument("--model-dir", type=Path, help="Owned policy replica directory; defaults to RUN_ROOT/model")
    parser.add_argument("--phase", choices=["canary", "full"], required=True)
    parser.add_argument("--concurrency", type=int, help="Override positive full-phase episode concurrency")
    parser.add_argument(
        "--canary-source-manifest",
        type=Path,
        help="Explicit preserved source manifest for a reviewed launcher-only resume across source snapshots",
    )
    args = parser.parse_args()
    if args.concurrency is not None and (args.concurrency < 1 or args.phase != "full"):
        parser.error("--concurrency must be positive and used only for the full phase")
    if args.canary_source_manifest is not None and args.phase != "full":
        parser.error("--canary-source-manifest applies only to a full resume")
    root = args.run_root.resolve()
    recipe, input_name, environment, port, concurrency = BENCHMARKS[args.benchmark]
    concurrency = args.concurrency or concurrency
    gym = args.gym_source.resolve(strict=True)
    input_path = (args.input or root / "data" / input_name).resolve(strict=True)
    port = args.port or port
    output = root / args.benchmark
    output.mkdir(parents=True, exist_ok=True)
    phase_dir = output / args.phase
    phase_dir.mkdir(exist_ok=True)
    launch_path = phase_dir / "launch.json"
    if launch_path.exists():
        raise FileExistsError("This phase already has a launch receipt; inspect it before any retry")
    model_dir = args.model_dir.resolve() if args.model_dir else root / "model"
    model = validated_model(model_dir)
    source_path = args.source_manifest or root / "source-manifest.json"
    if not source_path.is_file():
        if args.source_manifest:
            raise FileNotFoundError(source_path)
        source_path = snapshot_source(gym, phase_dir / "source-manifest.json")
    protocol = {
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256((model_dir / "model.json").read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "overlays_sha256": hashlib.sha256(
            b"".join(hashlib.sha256(path.read_bytes()).digest() for path in args.config)
        ).hexdigest(),
    }
    source_bridge = None
    if args.phase == "full":
        canary = json.loads((output / "canary/completion.json").read_text())
        prior = canary.get("protocol", {})
        if not canary["pipeline_passed"] or any(
            prior.get(key) != protocol[key] for key in ("input_sha256", "model_manifest_sha256", "overlays_sha256")
        ):
            raise RuntimeError(
                "Full dispatch requires a valid canary with the same input, model and configuration overlays"
            )
        if args.canary_source_manifest is not None:
            preserved_source = args.canary_source_manifest.resolve(strict=True)
            digest = hashlib.sha256(preserved_source.read_bytes()).hexdigest()
            if digest != prior.get("source_manifest_sha256"):
                raise RuntimeError("Preserved source manifest does not match the canary receipt")
            source_bridge = {
                "preserved_canary_manifest": str(preserved_source),
                "canary_source_manifest_sha256": digest,
                "full_source_manifest_sha256": protocol["source_manifest_sha256"],
            }
        elif prior.get("source_manifest_sha256") != protocol["source_manifest_sha256"]:
            raise RuntimeError("Source changed: explicitly supply the reviewed preserved --canary-source-manifest")
    env = os.environ.copy()
    env.update(
        POLICY_BASE_URL=model["base_url"],
        POLICY_API_KEY=os.environ[model["api_key_env"]],
        POLICY_MODEL_NAME=model["served_model"],
        NOOA_SWE_BASELINE_RUN_DIR=str(output),
        NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="1",
        NEMO_GYM_RUN_ID=env.get("NEMO_GYM_RUN_ID", root.name + "-" + args.benchmark),
        UV_CACHE_DIR=str(root / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / "uv-python"),
        UV_HTTP_TIMEOUT="300",
        UV_LOCK_TIMEOUT="1800",
        # Cache and server environments both live in this private run root.
        UV_LINK_MODE="hardlink",
        RAY_TMPDIR="/tmp",
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    for key in ("PYTHONHOME", "PYTHONPATH", "UV_CONSTRAINT", "PIP_CONSTRAINT", "UV_VENV_CLEAR"):
        env.pop(key, None)
    if args.benchmark == "gdp":
        env["PERSIST_DELIVERABLES_DIR"] = str(output / "deliverables")
    if args.prepare_apptainer:
        if args.benchmark != "gdp":
            parser.error("--prepare-apptainer is only applicable to GDP")
        from benchmarks.gdpval.prepare_apptainer_runtime import prepare_apptainer_config

        prepared_runtime = prepare_apptainer_config(run_root=root)
        write_receipt(phase_dir / "apptainer-config.json", prepared_runtime)
        if prepared_runtime["override_required"]:
            env["APPTAINER_CONFIG_FILE"] = str(prepared_runtime["config_path"])
    config = {
        "config_paths": [recipe],
        "policy_base_url": "${oc.env:POLICY_BASE_URL}",
        "policy_api_key": "${oc.env:POLICY_API_KEY}",
        "policy_model_name": "${oc.env:POLICY_MODEL_NAME}",
        "head_server": {"port": port},
        "port_range_low": port + 1,
        "port_range_high": port + 99,
        "use_absolute_ip": True,
        "server_spinup_timeout_seconds": 3600,
        "uv_venv_dir": str(root / "server-venvs" / args.benchmark),
        "uv_cache_dir": str(root / "uv-cache"),
        "nemo_gym_log_dir": str(phase_dir / "server-logs"),
        "observability_enabled": True,
        "model_call_capture_dir": str(output / "model-calls"),
        "environment_server_routes": {
            (
                "swebench-pro-nooa"
                if args.benchmark == "swe"
                else "terminal-bench-2.1-nooa"
                if args.benchmark == "tb"
                else "gdpval-nooa"
            ): environment
        },
        "agent_name": environment + "_agent",
        "output_jsonl_fpath": str(output / "rollouts.jsonl"),
        "num_repeats": 1,
        "resume_from_cache": True,
        "route_failures_to_sidecar": True,
        "count_failure_classes_as_zero": [],
        "num_samples_in_parallel": 1 if args.phase == "canary" else concurrency,
        "max_resident_rollout_tasks": 1 if args.phase == "canary" else concurrency,
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {"sampling_overrides": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 32768}}
            }
        },
    }
    if args.benchmark != "swe":
        config["config_paths"].append("responses_api_models/vllm_model/configs/vllm_model.yaml")
    if args.benchmark == "tb":
        config["config_paths"].append("nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
        config["terminal_bench_2_1_nooa_resources_server"] = {
            "resources_servers": {
                "terminal_bench_2_1": {
                    "sandbox_config": {"provider_options": {"platform": {"os": "linux", "arch": "amd64"}}}
                }
            }
        }
    if args.phase == "canary":
        config["limit"] = 1
    else:
        config["limit"] = None
    if args.config:
        overlays = [OmegaConf.load(path) for path in args.config]
        config = OmegaConf.to_container(OmegaConf.merge(config, *overlays), resolve=False)
    # These phase controls preserve the selected dataset and canary/full identity contract.
    config.update(
        limit=1 if args.phase == "canary" else None,
        num_samples_in_parallel=1 if args.phase == "canary" else concurrency,
        max_resident_rollout_tasks=1 if args.phase == "canary" else concurrency,
    )
    config, prepared_path = prepare_e2e_config(config, input_path=input_path)
    resume_schedule = (
        prepare_full_resume(config, input_path=prepared_path, phase_dir=phase_dir, benchmark=args.benchmark)
        if args.phase == "full"
        else None
    )
    venv_readiness = server_venv_readiness(config)
    # A timed-out installer leaves an interpreter and activate script behind.
    # Force Gym's locked, --allow-existing setup unless every marker is present.
    config["skip_venv_if_present"] = all(venv_readiness.values())
    config_path = phase_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [args.gym_executable, "eval", "run", "--config", str(config_path)]
    launch = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_job_id": model.get("job_id"),
        "source_manifest_path": str(source_path),
        "config_overlays_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.config
        },
        "benchmark": args.benchmark,
        "phase": args.phase,
        "command": command,
        "concurrency": config["num_samples_in_parallel"],
        "protocol": protocol,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "prepared_input_path": str(prepared_path),
        "prepared_input_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "source_recipe": recipe,
        "server_venvs_before_launch": venv_readiness,
        "uv_link_mode": env["UV_LINK_MODE"],
        "resume_schedule": resume_schedule,
        "canary_source_bridge": source_bridge,
    }
    write_receipt(launch_path, launch)
    if args.prefetch:
        with (phase_dir / "prefetch.log").open("x") as log:
            subprocess.run(
                [args.gym_executable, "env", "prefetch", "--config", str(config_path), "++skip_venv_if_present=false"],
                cwd=gym,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    with (phase_dir / "controller.log").open("w") as log:
        result = subprocess.run(command, cwd=gym, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    native_exit = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": result.returncode,
        "protocol": protocol,
        "launch_sha256": hashlib.sha256(launch_path.read_bytes()).hexdigest(),
    }
    # This must precede even opening/parsing results. A reporting error must not
    # erase the actual child's exit status or be mistaken for native success.
    write_receipt(phase_dir / "native-exit.json", native_exit)
    rows_path = output / "rollouts.jsonl"
    rows = [outcome_metadata(row) for row in read_jsonl(rows_path)] if rows_path.exists() else []
    # Canary success means a completed pipeline, never necessarily reward one.
    passed = result.returncode == 0 and bool(rows)
    coverage = None
    if args.phase == "canary":
        passed = passed and len(rows) == 1 and pipeline_result_valid(rows[0], args.benchmark)
    else:
        failures_path = failures_path_for(rows_path)
        failures = [outcome_metadata(row) for row in read_jsonl(failures_path)] if failures_path.exists() else []
        inputs = [
            {key: row[key] for key in ("task_id", "_ng_task_index") if key in row} for row in read_jsonl(input_path)
        ]
        coverage = reconcile_coverage(inputs, rows, failures, args.benchmark)
        coverage["failure_sidecar_path"] = str(failures_path)
        passed = result.returncode == 0 and coverage["coverage_complete"]
    completion = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exit_code": result.returncode,
        "rows": len(rows),
        "pipeline_passed": passed,
        "protocol": protocol,
        **({"coverage": coverage} if coverage is not None else {}),
    }
    write_receipt(phase_dir / "completion.json", completion)
    print(json.dumps(completion))
    raise SystemExit(result.returncode if result.returncode else (0 if passed else 1))


if __name__ == "__main__":
    main()
