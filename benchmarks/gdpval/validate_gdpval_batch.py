#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and stage a GDPVal-format batch for rollout collection.

The source JSONL stays immutable. Optional reference overrides are applied only
to a generated launch copy, and only when every added URL already appears in the
task prompt. This keeps small source-data repairs explicit and auditable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


REQUIRED_KEYS = {
    "responses_create_params",
    "task_id",
    "sector",
    "occupation",
    "prompt",
    "reference_files",
    "reference_file_urls",
    "rubric_json",
    "rubric_pretty",
}
# Permissive by default: a batch-specific pattern belongs in that batch's
# dataset profile, not baked in here, or ids from other GDPVal sources are
# rejected out of the box.
DEFAULT_TASK_ID_PATTERN = r"\S+$"
# Reference locators: some batches ship HTTPS URLs, others ship absolute
# paths to files already staged on a shared filesystem. "any" accepts either.
REFERENCE_MODES = ("https", "local", "any")
PROMPT_URL_RE = re.compile(r"https?://[^\s<>]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Launch inputs contain tokenized reference URLs; keep generated copies
        # private on shared cluster filesystems.
        temporary_path.chmod(0o600)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_deliverable_keys(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).startswith("deliverable_"):
                found.append(path)
            found.extend(_nested_deliverable_keys(nested, path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(_nested_deliverable_keys(nested, f"{prefix}[{index}]"))
    return found


def validate_input_rows(
    rows: list[dict[str, Any]],
    *,
    expected_count: int | None,
    task_id_pattern: str = DEFAULT_TASK_ID_PATTERN,
    reference_mode: str = "https",
) -> tuple[list[str], list[str]]:
    task_id_re = re.compile(task_id_pattern)
    errors: list[str] = []
    warnings: list[str] = []
    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"expected {expected_count} input rows, found {len(rows)}")

    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        task_id = row.get("task_id")
        label = str(task_id or f"row {index}")
        missing = sorted(REQUIRED_KEYS.difference(row))
        if missing:
            errors.append(f"{label}: missing keys: {', '.join(missing)}")
        if not isinstance(task_id, str) or not task_id_re.fullmatch(task_id):
            errors.append(f"row {index}: invalid task_id {task_id!r}")
        elif task_id in seen_ids:
            errors.append(f"{task_id}: duplicate task_id")
        else:
            seen_ids.add(task_id)

        leaked_keys = _nested_deliverable_keys(row)
        if leaked_keys:
            errors.append(f"{label}: gold-output keys are policy-visible: {', '.join(leaked_keys)}")

        responses_create_params = row.get("responses_create_params")
        if responses_create_params != {"input": []}:
            errors.append(f"{label}: responses_create_params must be exactly {{'input': []}}")

        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{label}: prompt must be a non-empty string")
            prompt = ""

        reference_files = row.get("reference_files")
        reference_urls = row.get("reference_file_urls")
        if not isinstance(reference_files, list) or not all(isinstance(item, str) for item in reference_files):
            errors.append(f"{label}: reference_files must be a list of strings")
            reference_files = []
        if not isinstance(reference_urls, list) or not all(isinstance(item, str) for item in reference_urls):
            errors.append(f"{label}: reference_file_urls must be a list of strings")
            reference_urls = []
        if len(reference_files) != len(reference_urls):
            errors.append(
                f"{label}: reference file/url count mismatch ({len(reference_files)} files, {len(reference_urls)} URLs)"
            )
        for reference_name in reference_files:
            reference_path = Path(reference_name)
            if reference_path.is_absolute() or ".." in reference_path.parts:
                errors.append(f"{label}: unsafe reference filename {reference_name!r}")
        for reference_url in reference_urls:
            scheme = urllib.parse.urlparse(reference_url).scheme
            if reference_mode == "https" and scheme != "https":
                errors.append(f"{label}: reference URL must use HTTPS")
            elif reference_mode == "local":
                if scheme:
                    errors.append(f"{label}: reference must be a local path, got {scheme}:// URL")
                elif not Path(reference_url).is_absolute():
                    errors.append(f"{label}: local reference path must be absolute: {reference_url!r}")
            elif reference_mode == "any" and scheme not in ("", "https"):
                errors.append(f"{label}: reference must be an HTTPS URL or a local path")

        prompt_urls = [url.rstrip(".,);]") for url in PROMPT_URL_RE.findall(prompt)]
        if not reference_urls and prompt_urls:
            warnings.append(f"{label}: prompt contains {len(prompt_urls)} URL(s), but no references will be staged")

        rubric = row.get("rubric_json")
        if not isinstance(rubric, list) or not all(isinstance(item, dict) for item in rubric):
            errors.append(f"{label}: rubric_json must be a list of objects")

    return errors, warnings


def apply_reference_overrides(
    rows: list[dict[str, Any]], override_path: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_overrides = json.loads(override_path.read_text(encoding="utf-8"))
    if not isinstance(raw_overrides, dict):
        raise ValueError(f"{override_path}: top-level value must be an object keyed by task_id")

    output_rows = copy.deepcopy(rows)
    row_by_id = {row.get("task_id"): row for row in output_rows}
    applied: list[str] = []
    for task_id, override in raw_overrides.items():
        if task_id not in row_by_id:
            raise ValueError(f"{override_path}: unknown task_id {task_id!r}")
        if not isinstance(override, dict) or set(override) != {"reference_files", "reference_file_urls"}:
            raise ValueError(f"{override_path}: {task_id} must contain only reference_files and reference_file_urls")
        files = override["reference_files"]
        urls = override["reference_file_urls"]
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError(f"{override_path}: {task_id}.reference_files must be a string list")
        if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
            raise ValueError(f"{override_path}: {task_id}.reference_file_urls must be a string list")
        if not files or len(files) != len(urls):
            raise ValueError(f"{override_path}: {task_id} must define aligned, non-empty reference lists")

        row = row_by_id[task_id]
        current_files = row.get("reference_files")
        current_urls = row.get("reference_file_urls")
        if current_files or current_urls:
            if current_files == files and current_urls == urls:
                continue
            raise ValueError(f"{override_path}: refusing to replace existing references for {task_id}")
        prompt = str(row.get("prompt", ""))
        absent_urls = [url for url in urls if url not in prompt]
        if absent_urls:
            raise ValueError(
                f"{override_path}: refusing {task_id} override because an added URL is absent from its prompt"
            )
        row["reference_files"] = files
        row["reference_file_urls"] = urls
        applied.append(task_id)
    return output_rows, applied


def _probe_local(task_id: str, filename: str, path: str) -> tuple[str, str, bool, str]:
    candidate = Path(path)
    if not candidate.exists():
        return task_id, filename, False, "missing"
    if not candidate.is_file():
        return task_id, filename, False, "not a file"
    if not os.access(candidate, os.R_OK):
        return task_id, filename, False, "unreadable"
    return task_id, filename, True, "ok"


def _probe_url(task_id: str, filename: str, url: str, timeout: float) -> tuple[str, str, bool, str]:
    headers = {"User-Agent": "nemo-gym-gdpval-preflight/1.0"}
    try:
        request = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = response.status
        return task_id, filename, 200 <= code < 400, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        if exc.code not in {405, 501}:
            return task_id, filename, False, f"HTTP {exc.code}"
    except Exception as exc:
        return task_id, filename, False, type(exc).__name__

    try:
        request = urllib.request.Request(url, headers=headers | {"Range": "bytes=0-0"}, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = response.status
            response.read(1)
        return task_id, filename, 200 <= code < 400, f"HTTP {code}"
    except urllib.error.HTTPError as exc:
        return task_id, filename, False, f"HTTP {exc.code}"
    except Exception as exc:
        return task_id, filename, False, type(exc).__name__


def check_reference_urls(
    rows: list[dict[str, Any]], *, workers: int, timeout: float, reference_mode: str = "https"
) -> list[str]:
    references = [
        (str(row["task_id"]), filename, url)
        for row in rows
        for filename, url in zip(row["reference_files"], row["reference_file_urls"])
    ]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for task_id, filename, locator in references:
            is_local = not urllib.parse.urlparse(locator).scheme
            if reference_mode == "local" or (reference_mode == "any" and is_local):
                futures.append(executor.submit(_probe_local, task_id, filename, locator))
            else:
                futures.append(executor.submit(_probe_url, task_id, filename, locator, timeout))
        for future in as_completed(futures):
            task_id, filename, ok, status = future.result()
            if not ok:
                failures.append(f"{task_id} {filename}: {status}")
    print(f"Reference URL preflight: {len(references) - len(failures)}/{len(references)} reachable")
    return sorted(failures)


def check_model_endpoint(*, base_url: str, model_name: str, api_key_env: str, timeout: float) -> list[str]:
    """Require an exact model-id match without putting the API key in argv."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return [f"model API key environment variable {api_key_env!r} is unset"]
    models_url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "nemo-gym-gdpval-preflight/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return [f"model endpoint /models returned HTTP {exc.code}"]
    except Exception as exc:
        return [f"model endpoint /models failed with {type(exc).__name__}"]

    data = payload.get("data") if isinstance(payload, dict) else None
    model_ids = {item.get("id") for item in data or [] if isinstance(item, dict) and isinstance(item.get("id"), str)}
    if model_name not in model_ids:
        return [f"exact served model id {model_name!r} was not present in /models ({len(model_ids)} ids returned)"]
    print(f"Model endpoint preflight: exact model id {model_name!r} is available")
    return []


def _failure_sidecar_path(rollouts_path: Path) -> Path:
    return rollouts_path.with_name(f"{rollouts_path.stem}_failures.jsonl")


def _materialized_path(rollouts_path: Path) -> Path:
    return rollouts_path.with_name(f"{rollouts_path.stem}_materialized_inputs.jsonl")


def validate_rollouts(
    rollout_path: Path,
    *,
    expected_task_ids: set[str],
    deliverables_dir: Path | None,
    require_deliverable: bool,
    expected_response_model: str | None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not rollout_path.is_file():
        return [f"rollout file does not exist: {rollout_path}"], warnings

    rows = _read_jsonl(rollout_path)
    task_ids = [str(row.get("task_id")) for row in rows]
    if len(rows) != len(expected_task_ids):
        errors.append(f"expected {len(expected_task_ids)} rollout rows, found {len(rows)}")
    if len(task_ids) != len(set(task_ids)):
        errors.append("rollout file contains duplicate task_ids")
    missing = sorted(expected_task_ids.difference(task_ids))
    unexpected = sorted(set(task_ids).difference(expected_task_ids))
    if missing:
        errors.append(f"missing rollout task_ids: {', '.join(missing)}")
    if unexpected:
        errors.append(f"unexpected rollout task_ids: {', '.join(str(item) for item in unexpected)}")

    models: set[str] = set()
    finish_count = 0
    tasks_with_deliverables = 0
    for row in rows:
        task_id = str(row.get("task_id", "unknown"))
        if row.get("execute_only") is not True:
            errors.append(f"{task_id}: rollout is not marked execute_only=true")
        if row.get("reward") is not None or row.get("judge_response") is not None:
            errors.append(f"{task_id}: judge/reward fields are populated in an execute-only run")
        response = row.get("response")
        if not isinstance(response, dict):
            errors.append(f"{task_id}: missing response object")
        else:
            if response.get("error") is not None:
                errors.append(f"{task_id}: response.error is populated")
            response_model = response.get("model")
            if isinstance(response_model, str):
                models.add(response_model)
            if expected_response_model is not None and response_model != expected_response_model:
                errors.append(
                    f"{task_id}: expected response model {expected_response_model!r}, found {response_model!r}"
                )
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, dict) or agent_ref.get("name") != "gdpval_stirrup_agent":
            errors.append(f"{task_id}: unexpected or missing agent_ref")

        if deliverables_dir is not None and task_id in expected_task_ids:
            expected_task_dir = deliverables_dir / f"task_{task_id}" / "repeat_0"
            reported_dir = row.get("deliverables_dir")
            if not isinstance(reported_dir, str) or Path(reported_dir).resolve() != expected_task_dir.resolve():
                errors.append(f"{task_id}: rollout deliverables_dir does not match the dedicated run cache")
            finish_marker = expected_task_dir / "finish_params.json"
            if not finish_marker.is_file():
                errors.append(f"{task_id}: missing finish marker {finish_marker}")
            else:
                finish_count += 1
                try:
                    finish_params = json.loads(finish_marker.read_text(encoding="utf-8"))
                    paths = finish_params.get("paths", []) if isinstance(finish_params, dict) else []
                    existing = [path for path in paths if (expected_task_dir / Path(str(path)).name).is_file()]
                    if existing:
                        tasks_with_deliverables += 1
                    elif require_deliverable:
                        errors.append(f"{task_id}: rollout finished without persisting a deliverable file")
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"{task_id}: invalid finish marker: {type(exc).__name__}")

    materialized_path = _materialized_path(rollout_path)
    if not materialized_path.is_file():
        errors.append(f"missing materialized input file: {materialized_path}")
    else:
        materialized = _read_jsonl(materialized_path)
        if len(materialized) != len(expected_task_ids):
            errors.append(f"expected {len(expected_task_ids)} materialized input rows, found {len(materialized)}")
        materialized_ids = {str(row.get("task_id")) for row in materialized}
        if materialized_ids != expected_task_ids:
            errors.append("materialized input task_ids do not match the requested batch")
        if any(
            not isinstance(row.get("agent_ref"), dict) or row["agent_ref"].get("name") != "gdpval_stirrup_agent"
            for row in materialized
        ):
            errors.append("materialized inputs contain an unexpected or missing agent_ref")

    failures_path = _failure_sidecar_path(rollout_path)
    if failures_path.is_file():
        failure_count = len(_read_jsonl(failures_path))
        if failure_count:
            warnings.append(f"failure history contains {failure_count} row(s): {failures_path}")

    if models:
        print(f"Rollout response model(s): {', '.join(sorted(models))}")
    if expected_response_model is not None and models != {expected_response_model}:
        errors.append(
            f"expected only response model {expected_response_model!r}, found {', '.join(sorted(models)) or 'none'}"
        )
    if deliverables_dir is not None:
        print(
            f"Deliverable cache: {finish_count}/{len(expected_task_ids)} finish markers; "
            f"{tasks_with_deliverables} task(s) with a declared persisted file"
        )
    return errors, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source GDPVal-format Gym JSONL")
    parser.add_argument(
        "--task-id-pattern",
        default=DEFAULT_TASK_ID_PATTERN,
        help="Regex every task_id must fully match (default: any non-empty token). "
        "Set it from the dataset profile to pin a batch's id format.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=REFERENCE_MODES,
        default="https",
        help="How reference locators are expressed: https URLs, absolute local paths, or either",
    )
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--expected-sha256", help="Expected SHA-256 of the immutable source JSONL")
    parser.add_argument("--reference-overrides", type=Path, help="Explicit prompt-backed reference repairs")
    parser.add_argument("--write-launch-input", type=Path, help="Write the validated, repaired launch JSONL")
    parser.add_argument("--smoke-task-id", help="Task to select when writing or validating a smoke run")
    parser.add_argument("--write-smoke-input", type=Path, help="Write a one-row smoke JSONL")
    parser.add_argument("--check-reference-urls", action="store_true")
    parser.add_argument("--url-workers", type=int, default=16)
    parser.add_argument("--url-timeout", type=float, default=30.0)
    parser.add_argument("--check-model-endpoint", action="store_true")
    parser.add_argument("--model-base-url")
    parser.add_argument("--model-name")
    parser.add_argument("--model-api-key-env", default="POLICY_API_KEY")
    parser.add_argument("--rollouts", type=Path, help="Validate a completed execute-only rollout JSONL")
    parser.add_argument("--deliverables-dir", type=Path, help="Validate per-task finish markers here")
    parser.add_argument("--require-deliverable", action="store_true", help="Require a persisted deliverable file")
    parser.add_argument("--expected-response-model", help="Require this exact response.model in every rollout")
    parser.add_argument(
        "--require-private-files",
        action="store_true",
        help="Reject input/reference files readable or writable by group/other users",
    )
    return parser.parse_args()


def _validate_private_file(path: Path) -> str | None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        return f"private file has group/world permissions {mode:04o}: {path}"
    return None


def main() -> int:
    args = _parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    if not args.input.is_file():
        print(f"ERROR: input does not exist: {args.input}")
        return 2
    if args.require_private_files:
        private_paths = [args.input]
        if args.reference_overrides:
            private_paths.append(args.reference_overrides)
        for private_path in private_paths:
            if not private_path.is_file():
                errors.append(f"private file does not exist: {private_path}")
                continue
            permission_error = _validate_private_file(private_path)
            if permission_error:
                errors.append(permission_error)
    source_hash = _sha256(args.input)
    if args.expected_sha256 and source_hash != args.expected_sha256:
        errors.append(f"source SHA-256 mismatch: expected {args.expected_sha256}, found {source_hash}")

    try:
        rows = _read_jsonl(args.input)
        applied: list[str] = []
        if args.reference_overrides:
            rows, applied = apply_reference_overrides(rows, args.reference_overrides)
            if applied:
                print(f"Applied reference overrides: {', '.join(applied)}")
        input_errors, input_warnings = validate_input_rows(
            rows,
            expected_count=args.expected_count,
            task_id_pattern=args.task_id_pattern,
            reference_mode=args.reference_mode,
        )
        errors.extend(input_errors)
        warnings.extend(input_warnings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    if args.write_launch_input and not errors:
        _write_jsonl_atomic(args.write_launch_input, rows)
        print(f"Wrote launch input: {args.write_launch_input} (sha256={_sha256(args.write_launch_input)})")

    if args.write_smoke_input:
        if not args.smoke_task_id:
            errors.append("--write-smoke-input requires --smoke-task-id")
        else:
            smoke_rows = [row for row in rows if row.get("task_id") == args.smoke_task_id]
            if len(smoke_rows) != 1:
                errors.append(f"smoke task {args.smoke_task_id!r} matched {len(smoke_rows)} rows")
            elif not errors:
                _write_jsonl_atomic(args.write_smoke_input, smoke_rows)
                print(f"Wrote smoke input: {args.write_smoke_input}")

    if args.check_reference_urls and not errors:
        failures = check_reference_urls(
            rows, workers=args.url_workers, timeout=args.url_timeout, reference_mode=args.reference_mode
        )
        errors.extend(f"reference URL unreachable: {failure}" for failure in failures)

    if args.check_model_endpoint and not errors:
        if not args.model_base_url or not args.model_name:
            errors.append("--check-model-endpoint requires --model-base-url and --model-name")
        else:
            errors.extend(
                check_model_endpoint(
                    base_url=args.model_base_url,
                    model_name=args.model_name,
                    api_key_env=args.model_api_key_env,
                    timeout=args.url_timeout,
                )
            )

    if args.rollouts and not errors:
        expected_ids = {str(row["task_id"]) for row in rows}
        if args.smoke_task_id:
            expected_ids = {args.smoke_task_id}
        # The agent falls back to the literal "default" when a dataset row
        # declares no model (stirrup_agent: `body.model or "default"`), because
        # the served model comes from the endpoint config rather than the row.
        # Asserting response.model == POLICY_MODEL_NAME can therefore never hold
        # for such a dataset, which made `run_gdpval_rollouts.sh validate`
        # permanently unpassable for it. Only assert when the rows opt in.
        expected_response_model = args.expected_response_model
        if expected_response_model is not None and not any(
            (row.get("responses_create_params") or {}).get("model") for row in rows
        ):
            warnings.append(
                "dataset rows declare no model; skipping the response.model assertion "
                f"(rollouts record the agent default, not {expected_response_model!r})"
            )
            expected_response_model = None
        rollout_errors, rollout_warnings = validate_rollouts(
            args.rollouts,
            expected_task_ids=expected_ids,
            deliverables_dir=args.deliverables_dir,
            require_deliverable=args.require_deliverable,
            expected_response_model=expected_response_model,
        )
        errors.extend(rollout_errors)
        warnings.extend(rollout_warnings)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1

    print(f"Input validation passed: {len(rows)} rows, source sha256={source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
