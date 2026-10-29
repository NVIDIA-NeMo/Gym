# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym external-rollout-driver: standard agent, frozen adaptive delivery, separate verification."""

import asyncio
import copy
import fcntl
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME
from resources_servers.agentic_if_realworld.benchmark.core import (
    PHASES,
    Dataset,
    digest,
    indexed,
    judge,
    read,
    runtime_identity,
    source_preflight,
    summarize,
    summary_metrics,
    write,
)
from resources_servers.agentic_if_realworld.benchmark.registry import private_path
from resources_servers.agentic_if_realworld.benchmark.runner import run_cell
from resources_servers.agentic_if_realworld.benchmark.transport import delivered


LOGGER = logging.getLogger(__name__)


async def run_rollout_collection(rollout_collection_config, global_config_dict) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.observability.model_calls import (
        await_model_call_capture_quiescence,
        merge_model_call_capture_into_record,
        model_call_capture_dirs_from_config,
    )
    from nemo_gym.path_utils import materialized_path_for
    from nemo_gym.rollout_collection import RolloutCollectionHelper, _attach_trajectory_record
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
    from nemo_gym.secret_utils import recursively_hide_secrets
    from nemo_gym.server_utils import get_response_json, raise_for_status

    config = rollout_collection_config
    if config.num_repeats != 1 or config.fan_out or config.responses_create_params:
        raise ValueError("Frozen benchmark prohibits repeats, fan-out, or prompt/sampling overrides")
    resource = global_config_dict["agentic_if_realworld_resources"]["resources_servers"]["agentic_if_realworld"]
    model = global_config_dict["policy_model"]["responses_api_models"]["agentic_if_text"]
    dataset = Dataset(resource["dataset_dir"], synthetic=resource["synthetic"])
    if not dataset.synthetic:
        source_preflight(resource["split_registry_path"], resource["ground_truth_path"], directory=dataset.directory)
    directory = private_path(Path(resource["receipt_dir"]))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings = {
        "policy": {k: model[k] for k in ["endpoint", "model", "api_key_env"]},
        "judge": {
            k: resource[k]
            for k in [
                "judge_endpoint",
                "judge_model",
                "judge_api_key_env",
                "judge_reasoning",
            ]
        },
        "policy_assets": dataset.pin["policy_assets"],
        "profile": "adaptive_first_complete_v1",
        "response_source": resource.get("response_source", "raw_receipt"),
        "runtime": runtime_identity(),
        "observability": {
            "enabled": global_config_dict.get("observability_enabled", False),
            "capture_dir": global_config_dict.get("model_call_capture_dir"),
        },
    }
    settings = OmegaConf.to_container(OmegaConf.create(settings), resolve=True)
    helper = RolloutCollectionHelper()
    capture_dirs = model_call_capture_dirs_from_config(global_config_dict)
    for capture_dir in capture_dirs:
        private_path(capture_dir)
    client = helper.setup_server_client()
    raw_rows = list((await asyncio.to_thread(indexed, Path(config.input_jsonl_fpath), "task_id")).values())
    canonical_rows = [
        {k: row[k] for k in ["task_id", "verifier_metadata", "responses_create_params"]} for row in raw_rows
    ]
    if canonical_rows != dataset.prepared():
        raise ValueError("Prepared input differs from frozen dataset")
    selected_rows = raw_rows[: config.limit] if config.limit else raw_rows
    rows = helper.preprocess_examples(
        selected_rows,
        agent_map={"_default": "agentic_if_realworld_agent"},
        global_config_dict=global_config_dict,
    )
    by_id = {r["task_id"]: r for r in rows}
    output = private_path(Path(config.output_jsonl_fpath))
    if output.exists() and not config.resume_from_cache:
        raise ValueError("Existing rollout output requires explicit resume or a new run directory")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if config.resume_from_cache and output.exists():
            for identifier in indexed(output, "task_id"):
                if identifier not in by_id or not (directory / "cells" / (judge.sha(identifier) + ".json")).exists():
                    raise ValueError("Missing journal for published rollout; reconcile this run before resuming")
        snapshot = OmegaConf.create(global_config_dict)
        OmegaConf.resolve(snapshot)
        recursively_hide_secrets(snapshot)
        snapshot_path = private_path(output.parent / "resolved-config.yaml")
        if snapshot_path.exists():
            if not config.resume_from_cache:
                raise ValueError("Existing resolved config requires resume or a new run directory")
        inventory = private_path(materialized_path_for(output))
        inventory_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        if inventory.exists():
            if not config.resume_from_cache or inventory.read_text(encoding="utf-8") != inventory_text:
                raise ValueError("Materialized rollout inventory mismatch")
        else:
            if config.resume_from_cache and (output.exists() or (directory / "cells").exists()):
                raise ValueError("Missing materialized rollout inventory; preserve this run and start a new directory")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=inventory.parent, delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(inventory_text)
                    stream.flush()
                    os.fsync(stream.fileno())
                    temporary.replace(inventory)
                finally:
                    temporary.unlink(missing_ok=True)
        if not snapshot_path.exists():
            # JSON is valid YAML; reuse the atomic private receipt writer.
            write(snapshot_path, OmegaConf.to_container(snapshot, resolve=True))
        results, failures, validated_results = [], [], {}
        writing_failed = False
        # Each task keeps its own journal and receipts, so tasks run concurrently (sequential by default).
        slots = asyncio.Semaphore(config.num_samples_in_parallel or 1)
        writing = asyncio.Lock()

        async def collect(row: dict, *, validate_only: bool = False) -> dict | None:
            nonlocal writing_failed
            identifier = row["task_id"]

            async def generate(prepared, seed, cap, receipt_id):
                attempt = copy.deepcopy(by_id[prepared["task_id"]])
                attempt[ATTEMPT_INDEX_KEY_NAME] = PHASES.index((seed, cap))
                attempt["responses_create_params"].update(
                    max_output_tokens=cap,
                    metadata={"seed": str(seed), "receipt_id": receipt_id},
                )
                completions = helper.run_examples([attempt], route_failures_to_sidecar=True)
                try:
                    request_row, agent_result = await next(iter(completions))
                    rollout = {**request_row, **agent_result}
                finally:
                    await completions.aclose()
                path = directory / "model" / (receipt_id + ".json")
                if not path.exists():
                    # Failed /run may have dispatched upstream: never automatically regenerate.
                    raise ValueError("Missing model receipt; reconcile ambiguous agent dispatch")
                receipt = read(path)
                if receipt["status"] == "started":
                    raise ValueError("Ambiguous upstream dispatch")
                try:
                    canonical = delivered(receipt["response"])
                    complete = True
                except (ValueError, TypeError, KeyError):
                    canonical, complete = None, False
                if complete and (
                    "response" not in rollout
                    or (rollout["response"].get("metadata") or {}).get("receipt_id") != receipt_id
                ):
                    raise ValueError("Complete delivery missing trusted Gym projection; reconcile receipt")
                return {
                    "complete": complete,
                    "terminal": receipt.get("terminal", False),
                    "canonical_response": canonical,
                    "rollout": rollout,
                    "receipt_path": str(path),
                    "receipt_sha256": digest(path),
                }

            async def verify(selected, criteria, identifier=identifier, *, cached_only=False, validate_only=False):
                path = Path(selected["receipt_path"])
                if digest(path) != selected["receipt_sha256"]:
                    raise ValueError("Saved delivery receipt changed")
                body = {
                    **by_id[identifier],
                    ATTEMPT_INDEX_KEY_NAME: selected["rollout"][ATTEMPT_INDEX_KEY_NAME],
                    "responses_create_params": selected["rollout"]["responses_create_params"],
                    "response": selected["rollout"]["response"],
                }
                if cached_only:
                    body["cached_judge_only"] = True
                if validate_only:
                    body["validate_judge_cache"] = True
                response = await client.post(
                    server_name="agentic_if_realworld_resources",
                    url_path="/verify",
                    json=body,
                )
                try:
                    await raise_for_status(response)
                    result = await get_response_json(response)
                    result.pop("cached_judge_only", None)
                    result.pop("validate_judge_cache", None)
                    return result
                finally:
                    response.release()

            if validate_only:
                return await run_cell(
                    dataset,
                    identifier,
                    directory=directory,
                    settings=settings,
                    generate=generate,
                    verify=verify,
                    resume=True,
                    validate_only=True,
                )
            async with slots:
                # After a failure, admit no new task, but let started ones settle their journals.
                if failures:
                    return
                try:
                    result = validated_results.get(identifier)
                    if result is None:
                        result = await run_cell(
                            dataset,
                            identifier,
                            directory=directory,
                            settings=settings,
                            generate=generate,
                            verify=verify,
                            resume=config.resume_from_cache,
                        )
                    # Exactly one canonical row per task; attempt placeholders never enter metrics.
                    state = read(directory / "cells" / (judge.sha(identifier) + ".json"))
                    landed = state.get("selected", state["attempts"][-1]).get("rollout", {})
                    record = {**row, **landed, **result}
                    # The agent skips inline judging; the driver verifies each selected delivery.
                    record["verification_skipped"] = "selected" not in state
                    if capture_dirs:
                        rollout_id = maybe_rollout_id_from_run_body(record)
                        if rollout_id:
                            await await_model_call_capture_quiescence(rollout_id, capture_dirs)
                        merge_model_call_capture_into_record(record, capture_dirs, include_payloads=True)
                        _attach_trajectory_record(record, record)
                    async with writing:
                        try:
                            results.append(result)
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                            write_reports(staged_output, results, dataset)
                        except Exception:
                            writing_failed = True
                            raise
                except (Exception, asyncio.CancelledError) as error:
                    LOGGER.error("Task %s failed: %r", identifier, error)
                    failures.append(error)

        if config.resume_from_cache:
            # Reject any saved task before dispatching new work or replacing published artifacts.
            for row in rows:
                validated_results[row["task_id"]] = await collect(row, validate_only=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{output.stem}-") as pending:
            staged_output = Path(pending) / output.name
            stream = os.fdopen(os.open(staged_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")
            completed = False
            try:
                await asyncio.gather(*(collect(row) for row in rows))
                if failures:
                    raise failures[0]
                write_reports(staged_output, results, dataset)
                write(
                    staged_output.with_suffix(".run.json"),
                    {
                        "created_at": datetime.now(UTC).isoformat(),
                        "dataset": dataset.report,
                        "settings": settings,
                        "settings_sha256": judge.sha(settings),
                        "new_protocol_not_historical_reproduction": True,
                    },
                )
                completed = True
            finally:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                finally:
                    stream.close()
                # A rejected resume preserves published artifacts; fresh interruptions retain settled rows.
                if completed or (not config.resume_from_cache and not writing_failed):
                    staged_output.replace(output)
                    for report in Path(pending).glob("*.json"):
                        report.replace(output.parent / report.name)
    finally:
        lock.close()


def write_reports(output: Path, results: list[dict], dataset: Dataset) -> None:
    """Expose full-denominator accuracy to Gym/EFB; incomplete accuracy stays null."""
    summary = summarize(results, dataset)
    write(output.with_suffix(".summary.json"), summary)
    metrics = summary_metrics(summary)
    write(
        output.with_name(output.stem + "_aggregate_metrics.json"),
        [
            {
                "agent_ref": {"name": "agentic_if_realworld_agent"},
                "agent_metrics": metrics,
                "key_metrics": metrics,
            }
        ],
    )
