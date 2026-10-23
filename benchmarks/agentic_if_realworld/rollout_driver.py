# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym external-rollout-driver: standard agent, frozen adaptive delivery, separate verification."""

import asyncio
import copy
import fcntl
import json
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
    write,
)
from resources_servers.agentic_if_realworld.benchmark.registry import private_path
from resources_servers.agentic_if_realworld.benchmark.runner import run_cell
from resources_servers.agentic_if_realworld.benchmark.transport import delivered


async def run_rollout_collection(rollout_collection_config, global_config_dict) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.observability.model_calls import (
        await_model_call_capture_quiescence,
        merge_model_call_capture_into_record,
        model_call_capture_dirs_from_config,
    )
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
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (directory / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = OmegaConf.create(global_config_dict)
        OmegaConf.resolve(snapshot)
        recursively_hide_secrets(snapshot)
        snapshot_path = private_path(output.parent / "resolved-config.yaml")
        if snapshot_path.exists():
            if not config.resume_from_cache:
                raise ValueError("Existing resolved config requires resume or a new run directory")
        else:
            # JSON is valid YAML; reuse the atomic private receipt writer.
            write(snapshot_path, OmegaConf.to_container(snapshot, resolve=True))
        results, rollouts = [], []
        for row in rows:
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

            async def verify(selected, criteria, identifier=identifier, *, cached_only=False):
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
                response = await client.post(
                    server_name="agentic_if_realworld_resources",
                    url_path="/verify",
                    json=body,
                )
                try:
                    await raise_for_status(response)
                    result = await get_response_json(response)
                    result.pop("cached_judge_only", None)
                    return result
                finally:
                    response.release()

            result = await run_cell(
                dataset,
                identifier,
                directory=directory,
                settings=settings,
                generate=generate,
                verify=verify,
                resume=config.resume_from_cache,
            )
            results.append(result)
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
            rollouts.append(record)
            fd = __import__("os").open(
                output,
                __import__("os").O_WRONLY | __import__("os").O_CREAT | __import__("os").O_TRUNC,
                0o600,
            )
            with __import__("os").fdopen(fd, "w") as stream:
                for value in rollouts:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            write_reports(output, results, dataset)
        write_reports(output, results, dataset)
        write(
            output.with_suffix(".run.json"),
            {
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": dataset.report,
                "settings": settings,
                "settings_sha256": judge.sha(settings),
                "new_protocol_not_historical_reproduction": True,
            },
        )
    finally:
        lock.close()


def write_reports(output: Path, results: list[dict], dataset: Dataset) -> None:
    """Expose full-denominator accuracy to Gym/EFB; incomplete accuracy stays null."""
    summary = summarize(results, dataset)
    write(output.with_suffix(".summary.json"), summary)
    metrics = {
        "composite/complete_accuracy": summary["complete_accuracy"],
        **{key: summary[key] for key in ["scoring_coverage", "expected_items", "pass", "fail", "not_scored"]},
    }
    for view, values in summary["views"].items():
        metrics[view + "/complete_accuracy"] = values["complete_accuracy"]
        metrics[view + "/scoring_coverage"] = values["scoring_coverage"]
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
