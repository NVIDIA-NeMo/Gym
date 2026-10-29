# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks.agentic_if_realworld.rollout_driver import write_reports
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import E2ERolloutCollectionConfig
from nemo_gym.rollout_reverification import RolloutReverificationConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.agentic_if_realworld.app import FixtureVerifier, ResourcesConfig, ResourcesServer, VerifyRequest
from resources_servers.agentic_if_realworld.benchmark.core import Dataset, digest, judge, read, unscored, write
from resources_servers.agentic_if_realworld.benchmark.synthetic import create, delivery


def test_complete_accuracy_stays_null_until_all_tasks_scored(tmp_path):
    verifier = FixtureVerifier()
    try:
        output = tmp_path / "rollouts.jsonl"
        write_reports(output, [], verifier.dataset)
        metrics = read(tmp_path / "rollouts_aggregate_metrics.json")[0]["key_metrics"]
        assert metrics["composite/complete_accuracy"] is None
        assert metrics["not_scored"] == 1 and metrics["scoring_coverage"] == 0
        body = VerifyRequest.model_validate(delivery(verifier.dataset, verifier.receipt_dir, "OK.."))
        result = asyncio.run(verifier.verify(body)).model_dump()
        write_reports(output, [result], verifier.dataset)
        metrics = read(tmp_path / "rollouts_aggregate_metrics.json")[0]["key_metrics"]
        assert metrics["composite/complete_accuracy"] == 0
        assert metrics["fail"] == 1 and metrics["scoring_coverage"] == 1
    finally:
        verifier.work.cleanup()


@pytest.fixture
def aggregation(tmp_path):
    directory = tmp_path / "data"
    create(directory, examples=True)
    criteria = [json.loads(line) for line in (directory / "criteria.jsonl").read_text().splitlines()]
    for index, row in enumerate(criteria):
        if index >= 3:
            row["scoring_view"] = "supplemental_active_single_turn"
        extra = deepcopy(row["constraints"][0])
        extra.update(constraint_id="synthetic.literal_length", subtype="IF-LENGTH")
        row["constraints"].append(extra)
    (directory / "criteria.jsonl").write_text("".join(json.dumps(row) + "\n" for row in criteria))
    manifest = read(directory / "release_manifest.json")
    manifest["artifacts"]["criteria.jsonl"]["sha256"] = digest(directory / "criteria.jsonl")
    write(directory / "release_manifest.json", manifest)
    dataset = Dataset(directory, synthetic=True)
    config = ResourcesConfig(
        host="",
        port=0,
        entrypoint="app.py",
        name="agentic_if_realworld_resources",
        dataset_dir=str(directory),
        synthetic=True,
        receipt_dir=str(tmp_path / "receipts"),
        judge_endpoint="http://synthetic.invalid",
    )
    server = ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def record(index, verdict):
        identifier = sorted(dataset.inputs)[index]
        row = dataset.prepared_row(identifier)
        row.pop("responses_create_params")
        if verdict is None:
            result = unscored(identifier, "synthetic_judge_outage")
        else:
            result = {
                "item_id": identifier,
                "scoring_status": "scored",
                "validation_status": "valid",
                "policy_version": judge.VERSION,
                "item_verdict": verdict,
                "reward": float(verdict == "pass"),
                "mask_sample": False,
                "subtype_verdicts": {"IF-FORMAT": verdict, "IF-LENGTH": verdict},
            }
        return {**row, **result, "_ng_task_index": index, "evaluation_completed": verdict is not None}

    return server, dataset, record


def test_native_aggregate_uses_full_dataset_and_masks_placeholders(aggregation):
    server, _, record = aggregation
    rows = [record(0, "pass"), record(1, "fail"), record(2, None)]
    rows[1]["subtype_verdicts"]["IF-LENGTH"] = "pass"
    rows[-1].pop(
        "evaluation_completed"
    )  # Adaptive exhausted deliveries carry the same masked grade without this flag.
    aggregate = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))
    metrics = aggregate.agent_metrics
    assert metrics["composite/complete_accuracy"] is None
    assert (metrics["expected_items"], metrics["pass"], metrics["fail"], metrics["not_scored"]) == (5, 1, 1, 3)
    assert (metrics["masked"], metrics["unattempted"], metrics["scoring_coverage"]) == (1, 2, 0.4)
    assert metrics["original_failure_focused/expected_items"] == 3
    assert metrics["supplemental_active_single_turn/expected_items"] == 2
    assert metrics["subtypes/IF-FORMAT/expected_items"] == metrics["subtypes/IF-LENGTH/expected_items"] == 5
    assert metrics["subtypes/IF-FORMAT/pass"] == 1 and metrics["subtypes/IF-LENGTH/pass"] == 2
    assert metrics["subtype_denominators_overlap"] is True
    assert aggregate.key_metrics["composite/complete_accuracy"] is None
    assert not any(key.startswith("mean") for key in aggregate.key_metrics)
    complete = [record(index, "fail" if index == 1 else "pass") for index in range(5)]
    aggregate = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=complete)))
    assert aggregate.key_metrics["composite/complete_accuracy"] == 0.8
    assert aggregate.agent_metrics["not_scored"] == aggregate.agent_metrics["masked"] == 0
    assert len(aggregate.group_level_metrics) == 5
    masked = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[record(0, None)])))
    assert masked.key_metrics["composite/complete_accuracy"] is None
    assert (masked.key_metrics["masked"], masked.key_metrics["unattempted"], masked.key_metrics["not_scored"]) == (
        1,
        4,
        5,
    )
    empty = asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=[])))
    assert empty.key_metrics["expected_items"] == empty.key_metrics["unattempted"] == 5
    assert empty.key_metrics["composite/complete_accuracy"] is None


@pytest.mark.parametrize("damage", ["duplicate", "unknown", "binding", "reward", "mask"])
def test_native_aggregation_rejects_inconsistent_saved_results(aggregation, damage):
    server, _, record = aggregation
    rows = [record(0, "pass")]
    if damage == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif damage == "unknown":
        rows[0]["item_id"] = rows[0]["task_id"] = "not-a-frozen-task"
    elif damage == "binding":
        rows[0]["verifier_metadata"]["dataset_sha256"] = "0" * 64
    elif damage == "reward":
        rows[0]["reward"] = 0.0
    else:
        rows[0]["mask_sample"] = True
    with pytest.raises(ValueError):
        asyncio.run(server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rows)))


@pytest.mark.parametrize(
    "profile",
    [
        "benchmarks/agentic_if_realworld/config.yaml",
        "benchmarks/agentic_if_realworld/native.yaml",
        "resources_servers/agentic_if_realworld/configs/reverify_native.yaml",
    ],
)
def test_all_profiles_disable_raw_rollout_upload_by_default(profile, tmp_path):
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [profile]},
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    values = OmegaConf.to_container(resolved, resolve=True)
    output = str(tmp_path / "rollouts.jsonl")
    collection = E2ERolloutCollectionConfig.model_validate(
        {**values, "split": "benchmark", "output_jsonl_fpath": output}
    )
    reverification = RolloutReverificationConfig.model_validate(
        {
            **values,
            "materialized_inputs_jsonl_fpath": str(tmp_path / "inputs.jsonl"),
            "rollouts_jsonl_fpath": str(tmp_path / "saved.jsonl"),
            "output_jsonl_fpath": output,
        }
    )
    assert collection.upload_rollouts is False
    assert reverification.upload_rollouts is False
