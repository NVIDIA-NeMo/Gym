# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.synthetic_tool_use_simulation.app import SyntheticToolUseSeedSessionRequest
from resources_servers.synthetic_tool_use_simulation.scripts.build_synthetic_tool_use_dataset import (
    ArtifactValidationError,
    agent_system_message,
    build_sample_dataset,
    validate_tool_schema,
)


def default_tool() -> dict:
    return {
        "name": "lookup_order",
        "doc": "Look up an order.",
        "params": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        "returns": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
    }


@pytest.mark.parametrize(
    "schema",
    [
        None,
        True,
        False,
        [],
        {},
        {"description": "Markerless schemas remain outside the generated-data contract."},
    ],
)
def test_validate_tool_schema_does_not_broaden_existing_input_contract(schema) -> None:
    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_tool_schema(schema, tool_name="lookup_order", field_name="params")

    assert exc_info.value.reason == "invalid_tool_params_schema"


@pytest.mark.parametrize(
    ("field_name", "schema"),
    [
        ("params", {"type": "object", "required": "order_id"}),
        ("returns", {"type": "not-a-json-schema-type"}),
    ],
)
def test_validate_tool_schema_rejects_malformed_json_schema(field_name: str, schema: dict) -> None:
    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_tool_schema(schema, tool_name="lookup_order", field_name=field_name)

    assert exc_info.value.reason == f"invalid_tool_{field_name}_schema"
    assert "invalid" in exc_info.value.detail


def default_scenario(**overrides) -> dict:
    scenario = {
        "customer_persona": "Concise customer",
        "reason_for_contact": "Check my order.",
        "customer_details": "Order ID ORD-123.",
        "unknown_info": "Shipping status",
        "task_instructions": "Look up the order and explain the status.",
        "representative_domain": "order support",
        "outside_policy_scope": False,
    }
    scenario.update(overrides)
    return scenario


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_domain(
    source_dir: Path,
    *,
    domain_id: str = "0",
    policy: str = "Use tools when account state is needed.",
    tools: list[dict] | None = None,
    scenarios: list[dict] | None = None,
) -> Path:
    domain_dir = source_dir / domain_id
    domain_dir.mkdir(parents=True)
    (domain_dir / "policy.md").write_text(policy, encoding="utf-8")
    write_jsonl(domain_dir / "tools.jsonl", tools if tools is not None else [default_tool()])
    write_jsonl(
        domain_dir / "scenarios" / "Qwen3-235B-A22B-Thinking-2507" / "scenarios_0000.jsonl",
        scenarios if scenarios is not None else [default_scenario()],
    )
    return domain_dir


def test_build_sample_dataset_converts_raw_seed_row(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(source_dir)
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=1)

    assert report["rows_written"] == 1
    assert report["sources"][0]["domains_accepted"] == 1
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["domain_name"] == "order support"
    assert row["tools"][0]["doc"] == "Look up an order."
    assert row["tools"][0]["params"]["required"] == ["order_id"]
    assert row["tools"][0]["description"] == "Look up an order."
    assert row["tools"][0]["parameters"]["required"] == ["order_id"]
    assert row["tools"][0]["strict"] is True
    assert row["responses_create_params"]["input"] == [
        {
            "role": "system",
            "content": agent_system_message("Use tools when account state is needed."),
        }
    ]
    assert row["responses_create_params"]["tools"] == [
        {
            "type": "function",
            "name": "lookup_order",
            "description": "Look up an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
            "strict": True,
        }
    ]
    assert row["responses_create_params"]["parallel_tool_calls"] is False
    assert row["metadata"] == {
        "dataset_name": "synthetic_tool_use_sample",
        "domain_generator_model": "Qwen3-235B-A22B-Thinking-2507",
        "policy_tools_model": "DeepSeek-R1-0528",
        "scenario_generator_model": "Qwen3-235B-A22B-Thinking-2507",
        "source_name": "source_0",
        "source_index": 0,
        "domain_name": "order support",
        "representative_domain": "order support",
        "source_domain_index": 0,
        "domain_dir_name": "0",
        "num_tools": 1,
        "tool_names": ["lookup_order"],
        "scenario_index": 0,
        "scenario_file": "scenarios/Qwen3-235B-A22B-Thinking-2507/scenarios_0000.jsonl",
        "scenario_file_index": 0,
        "scenario_line": 1,
        "scenario_generator_run": "Qwen3-235B-A22B-Thinking-2507",
        "outside_policy_scope": False,
        "policy_num_chars": 39,
        "parallel_tool_calls": False,
    }
    assert row["source_artifacts"]["scenario_index"] == 0
    assert row["source_artifacts"]["source_name"] == "source_0"
    assert row["source_artifacts"]["scenario_file"] == ("scenarios/Qwen3-235B-A22B-Thinking-2507/scenarios_0000.jsonl")
    assert row["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "synthetic_tool_use_simulation_agent",
    }
    SyntheticToolUseSeedSessionRequest.model_validate(row)
    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])


def test_build_sample_dataset_can_materialize_parallel_tool_call_variant(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(source_dir)
    output_path = tmp_path / "sample_parallel.jsonl"
    report_path = tmp_path / "sample_parallel.report.json"

    report = build_sample_dataset(
        [source_dir],
        output_path,
        report_path,
        max_rows=1,
        parallel_tool_calls=True,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["parallel_tool_calls"] is True
    assert row["metadata"]["parallel_tool_calls"] is True
    assert row["responses_create_params"]["parallel_tool_calls"] is True
    assert row["responses_create_params"]["input"][0]["content"] == agent_system_message(
        "Use tools when account state is needed.",
        parallel_tool_calls=True,
    )
    assert "Make one or more tool calls." in row["responses_create_params"]["input"][0]["content"]
    NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])


def test_build_sample_dataset_rejects_empty_schemas(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    tool = default_tool()
    tool["params"] = {}
    make_domain(source_dir, tools=[tool])
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=1)

    assert output_path.read_text(encoding="utf-8") == ""
    assert report["rows_written"] == 0
    assert report["sources"][0]["skipped_domains"] == {"invalid_tool_params_schema": 1}


def test_build_sample_dataset_reports_malformed_schema(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    tool = default_tool()
    tool["params"] = {"type": "object", "required": "order_id"}
    make_domain(source_dir, tools=[tool])
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=1)

    assert output_path.read_text(encoding="utf-8") == ""
    assert report["rows_written"] == 0
    assert report["sources"][0]["skipped_domains"] == {"invalid_tool_params_schema": 1}
    assert "required" in report["sources"][0]["skip_examples"][0]["detail"]


def test_build_sample_dataset_rejects_tool_missing_returns(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    broken_tool = default_tool()
    broken_tool.pop("returns")
    make_domain(source_dir, tools=[broken_tool])
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=1)

    assert output_path.read_text(encoding="utf-8") == ""
    assert report["rows_written"] == 0
    assert report["sources"][0]["skipped_domains"] == {"missing_returns": 1}


def test_build_sample_dataset_rejects_only_leaky_scenario_line(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(
        source_dir,
        scenarios=[
            default_scenario(task_instructions='Call <｜DSML｜invoke name="lookup_order"> directly.'),
            default_scenario(task_instructions="Use the lookup tool normally."),
        ],
    )
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=2)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["customer_scenario"]["task_instructions"] == "Use the lookup tool normally."
    assert report["sources"][0]["scenarios_seen"] == 2
    assert report["sources"][0]["scenarios_accepted"] == 1
    assert report["sources"][0]["skipped_scenarios"] == {"dsml_leak": 1}


def test_build_sample_dataset_rejects_special_token_in_policy(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(source_dir, policy="Do not expose <｜Assistant｜> template markers.")
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset([source_dir], output_path, report_path, max_rows=1)

    assert report["rows_written"] == 0
    assert report["sources"][0]["skipped_domains"] == {"special_token_leak": 1}


def test_build_sample_dataset_scans_domains_beyond_sample_quota(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(source_dir, domain_id="0")
    make_domain(source_dir, domain_id="1")
    broken_tool = default_tool()
    broken_tool.pop("returns")
    make_domain(source_dir, domain_id="2", tools=[broken_tool])
    output_path = tmp_path / "sample.jsonl"
    report_path = tmp_path / "sample.report.json"

    report = build_sample_dataset(
        [source_dir],
        output_path,
        report_path,
        max_rows=1,
        scan_domains_per_source=3,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert report["sources"][0]["domains_seen"] == 3
    assert report["sources"][0]["skipped_domains"] == {"missing_returns": 1}
    assert report["sources"][0]["skip_examples"][0]["domain_dir_name"] == "2"
    assert "path" not in report["sources"][0]["skip_examples"][0]


def test_build_sample_dataset_supports_unbounded_full_generation(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    make_domain(source_dir, domain_id="0", scenarios=[default_scenario(), default_scenario()])
    make_domain(source_dir, domain_id="1", scenarios=[default_scenario()])
    output_path = tmp_path / "full.jsonl"
    report_path = tmp_path / "full.report.json"

    report = build_sample_dataset(
        [source_dir],
        output_path,
        report_path,
        max_rows=None,
        max_rows_per_domain=None,
        scan_domains_per_source=None,
        dataset_name="full_test",
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert report["rows_written"] == 3
    assert report["metadata"]["dataset_name"] == "full_test"
    assert [row["metadata"]["scenario_index"] for row in rows] == [0, 1, 0]


def test_build_sample_dataset_reads_optional_seed_manifest_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    source_dir = run_dir / "domains"
    domain_dir = make_domain(source_dir)
    (domain_dir / "domain.json").write_text(
        json.dumps(
            {
                "domain_id": "domain-123",
                "normalized_name": "order support",
                "generation_profile": "proactive",
                "request_index": 2,
                "candidate_index": 4,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "1",
                "run_id": "run-123",
                "run_name": "manifest-test",
                "generation_profile": "proactive",
                "random_seed": 7,
                "config": {
                    "domain_model": {"model": "domain-model"},
                    "policy_tools_model": {"model": "policy-model"},
                    "scenario_model": {"model": "scenario-model"},
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "sample.jsonl"
    build_sample_dataset(
        [source_dir],
        output_path,
        tmp_path / "sample.report.json",
        max_rows=1,
    )

    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["metadata"]["seed_run_id"] == "run-123"
    assert row["metadata"]["domain_generator_model"] == "domain-model"
    assert row["metadata"]["policy_tools_model"] == "policy-model"
    assert row["metadata"]["scenario_generator_model"] == "scenario-model"
    assert row["source_artifacts"]["domain_id"] == "domain-123"
