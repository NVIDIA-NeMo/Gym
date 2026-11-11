# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same schema validates source rows and collated tasks without exposing labels."""

import ast
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from nemo_gym.data import TaskDataValidator
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request, server  # noqa: F401
from resources_servers.opencode_multimodal_benchmarks.verifier import VisualVerifyRequest


BENCHMARKS = ("VStarBench", "HRBench8K", "VTCBench", "ZeroBench")


def _minimal_metadata(benchmark):
    metadata = {
        "benchmark": benchmark,
        "index": "1",
        "question": "Which animal?",
        "answer": "cat",
        "expected_samples": 1,
    }
    if benchmark in {"VStarBench", "HRBench8K"}:
        metadata.update(answer="A", options={"A": "cat", "B": "dog"})
    if benchmark == "HRBench8K":
        metadata["cycle_category"] = "1"
    if benchmark == "AgentX":
        metadata.update(
            GT_json='{"value":"cat"}', query_type="objective", query_type_raw="Objective", modality="Image"
        )
    return metadata


def _wire_request(metadata):
    value = request(metadata["benchmark"], "cat").model_dump()
    value["verifier_metadata"] = metadata
    return VisualVerifyRequest.model_validate(value)


def _assert_declared_task_reads(source, names, *, derived=()):
    """Guard literal metadata reads; minimal-task execution below covers dynamic reads."""
    reads = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in names:
            key = node.slice
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in names
            and node.func.attr == "get"
            and node.args
        ):
            key = node.args[0]
        else:
            continue
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            reads.add(key.value)
    missing = reads - TaskData.model_fields.keys() - set(derived)
    assert not missing, f"Consumer reads undeclared task fields: {sorted(missing)}"


@pytest.mark.parametrize(
    "filename,names,derived",
    [
        ("verifier.py", {"task"}, ()),
        ("mcq.py", {"task"}, ()),
        ("vtcbench.py", {"task"}, ()),
        ("agentx.py", {"task"}, ()),
        ("metrics.py", {"metadata"}, ()),
        # These rows also carry explicitly derived scoring/report fields.
        ("reporting.py", {"row"}, ("hit", "type", "accuracy")),
        ("app.py", {"row"}, ("verifier_metadata", "reward", "hit", "accuracy", "cycle", "type")),
    ],
)
def test_consumer_metadata_reads_are_declared(filename, names, derived):
    source = (Path(__file__).resolve().parents[1] / filename).read_text()
    _assert_declared_task_reads(source, names, derived=derived)


def test_consumer_guard_detects_an_added_undeclared_field():
    with pytest.raises(AssertionError, match="undeclared_reference"):
        _assert_declared_task_reads('def score(task): return task["undeclared_reference"]', {"task"})


def test_consumer_guard_detects_a_removed_schema_field(monkeypatch):
    monkeypatch.delitem(TaskData.model_fields, "answer")
    source = (Path(__file__).resolve().parents[1] / "vtcbench.py").read_text()
    with pytest.raises(AssertionError, match="answer"):
        _assert_declared_task_reads(source, {"task"})


@pytest.mark.parametrize("benchmark", [*BENCHMARKS, "AgentX"])
@pytest.mark.parametrize("field", ["index", "question", "answer", "expected_samples"])
def test_wire_boundary_requires_shared_consumer_fields(benchmark, field):
    metadata = _minimal_metadata(benchmark)
    metadata.pop(field)
    with pytest.raises(ValidationError):
        _wire_request(metadata)


@pytest.mark.parametrize("benchmark", [*BENCHMARKS, "AgentX"])
@pytest.mark.parametrize(
    "field,value", [("answer", ""), ("answer", " \n"), ("expected_samples", 0), ("expected_samples", -1)]
)
def test_wire_boundary_rejects_unusable_consumer_values(benchmark, field, value):
    with pytest.raises(ValidationError):
        _wire_request(_minimal_metadata(benchmark) | {field: value})


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K"])
@pytest.mark.parametrize("options", [None, {}, {"A": " "}])
def test_wire_boundary_requires_usable_mcq_options(benchmark, options):
    metadata = _minimal_metadata(benchmark)
    metadata.pop("options")
    if options is not None:
        metadata["options"] = options
    with pytest.raises(ValidationError):
        _wire_request(metadata)


@pytest.mark.parametrize(
    "answer,options",
    [
        ("cat", {"A": "cat", "B": "CAT"}),
        ("missing", {"A": "cat", "B": "dog"}),
        ("C", {"A": "cat", "B": "dog"}),
        ("A, A", {"A": "cat", "B": "dog"}),
        ("A", {"A": " "}),
        ("H", {"H": "cat"}),
    ],
)
def test_vtc_reference_is_validated_before_scoring(answer, options):
    with pytest.raises(ValidationError, match="VTCBench"):
        _wire_request(_minimal_metadata("VTCBench") | {"answer": answer, "options": options})


@pytest.mark.parametrize("cycle", [None, "", " \n"])
def test_wire_boundary_requires_hrbench_cycle(cycle):
    metadata = _minimal_metadata("HRBench8K")
    metadata.pop("cycle_category")
    if cycle is not None:
        metadata["cycle_category"] = cycle
    with pytest.raises(ValidationError):
        _wire_request(metadata)


@pytest.mark.parametrize("field", ["query_type", "modality"])
@pytest.mark.parametrize("value", [None, "", " \n"])
def test_agentx_wire_boundary_requires_judge_and_reporting_metadata(field, value):
    metadata = _minimal_metadata("AgentX")
    metadata.pop(field)
    if value is not None:
        metadata[field] = value
    with pytest.raises(ValidationError):
        _wire_request(metadata)


@pytest.mark.asyncio
async def test_minimal_agentx_wire_task_runs_verification_and_reporting(server, monkeypatch):
    from unittest.mock import AsyncMock

    from resources_servers.opencode_multimodal_benchmarks.agentx import AgentXConfig, AgentXServer

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    agentx = AgentXServer(
        config=AgentXConfig(**(server.config.model_dump() | {"entrypoint": "agentx.py"})),
        server_client=server.server_client,
    )
    monkeypatch.setattr(agentx, "_judge_completion", AsyncMock(return_value='{"verdict": "equivalent"}'))
    body = _wire_request(_minimal_metadata("AgentX"))
    result = await agentx.verify(body)
    assert result.reward == 1
    metrics = agentx.compute_metrics([[result.model_dump()]])
    assert metrics["AgentX/complete"] is True
    assert metrics["AgentX/Image/completed"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", BENCHMARKS)
async def test_minimal_wire_task_runs_verification_and_reporting(server, benchmark):
    body = _wire_request(_minimal_metadata(benchmark))
    if benchmark == "VTCBench":
        assert body.verifier_metadata["options"] == {}
    # Avoid a judge: each benchmark receives an unambiguous correct prediction.
    prediction = "A" if benchmark in {"VStarBench", "HRBench8K"} else "cat"
    if benchmark == "ZeroBench":
        prediction = "{cat}"
    value = request(benchmark, prediction).model_dump()
    value["verifier_metadata"] = body.verifier_metadata
    result = await server.verify(VisualVerifyRequest.model_validate(value))
    assert result.reward == 1
    repeats = 5 if benchmark == "ZeroBench" else 1
    metrics = server.compute_metrics([[result.model_dump() | {"_ng_rollout_index": i} for i in range(repeats)]])
    assert metrics[f"{benchmark}/complete"] is True
    assert metrics[f"{benchmark}/expected"] == 1


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_aggregation_does_not_infer_completeness_from_landed_rows(server, benchmark):
    metadata = _wire_request(_minimal_metadata(benchmark)).verifier_metadata
    repeats = 5 if benchmark == "ZeroBench" else 1

    def rows(value):
        return [[{"verifier_metadata": value, "reward": 1, "_ng_rollout_index": i} for i in range(repeats)]]

    missing_count = dict(metadata)
    missing_count.pop("expected_samples")
    with pytest.raises(ValueError, match="require expected_samples"):
        server.compute_metrics(rows(missing_count))
    metrics = server.compute_metrics(rows(metadata | {"expected_samples": 2}))
    assert metrics[f"{benchmark}/complete"] is False


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K", "VTCBench", "ZeroBench", "AgentX"])
def test_flat_legacy_and_materialized_tasks(benchmark, tmp_path):
    image = tmp_path / "input-0.png"
    image.write_bytes(b"image bytes")
    metadata = {
        "index": "1",
        "benchmark": benchmark,
        "question": "Visible question",
        "answer": "A",
        "options": {"A": "cat"},
        "expected_samples": 1,
        "cycle_category": "1",
        "GT_json": '{"value":"a"}',
        "query_type": "objective",
        "query_type_raw": "Objective",
        "modality": "Image",
    }
    row = task_row(metadata | {"prompt": "Visible question", "images": [str(image)]})
    adapter = TypeAdapter(TaskData)
    assert adapter.validate_python(metadata).answer == "A"
    validator = TaskDataValidator("opencode_multimodal_benchmarks", adapter, "synthetic.jsonl")
    validator.validate_row(0, row)
    validator.validate_row(1, {"task_id": {"taskset": "visual_tasks", "task_id": "1"}, "task_input": row})
    assert validator.report.clean, validator.report.summary()
    assert "private answer" not in str(row["responses_create_params"])
    with pytest.raises(ValidationError):
        adapter.validate_python(metadata | {"benchmark": "not-a-benchmark"})
    with pytest.raises(ValidationError):
        adapter.validate_python({key: value for key, value in metadata.items() if key != "answer"})


def test_task_schema_retains_unknown_fields_and_json_encoded_references():
    task = TaskData.model_validate(
        {
            "benchmark": "AgentX",
            "index": "1",
            "question": "q",
            "answer": "a",
            "GT_json": '{"value":"a"}',
            "expected_samples": 1,
            "query_type": "objective",
            "query_type_raw": "Objective",
            "modality": "Image",
            "custom_provenance": "fixture",
        }
    )
    assert task.GT_json == '{"value":"a"}'
    assert task.model_dump()["custom_provenance"] == "fixture"


def test_optional_categories_do_not_block_overall_reporting():
    rows = [{"hit": 1.0}, {"hit": 0.0}]
    assert benchmark_report("VStarBench", rows) == [{"split": "none", "Overall": 0.5}]
    assert benchmark_report("VTCBench", rows) == [{"Overall": 50.0}]
    with pytest.raises(ValueError, match="cycle_category"):
        benchmark_report("HRBench8K", rows)


@pytest.mark.parametrize("field", ["answer", "expected_samples", "options"])
def test_mcq_wire_metadata_rejects_missing_required_fields(field):
    from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request
    from resources_servers.opencode_multimodal_benchmarks.verifier import VisualVerifyRequest

    value = request("VStarBench", "A").model_dump()
    value["verifier_metadata"].pop(field)
    with pytest.raises(ValidationError):
        VisualVerifyRequest.model_validate(value)
