# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate partner_if rows before data tooling hands them to an agent or verifier."""

import pytest
from pydantic import ValidationError

from resources_servers.partner_if.task_data import TaskData


def _item(identifier: str, *, dimension: str = "if", depends_on: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "id": identifier,
        "constraint": "Preserve the required evidence.",
        "verifier": {"id": "text_contains_v1", "params": {"source": "answer", "literal": "verified"}},
        "dimension": dimension,
        "depends_on": list(depends_on),
    }


def _row(**contract_updates: object) -> dict[str, object]:
    contract = {
        "task_id": "synthetic-contract-validation",
        "origin": "synthetic",
        "source": "authored core test",
        "rubric_version": "test-v1",
        "audit_signoff": True,
        "data_kind": "synthetic_smoke",
        "rubrics": [_item("format")],
    }
    return {"partner_if": contract | contract_updates}


def _bundle() -> dict[str, object]:
    return {
        "entrypoint": "tools:Backend",
        "files": {"tools.py": "class Backend: pass\n", "fixtures/case.json": "{}"},
        "fixture": "fixtures/case.json",
        "requirements": ["jsonschema==4.26.0"],
    }


def test_shared_prerequisites_preserve_if_and_completion_contracts() -> None:
    row = TaskData.model_validate(
        _row(
            evidence_mode="live_agent",
            actor_system_prompt="Use the recorded evidence.",
            tool_description_additions={"ipython": "Keep the delivery receipt."},
            prerequisites=[_item("seeded")],
            rubrics=[
                _item("format", depends_on=("seeded",)),
                _item("grounding", depends_on=("seeded",)),
                _item("delivery", dimension="completion", depends_on=("grounding",)),
            ],
        )
    )
    assert [(item.id, item.dimension, item.depends_on) for item in row.partner_if.checks] == [
        ("seeded", "if", []),
        ("format", "if", ["seeded"]),
        ("grounding", "if", ["seeded"]),
        ("delivery", "completion", ["grounding"]),
    ]
    assert row.partner_if.actor_system_prompt == "Use the recorded evidence."
    assert row.partner_if.tool_description_additions == {"ipython": "Keep the delivery receipt."}


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"rubrics": [_item("same"), _item("same")]}, "Duplicate rubric IDs"),
        ({"prerequisites": [_item("format")]}, "Duplicate rubric IDs"),
        ({"rubrics": [_item("delivery", dimension="completion")]}, "must have an IF dimension"),
        ({"rubrics": [_item("format", depends_on=("missing",))]}, "Unknown prerequisite"),
        ({"rubrics": [_item("format", depends_on=("format",))]}, "Cyclic rubric dependencies"),
        (
            {"rubrics": [_item("a", depends_on=("b",)), _item("b", depends_on=("a",))]},
            "Cyclic rubric dependencies",
        ),
        (
            {"prerequisites": [_item("seeded")], "rubrics": [_item("format", depends_on=("seeded", "seeded"))]},
            "Duplicate prerequisites",
        ),
        (
            {"rubrics": [_item("delivery", dimension="completion"), _item("format", depends_on=("delivery",))]},
            "IF requirement cannot depend on a completion requirement",
        ),
        ({"actor_system_prompt": "Use evidence."}, "describe a live agent's harness"),
        ({"tool_description_additions": {"ipython": "Keep receipts."}}, "describe a live agent's harness"),
    ],
)
def test_invalid_grading_contract_is_rejected_at_the_task_boundary(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message) as error:
        TaskData.model_validate(_row(**updates))
    assert error.value.errors()[0]["loc"] == ("partner_if",)


@pytest.mark.parametrize("module_path", ["tools.py", "tools/__init__.py"])
def test_self_contained_module_and_package_backends_are_accepted(module_path: str) -> None:
    bundle = _bundle() | {"files": {module_path: "class Backend: pass\n", "fixtures/case.json": "{}"}}
    row = TaskData.model_validate(_row() | {"tool_backend": bundle})
    assert row.tool_backend is not None
    assert row.tool_backend.entrypoint == "tools:Backend"
    assert row.tool_backend.files == bundle["files"]
    assert row.tool_backend.fixture == "fixtures/case.json"
    assert row.tool_backend.requirements == ["jsonschema==4.26.0"]


@pytest.mark.parametrize(
    "path",
    ["../escape.py", "/absolute.py", "./extra.py", "nested//extra.py", "nested/../extra.py", "nested\\extra.py"],
)
def test_backend_files_cannot_escape_or_alias_the_bundle_root(path: str) -> None:
    bundle = _bundle()
    bundle["files"] = {**bundle["files"], path: "untrusted content"}
    with pytest.raises(ValidationError, match="normalized relative paths") as error:
        TaskData.model_validate(_row() | {"tool_backend": bundle})
    assert error.value.errors()[0]["loc"] == ("tool_backend",)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"fixture": "missing.json"}, "fixture 'missing.json' is not one of"),
        ({"entrypoint": "missing:Backend"}, "entrypoint module 'missing' is not one of"),
        ({"requirements": ["jsonschema>=4"]}, "requirements"),
    ],
)
def test_backend_bundle_requires_its_fixture_module_and_exact_dependency_pins(
    updates: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message) as error:
        TaskData.model_validate(_row() | {"tool_backend": _bundle() | updates})
    assert error.value.errors()[0]["loc"][0] == "tool_backend"
