# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from copy import deepcopy
from unittest.mock import patch

import pytest

from nemo_gym.data import load_dataset_hooks
from resources_servers.deepcad.dataset import lazy_inputs
from resources_servers.deepcad.tests import test_lazy_inputs as fixtures


def test_hooks_load_without_cad_or_training_framework():
    code = (
        "import sys; from nemo_gym.data import load_dataset_hooks; "
        "load_dataset_hooks('resources_servers.deepcad.dataset.training'); "
        "assert 'cadquery' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hook_loader_rejects_missing_exports_and_invalid_names():
    for name in ("", "../app", "resources_servers.deepcad.operations"):
        with pytest.raises(ValueError):
            load_dataset_hooks(name)


def test_group_dedup_preserves_each_callers_sampling_fields():
    fixture = fixtures.LazyInputTests()
    fixture.setUp()
    try:
        rows = [deepcopy(fixture.example), deepcopy(fixture.example)]
        for index, row in enumerate(rows):
            row["responses_create_params"].update(temperature=index / 2, max_output_tokens=32768 + index)
        with patch.object(lazy_inputs.subprocess, "run", side_effect=fixture.worker) as render:
            load_dataset_hooks("resources_servers.deepcad.dataset.training").materialize_examples(rows, config={})
        assert render.call_count == 1
        assert [r["responses_create_params"]["temperature"] for r in rows] == [0, 0.5]
        assert [r["responses_create_params"]["max_output_tokens"] for r in rows] == [32768, 32769]
        assert rows[0]["responses_create_params"]["input"] == rows[1]["responses_create_params"]["input"]
    finally:
        fixture.doCleanups()


def test_replacement_preserves_selected_source_and_operations(tmp_path, monkeypatch):
    import sqlite3

    from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected

    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(tmp_path),
        renderer_python="python",
        cache_root=str(tmp_path / "cache"),
        reference_catalog=str(tmp_path / "catalog.sqlite"),
    )

    def row(asset, source, operations):
        return {
            "task_source": "deepcad",
            "deepcad_lazy": {
                "config": config.model_dump(),
                "record": {
                    "id": asset,
                    "asset_id": asset,
                    "task_id": asset,
                    "source_kind": "cad_corpus",
                    "dataset": source,
                    "reference_operations": operations,
                },
            },
        }

    original = row("bad", "a", ["hole"])
    with sqlite3.connect(config.reference_catalog) as db:
        db.execute("CREATE TABLE assets (id TEXT, status TEXT, row_json TEXT)")
        for candidate in [
            row("0-other-source", "b", ["hole"]),
            row("1-missing-op", "a", ["box"]),
            row("2-good", "a", ["box", "hole"]),
        ]:
            db.execute(
                "INSERT INTO assets VALUES (?, 'selected', ?)",
                (candidate["deepcad_lazy"]["record"]["id"], json.dumps(candidate)),
            )
    seen = []

    def render(example):
        record = example["deepcad_lazy"]["record"]
        seen.append(record)
        if record["asset_id"] == "bad":
            raise ReferenceRejected("bad geometry")
        return {"verifier_metadata": {"task_id": record["task_id"]}}

    monkeypatch.setattr(lazy_inputs, "_materialize_one", render)
    result = lazy_inputs.materialize_one(original)
    assert seen[1]["dataset"] == "a" and "hole" in seen[1]["reference_operations"]
    assert result["verifier_metadata"]["task_id"] == "2-good"


def test_real_lazy_render_is_shared_and_warm_cache_matches(tmp_path, monkeypatch):
    import hashlib

    import cadquery as cq

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "sequence": [],
                "entities": {},
                "properties": {
                    "bounding_box": {
                        "min_point": {"x": -75, "y": -75, "z": -75},
                        "max_point": {"x": 75, "y": 75, "z": 75},
                    }
                },
            }
        )
    )
    step = tmp_path / "reference.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(step))
    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(tmp_path),
        renderer_python=sys.executable,
        cache_root=str(tmp_path / "cache"),
        diagonal_view_probability=0,
        no_annotations_probability=1,
        harness="opencode",
    )
    example = {
        "task_source": "deepcad_opencode",
        "responses_create_params": {"input": [], "max_output_tokens": 32768},
        "deepcad_lazy": {
            "config": config.model_dump(),
            "record": {
                "id": "synthetic",
                "task_id": "synthetic",
                "cad_json": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "reference_step": step.name,
                "reference_step_sha256": hashlib.sha256(step.read_bytes()).hexdigest(),
                "difficulty": 0,
                "construction_metrics": {},
            },
        },
    }
    monkeypatch.delenv("PYTHONPATH", raising=False)
    cold = [deepcopy(example) | {"_rowidx": i} for i in range(16)]
    with patch.object(lazy_inputs.subprocess, "run", wraps=lazy_inputs.subprocess.run) as render:
        lazy_inputs.materialize_examples(cold)
        warm = [deepcopy(example)]
        lazy_inputs.materialize_examples(warm)
    assert render.call_count == 1
    assert [row["_rowidx"] for row in cold] == list(range(16))
    assert all(row["responses_create_params"] == warm[0]["responses_create_params"] for row in cold)
    assert len(cold[0]["target_images"]) == 3
    assert cold[0]["verifier_metadata"]["source_sha256"] == example["deepcad_lazy"]["record"]["source_sha256"]
    assert "model.py" in cold[0]["responses_create_params"]["input"][0]["content"]
    assert "deepcad_lazy" in example


@pytest.mark.parametrize("flag", ["api_cheatsheet", "orientation_axes"])
def test_optional_full_branch_prompt_features_are_not_silently_enabled(tmp_path, flag):
    with pytest.raises(ValueError):
        lazy_inputs.LazyRenderConfig(
            dataset_root=str(tmp_path),
            renderer_python=sys.executable,
            cache_root=str(tmp_path / "cache"),
            **{flag: True},
        )
