# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sqlite3
from copy import deepcopy
from unittest.mock import patch

import pytest

from resources_servers.deepcad.dataset import lazy_inputs, training
from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected


def test_deployment_overrides_preserve_input_and_rendering_settings():
    examples = [
        {
            "deepcad_lazy": {
                "config": {
                    "renderer_python": "old-python",
                    "cache_root": "old-cache",
                    "seed": 42,
                    "diagonal_view_probability": 0.9,
                },
                "record": {"asset_id": "original"},
            }
        },
        {"task_source": "ordinary"},
    ]
    original = deepcopy(examples)
    settings = {
        "lazy_render_overrides": {
            "renderer_python": "node-python",
            "cache_root": "/tmp/cache",
            "reference_catalog": "/tmp/catalog.sqlite",
            "reference_catalog_root": "/tmp",
        }
    }
    with patch.object(training, "materialize_cad_examples", side_effect=lambda selected: selected):
        result = training.materialize_examples(examples, config=settings)
    assert examples == original
    assert result[1] == original[1]
    effective = result[0]["deepcad_lazy"]["config"]
    assert effective == {**original[0]["deepcad_lazy"]["config"], **settings["lazy_render_overrides"]}
    with patch.object(training, "materialize_cad_examples", side_effect=RuntimeError("render failed")):
        with pytest.raises(RuntimeError, match="render failed"):
            training.materialize_examples(examples, config=settings)
    assert examples == original


def test_deployment_override_typos_fail():
    with pytest.raises(ValueError, match="cache_rooot"):
        training.materialize_examples([], config={"lazy_render_overrides": {"cache_rooot": "/tmp"}})


@pytest.mark.parametrize("explicit_root,allowed", [(False, False), (True, True)])
def test_replacement_catalog_confined_to_explicit_root(tmp_path, explicit_root, allowed):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    local = tmp_path / "node-local"
    local.mkdir()
    catalog = local / "catalog.sqlite"
    record = {
        "source_kind": "cad_corpus",
        "id": "requested",
        "asset_id": "requested",
        "task_id": "requested-task",
        "dataset": "cadevolve",
        "reference_operations": ["extrude"],
    }
    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(dataset_root),
        renderer_python="python",
        cache_root=str(local / "cache"),
        reference_catalog=str(catalog),
        reference_catalog_root=str(local) if explicit_root else None,
    )
    example = {
        "task_source": "deepcad_simple_agent",
        "responses_create_params": {"input": []},
        "deepcad_lazy": {"config": config.model_dump(), "record": record},
    }
    replacement = deepcopy(example)
    replacement["deepcad_lazy"]["record"].update(id="replacement", asset_id="replacement")
    with sqlite3.connect(catalog) as db:
        db.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, status TEXT, row_json TEXT)")
        db.execute("INSERT INTO assets VALUES (?,?,?)", ("replacement", "selected", json.dumps(replacement)))
    selected = []

    def render(row):
        selected.append(deepcopy(row))
        if row["deepcad_lazy"]["record"]["asset_id"] == "requested":
            raise ReferenceRejected("invalid requested asset")
        return {"verifier_metadata": {}}

    with patch.object(lazy_inputs, "_materialize_one", side_effect=render):
        if allowed:
            result = lazy_inputs.materialize_one(example)
            assert result["verifier_metadata"]["replaced_reference_asset_ids"] == ["requested"]
            assert selected[-1]["deepcad_lazy"]["config"] == config.model_dump()
        else:
            with pytest.raises(ValueError, match="catalog must remain inside"):
                lazy_inputs.materialize_one(example)


def test_explicit_catalog_root_rejects_escape(tmp_path):
    # The explicit root narrows catalog access; it does not disable confinement.
    root = tmp_path / "allowed"
    root.mkdir()
    config = lazy_inputs.LazyRenderConfig(
        dataset_root=str(tmp_path),
        renderer_python="python",
        cache_root=str(root / "cache"),
        reference_catalog=str(tmp_path / "outside.sqlite"),
        reference_catalog_root=str(root),
    )
    row = {
        "deepcad_lazy": {
            "config": config.model_dump(),
            "record": {"source_kind": "cad_corpus", "asset_id": "bad", "task_id": "bad"},
        }
    }
    with patch.object(lazy_inputs, "_materialize_one", side_effect=ReferenceRejected("invalid")):
        with pytest.raises(ValueError, match="catalog must remain inside"):
            lazy_inputs.materialize_one(row)
