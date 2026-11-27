# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from resources_servers.deepcad.dataset.lazy_inputs import (
    LazyRenderConfig,
    main,
    materialize_examples,
    materialize_one,
)


class LazyInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = dict(
            id="0000_00000007",
            task_id="0000/00000007",
            source_sha256="source",
            cad_json="cad_json/0000/00000007.json",
            reference_step="references/0000_00000007.step",
            reference_step_sha256="step",
            difficulty=5,
            construction_metrics={},
        )
        self.config = LazyRenderConfig(
            dataset_root=str(self.root), renderer_python="python", cache_root=str(self.root / "cache")
        )
        self.example = dict(
            task_source="deepcad_simple_agent",
            responses_create_params={"input": []},
            deepcad_lazy=dict(config=self.config.model_dump(), record=self.record),
        )
        self.row = dict(
            task_source="deepcad_simple_agent",
            responses_create_params={
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/png;base64,test", "detail": "high"}
                        ],
                    }
                ]
            },
            verifier_metadata=dict(
                task_id=self.record["task_id"],
                source_sha256="source",
                reference_step="0000_00000007.step",
                difficulty=5,
                view_mode="diagonal",
            ),
        )

    def worker(self, argv, **kwargs):
        Path(argv[-1]).write_text(json.dumps(self.row))
        return SimpleNamespace(returncode=0)

    def test_concurrent_cold_requests_render_once(self):
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run", side_effect=self.worker) as run:
            with ThreadPoolExecutor(max_workers=8) as pool:
                rows = list(pool.map(materialize_one, [deepcopy(self.example) for _ in range(8)]))
            self.assertEqual(run.call_count, 1)
            self.assertTrue(all(row == rows[0] for row in rows))
            materialize_one(self.example)
            self.assertEqual(run.call_count, 1)

    def test_group_dedup_preserves_transport_and_inputs(self):
        examples = [{**deepcopy(self.example), "_rowidx": i} for i in range(8)]
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run", side_effect=self.worker) as run:
            materialize_examples(examples)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([row["_rowidx"] for row in examples], list(range(8)))
        self.assertTrue(
            all(row["responses_create_params"] == examples[0]["responses_create_params"] for row in examples)
        )
        self.assertTrue(all("deepcad_lazy" not in row for row in examples))
        self.assertIn("deepcad_lazy", self.example)

    def test_lazy_cache_preserves_agent_routing(self):
        self.example["task_source"] = "deepcad_multi_turn_agent"
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run", side_effect=self.worker):
            row = materialize_one(self.example)
        self.assertEqual(row["task_source"], "deepcad_multi_turn_agent")

    def test_prepared_row_never_launches_worker(self):
        legacy = self.root / "rows"
        legacy.mkdir()
        (legacy / (self.record["id"] + ".json")).write_text(json.dumps(self.row))
        self.example["deepcad_lazy"]["config"]["legacy_rows_root"] = str(legacy)
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run") as run:
            resolved = materialize_one(self.example)
        run.assert_not_called()
        self.assertEqual(resolved["responses_create_params"], self.row["responses_create_params"])

    def test_failure_is_not_cached_and_batch_is_unchanged(self):
        examples = [deepcopy(self.example)]
        with patch(
            "resources_servers.deepcad.dataset.lazy_inputs.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stderr="mesh failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mesh failed"):
                materialize_examples(examples)
        self.assertEqual(examples[0], self.example)
        self.assertEqual(list((self.root / "cache").glob("*.json")), [])

    def test_manifest_needs_no_cad_files_or_rendering(self):
        (self.root / "easy").mkdir()
        (self.root / "_work").mkdir()
        record = {**self.record, "split": "train", "bucket": "easy", "geometry_status": "ok"}
        (self.root / "easy/train.jsonl").write_text(json.dumps(record) + "\n")
        (self.root / "validation.json").write_text('{"passed": true}')
        (self.root / "_work/features.jsonl").write_text(
            json.dumps({"id": record["id"], "status": "ok", "step_sha256": "step"}) + "\n"
        )
        output = self.root / "train.lazy.jsonl"
        args = [
            "lazy_inputs.py",
            "manifest",
            str(self.root),
            str(output),
            "--runtime-root",
            "/data/deepcad-dedup",
            "--renderer-python",
            "python",
            "--cache-root",
            "/tmp/deepcad-lazy",
        ]
        with patch("sys.argv", args), patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run") as run:
            main()
        run.assert_not_called()
        row = json.loads(output.read_text())
        self.assertEqual(row["responses_create_params"]["input"], [])
        self.assertEqual(row["deepcad_lazy"]["record"]["source_sha256"], "source")

    def test_ordinary_rows_are_unchanged(self):
        examples = [{"task_source": "other", "responses_create_params": {"input": []}}]
        original = deepcopy(examples)
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run") as run:
            materialize_examples(examples)
        run.assert_not_called()
        self.assertEqual(examples, original)

    def test_reference_replacement_preserves_opencode_routing(self):
        from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected
        from resources_servers.deepcad.dataset.lazy_inputs import ReferenceRenderTimeout

        catalog = self.root / "references.sqlite"
        replacement = deepcopy(self.example)
        replacement["agent_ref"] = {"name": "deepcad_simple_agent"}
        replacement["deepcad_lazy"]["record"].update(asset_id="replacement", task_id="replacement-task")
        with sqlite3.connect(catalog) as db:
            db.execute("CREATE TABLE assets (id TEXT, status TEXT, row_json TEXT)")
            db.execute("INSERT INTO assets VALUES (?, ?, ?)", ("replacement", "selected", json.dumps(replacement)))
        self.example["task_source"] = "deepcad_opencode_agent"
        self.example["deepcad_lazy"]["config"]["reference_catalog"] = str(catalog)
        self.example["deepcad_lazy"]["record"].update(source_kind="cad_corpus", asset_id="bad")

        for failure in (ReferenceRejected("invalid"), ReferenceRenderTimeout("timeout")):
            for agent_ref in (None, {"name": "deepcad_opencode_agent"}):
                with self.subTest(failure=type(failure).__name__, agent_ref=agent_ref):
                    example = deepcopy(self.example)
                    example["deepcad_lazy"]["config"]["cache_root"] = str(
                        self.root / f"cache-{type(failure).__name__}-{agent_ref is None}"
                    )
                    if agent_ref is not None:
                        example["agent_ref"] = agent_ref
                    seen = []

                    def render(current):
                        seen.append(deepcopy(current))
                        if len(seen) == 1:
                            raise failure
                        resolved = deepcopy(current)
                        resolved.pop("deepcad_lazy")
                        resolved["verifier_metadata"] = {"task_id": "replacement-task"}
                        return resolved

                    with patch("resources_servers.deepcad.dataset.lazy_inputs._materialize_one", side_effect=render):
                        row = materialize_one(example)
                    self.assertEqual(seen[1]["task_source"], "deepcad_opencode_agent")
                    self.assertEqual(row["task_source"], "deepcad_opencode_agent")
                    self.assertEqual(row.get("agent_ref"), agent_ref)
                    self.assertEqual(row["verifier_metadata"]["task_id"], "replacement-task")
                    self.assertEqual(row["verifier_metadata"]["replaced_reference_asset_ids"], ["bad"])

    def test_shared_geometry_keeps_each_callers_route(self):
        examples = [deepcopy(self.example), deepcopy(self.example)]
        examples[1]["task_source"] = "deepcad_opencode_agent"
        with patch("resources_servers.deepcad.dataset.lazy_inputs.subprocess.run", side_effect=self.worker) as run:
            materialize_examples(examples)
        self.assertEqual(run.call_count, 1)
        self.assertEqual([row["task_source"] for row in examples], ["deepcad_simple_agent", "deepcad_opencode_agent"])
