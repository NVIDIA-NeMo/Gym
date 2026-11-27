# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from resources_servers.deepcad.dataset import lazy_inputs as cache
from resources_servers.deepcad.dataset import sampled_lazy as module
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)


class SampledLazyDeploymentTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.record = dict(
            id="asset",
            task_id="task",
            source_sha256="source",
            reference_step="references/asset.step",
            difficulty=0,
            construction_metrics={},
        )
        self.example = dict(
            task_source="deepcad_simple_agent",
            responses_create_params={"input": []},
            deepcad_lazy=dict(
                record=self.record,
                config=dict(
                    dataset_root=str(self.root),
                    renderer_python="python",
                    cache_root=str(self.root / "cache"),
                    warm_start_cache=True,
                ),
            ),
        )
        self.row = dict(
            responses_create_params={
                "input": [
                    dict(
                        role="user",
                        content=[dict(type="input_image", image_url="data:image/png;base64,test", detail="high")],
                    )
                ]
            },
            verifier_metadata=dict(
                task_id="task", source_sha256="source", reference_step="asset.step", difficulty=0, view_mode="diagonal"
            ),
        )

        reference = self.root / self.record["reference_step"]
        reference.parent.mkdir()
        reference.write_bytes(b"reference STEP")
        self.record["reference_step_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()

    def worker(self, args, **kwargs):
        work = Path(args[-1]).parent
        self.assertFalse(work.is_relative_to(self.root / "cache"))
        Path(args[-1]).write_text(json.dumps(self.row))
        return SimpleNamespace(returncode=0)

    def test_sharded_cache_scratch_and_concurrent_hit(self):
        with patch.object(cache.subprocess, "run", side_effect=self.worker) as render:
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(cache.materialize_one, [deepcopy(self.example) for _ in range(4)]))
            self.assertEqual(render.call_count, 1)
        target = cache.cache_target(self.example["deepcad_lazy"])
        self.assertEqual(target.parent.name, target.stem[:2])
        self.assertTrue(target.is_file())
        with patch.object(cache.subprocess, "run", side_effect=AssertionError("rendered on a hit")):
            self.assertEqual(cache.materialize_one(self.example), rows[0])
        target.write_text("invalid json")
        with self.assertRaises(json.JSONDecodeError):
            cache.materialize_one(self.example)

    def test_persistent_step_missing_and_corrupt_fail(self):
        config = cache.LazyRenderConfig.model_validate(self.example["deepcad_lazy"]["config"])
        record = {**self.record, "source_kind": "cad_corpus"}
        step = self.root / "references/asset.step"
        step.parent.mkdir(exist_ok=True)
        step.write_bytes(b"valid-step")
        step.with_suffix(".json").write_text(
            json.dumps({"step_sha256": hashlib.sha256(step.read_bytes()).hexdigest()})
        )
        cache._validate_reference(config, record)
        step.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            cache._validate_reference(config, record)
        step.unlink()
        with self.assertRaises(FileNotFoundError):
            cache._validate_reference(config, record)

    def test_replacement_choice_pinned_without_catalog_rescan(self):
        from resources_servers.deepcad.dataset.corpus_inputs import ReferenceRejected

        original = deepcopy(self.example)
        original["deepcad_lazy"]["record"].update(
            source_kind="cad_corpus", asset_id="bad", dataset="a", reference_operations=[]
        )
        catalog = self.root / "catalog.sqlite"
        original["deepcad_lazy"]["config"]["reference_catalog"] = str(catalog)
        replacement = deepcopy(original)
        replacement["deepcad_lazy"]["record"].update(asset_id="good", task_id="good-task")
        with sqlite3.connect(catalog) as db:
            db.execute("CREATE TABLE assets(id TEXT,status TEXT,row_json TEXT)")
            db.execute("INSERT INTO assets VALUES('good','selected',?)", (json.dumps(replacement),))
        seen = []

        def resolve(example, *, cache_only=False):
            record = example["deepcad_lazy"]["record"]
            seen.append(record["asset_id"])
            if record["asset_id"] == "bad":
                raise ReferenceRejected("bad geometry")
            result = deepcopy(self.row)
            result["verifier_metadata"]["task_id"] = record["task_id"]
            return result

        with patch.object(cache, "_materialize_one", side_effect=resolve):
            first = cache.materialize_one(original)
            with patch.object(cache.sqlite3, "connect", side_effect=AssertionError("rescanned catalog")):
                second = cache.materialize_one(original)
        self.assertEqual(first, second)
        self.assertEqual(seen, ["bad", "good", "good"])

    def test_export_matches_native_draw_order_and_multiplicity(self):
        manifest = self.root / "manifest.jsonl"
        catalog = self.root / "catalog.sqlite"
        catalog.write_bytes(b"fixed catalog")
        rows = []
        for index in range(8):
            row = deepcopy(self.example)
            row["deepcad_lazy"]["record"].update(
                id=str(index), task_id=str(index), dataset="a" if index % 2 else "b", reference_operations=["box"]
            )
            row["deepcad_lazy"]["config"]["reference_catalog"] = str(catalog)
            rows.append(row)
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        sampler = self.root / "sampler.json"
        sampler.write_text(json.dumps(dict(sources=["a", "b"], uniform_sources=[], seed=42)))
        overrides = self.root / "overrides.json"
        overrides.write_text(json.dumps({"workers": 1}))
        output = self.root / "export"
        with patch.object(module, "materialize_one", side_effect=AssertionError("render during export")):
            module.export(manifest, output, sampler, overrides, 64, 8)
        selected = [json.loads(line) for line in (output / "draws.jsonl").read_text().splitlines()]
        config = SourceOperationSamplingConfig(sources=["a", "b"], uniform_sources=[], seed=42, samples_per_epoch=64)
        native = list(SourceOperationSampler(manifest.read_text().splitlines(keepends=True), config, 8))
        self.assertEqual([row["deepcad_lazy"]["record"]["id"] for row in selected], [str(i) for i in native])
        self.assertEqual(np.fromfile(output / "source-indices.i64", dtype="<i8").tolist(), native)
        self.assertLess(len(set(native)), len(native))
        self.assertTrue(all(row["deepcad_lazy"]["config"]["workers"] == 1 for row in selected))
        info = json.loads((output / "draws-manifest.json").read_text())
        self.assertTrue(info["ordered"])
        self.assertEqual(info["rendered_during_export"], 0)
        self.assertEqual(info["reference_catalog_sha256"], module.sha256(catalog))
        self.assertEqual(info["overrides_sha256"], module.sha256(overrides))
        self.assertEqual(info["source_indices_sha256"], module.sha256(output / "source-indices.i64"))
        for name, digest in info["gym_sources"].items():
            self.assertEqual(digest, module.sha256(Path(module.__file__).with_name(name)))
        with self.assertRaises(FileExistsError):
            module.export(manifest, output, sampler, overrides, 64, 8)
        info["gym_sources"]["sampled_lazy.py"] = "0" * 64
        (output / "draws-manifest.json").write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "source changed"):
            list(module.read_range(output, 0, 1))

    def test_verify_refuses_render_on_cold_cache(self):
        output = self.root / "export"
        output.mkdir()
        (output / "draws.jsonl").write_text(json.dumps(self.example) + "\n")
        (output / "draws-manifest.json").write_text(
            json.dumps({"draw_count": 1, "draws_sha256": module.sha256(output / "draws.jsonl")})
        )
        with patch.object(cache.subprocess, "run", side_effect=AssertionError("attempted a render")):
            with self.assertRaisesRegex(FileNotFoundError, "Prompt cache entry missing"):
                module.verify(output, 0, 1)
