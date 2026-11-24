# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import base64
import json
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from resources_servers.deepcad.dataset import lazy_inputs
from resources_servers.deepcad.dataset import materialized_inputs as module
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)


class MaterializedInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        buffer = BytesIO()
        Image.new("RGB", (2, 2), color="white").save(buffer, format="PNG")
        self.image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        (self.root / "references").mkdir()
        self.manifest = self.root / "source.jsonl"
        self.catalog = self.root / "catalog.sqlite"
        self.sampler_config = self.root / "sampler.json"
        self.sampler_config.write_text(json.dumps({"sources": ["a", "b"], "uniform_sources": []}))
        rows = []
        for index in range(16):
            record = dict(
                id=str(index),
                task_id=f"task-{index}",
                source_kind="cad_corpus",
                asset_id=str(index),
                dataset="a" if index % 2 else "b",
                reference_operations=["box"],
                source_sha256=f"source-{index}",
                reference_step=f"references/{index}.step",
                difficulty=1,
            )
            row = dict(
                task_source="deepcad_simple_agent",
                responses_create_params={"input": []},
                deepcad_lazy=dict(
                    record=record,
                    config=dict(
                        dataset_root=str(self.root),
                        renderer_python="python",
                        cache_root=str(self.root / "cache"),
                        reference_catalog=str(self.catalog),
                    ),
                ),
            )
            rows.append(row)
            (self.root / "references" / f"{index}.step").write_text(f"STEP-{index}")
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with sqlite3.connect(self.catalog) as db:
            db.execute("CREATE TABLE assets (id TEXT, status TEXT, row_json TEXT)")
            db.executemany(
                "INSERT INTO assets VALUES (?, 'selected', ?)",
                [(str(i), json.dumps(row)) for i, row in enumerate(rows)],
            )
        self.selection = self.root / "selection"
        self.shards = self.root / "shards"
        self.release = self.root / "release"
        module.select(self.manifest, self.selection, self.sampler_config, 32, 8, 42)

    def materialize(self, example):
        record = example["deepcad_lazy"]["record"]
        return dict(
            task_source=example["task_source"],
            responses_create_params={
                "input": [
                    dict(
                        role="user",
                        content=[
                            dict(
                                type="input_image",
                                image_url=self.image_url,
                                detail="high",
                            )
                        ],
                    )
                ]
            },
            verifier_metadata={
                **{
                    k: record[k]
                    for k in (
                        "task_id",
                        "source_sha256",
                        "difficulty",
                        "asset_id",
                        "dataset",
                    )
                },
                "reference_step": Path(record["reference_step"]).name,
                "view_mode": "diagonal",
            },
        )

    def render_all(self):
        with patch.object(module, "materialize_one", side_effect=self.materialize) as renderer:
            for shard in range(2):
                module.render(self.selection, self.shards, shard, 2, 2, 8)
        return renderer.call_count

    def test_exact_sampler_order_and_occurrences(self):
        config = SourceOperationSamplingConfig(sources=["a", "b"], uniform_sources=[], seed=42, samples_per_epoch=32)
        sampler = SourceOperationSampler(self.manifest.read_text().splitlines(keepends=True), config, 8)
        draws = list(module.read_jsonl(self.selection / "selected.jsonl"))
        self.assertEqual([draw["source_index"] for draw in draws], list(sampler))
        self.assertEqual(len(draws), 32)
        self.assertLess(len({draw["source_index"] for draw in draws}), 32)

    def test_complete_sharded_release_resume_and_relocation(self):
        calls = self.render_all()
        self.assertLess(calls, 32)
        with patch.object(module, "materialize_one") as renderer:
            module.render(self.selection, self.shards, 0, 2, 2, 8)
            renderer.assert_not_called()
        module.assemble(self.selection, self.shards, self.release)
        module.validate(self.release)
        rows = list(module.read_jsonl(self.release / "train.materialized.jsonl"))
        self.assertEqual(len(rows), 32)
        original = list(module.read_jsonl(self.selection / "selected.jsonl"))
        self.assertEqual(
            [row["verifier_metadata"]["task_id"] for row in rows],
            [draw["example"]["deepcad_lazy"]["record"]["task_id"] for draw in original],
        )
        self.assertTrue(all("deepcad_lazy" not in row for row in rows))
        import tarfile

        with tarfile.open(self.release / "references.tar") as archive:
            archive.extractall(self.root / "relocated", filter="data")
        self.assertTrue(
            all((self.root / "relocated" / row["verifier_metadata"]["reference_step"]).is_file() for row in rows)
        )
        with self.assertRaisesRegex(ValueError, "output already exists"):
            module.assemble(self.selection, self.shards, self.release)

    def test_prefix_can_publish_before_other_shards(self):
        with patch.object(module, "materialize_one", side_effect=self.materialize):
            module.render(self.selection, self.shards, 0, 2, 2, 8, max_chunks=1)
        module.assemble(self.selection, self.shards, self.release, prefix_count=8)
        self.assertTrue(json.loads((self.release / "release.json").read_text())["is_smoke_prefix"])
        with self.assertRaisesRegex(ValueError, "Incomplete release"):
            module.assemble(self.selection, self.shards, self.root / "incomplete")

    def test_corrupt_chunk_and_selection_refused(self):
        self.render_all()
        (self.shards / "chunk-000000.jsonl").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            module.render(self.selection, self.shards, 0, 2, 2, 8)
        with self.assertRaisesRegex(ValueError, "Corrupt chunk"):
            module.assemble(self.selection, self.shards, self.release)
        with (self.selection / "selected.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "selection checksum"):
            module.load_selection(self.selection)

    def test_missing_step_prevents_chunk_publication(self):
        for path in (self.root / "references").glob("*.step"):
            path.unlink()
        with patch.object(module, "materialize_one", side_effect=self.materialize):
            with self.assertRaisesRegex(ValueError, "Missing or escaping"):
                module.render(self.selection, self.shards, 0, 2, 2, 8)
        self.assertFalse(list(self.shards.glob("*.receipt.json")))

    def test_prepared_row_rejects_lazy_missing_images_and_path_escape(self):
        example = next(module.read_jsonl(self.selection / "selected.jsonl"))["example"]
        row = self.materialize(example)
        module.validate_materialized_row(row)
        for field, value in (("deepcad_lazy", {}), ("_ng_rollout_id", "stale")):
            with self.assertRaisesRegex(ValueError, "lazy or per-run"):
                module.validate_materialized_row({**row, field: value})
        bad = deepcopy(row)
        bad["responses_create_params"]["input"][0]["content"] = []
        with self.assertRaisesRegex(ValueError, "incomplete view"):
            module.validate_materialized_row(bad)
        bad = deepcopy(row)
        bad["verifier_metadata"]["reference_step"] = "../escape.step"
        with self.assertRaisesRegex(ValueError, "contained relative"):
            module.validate_materialized_row(bad)

    def test_catalog_change_and_cross_source_replacement_refused(self):
        def invalid(example):
            row = self.materialize(example)
            row["verifier_metadata"]["dataset"] = "excluded-source"
            return row

        with patch.object(module, "materialize_one", side_effect=invalid):
            with self.assertRaisesRegex(ValueError, "eligible catalog"):
                module.render(self.selection, self.shards, 0, 2, 2, 8)
        with self.catalog.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            module.render(self.selection, self.shards, 0, 2, 2, 8)

    def test_render_outage_does_not_quarantine_or_publish_chunk(self):
        draws = list(module.read_jsonl(self.selection / "selected.jsonl"))[:8]
        expected = {draw["example"]["deepcad_lazy"]["record"]["id"] for draw in draws}
        calls = []

        def crash(args, **kwargs):
            calls.append(json.loads(Path(args[-2]).read_text())["record"]["id"])
            return SimpleNamespace(returncode=-11, stderr="broken renderer")

        with patch.object(lazy_inputs.subprocess, "run", side_effect=crash):
            with self.assertRaises(lazy_inputs.UnconfirmedReferenceFailure):
                module.render(self.selection, self.shards, 0, 1, 4, 8, max_chunks=1)
        self.assertTrue(calls)
        self.assertLessEqual(set(calls), expected)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertFalse(list((self.root / "cache").glob("*.rejected.json")))
        self.assertFalse(list(self.shards.glob("*.receipt.json")))

        def recover(args, **kwargs):
            lazy = json.loads(Path(args[-2]).read_text())
            row = self.materialize({"task_source": "deepcad_simple_agent", "deepcad_lazy": lazy})
            Path(args[-1]).write_text(json.dumps(row))
            return SimpleNamespace(returncode=0)

        with patch.object(lazy_inputs.subprocess, "run", side_effect=recover) as renderer:
            module.render(self.selection, self.shards, 0, 1, 4, 8, max_chunks=1)
            self.assertEqual(renderer.call_count, len(expected))
        self.assertTrue((self.shards / "chunk-000000.receipt.json").exists())
        self.assertFalse(list((self.root / "cache").glob("*.failure.json")))
        self.assertFalse(list((self.root / "cache").glob("*.rejected.json")))
        with patch.object(lazy_inputs.subprocess, "run", side_effect=AssertionError("rendered completed chunk")):
            module.render(self.selection, self.shards, 0, 1, 4, 8, max_chunks=1)

    def test_descriptor_variants_do_not_confirm_each_other_and_keep_overrides(self):
        draws = list(module.read_jsonl(self.selection / "selected.jsonl"))
        example = draws[0]["example"]
        for index, draw in enumerate(draws):
            draw["example"] = deepcopy(example)
            draw["example"]["responses_create_params"]["temperature"] = (index % 2) / 2
            draw["example"]["agent_ref"] = {"name": f"agent-{index % 2}"}
        path = self.selection / "selected.jsonl"
        path.write_text("".join(json.dumps(draw) + "\n" for draw in draws))
        metadata = json.loads((self.selection / "selection.json").read_text())
        metadata["selected_sha256"] = module.sha256(path)
        module.write_json(self.selection / "selection.json", metadata)
        with patch.object(
            lazy_inputs.subprocess, "run", return_value=SimpleNamespace(returncode=-11, stderr="broken renderer")
        ) as renderer:
            with self.assertRaises(lazy_inputs.UnconfirmedReferenceFailure):
                module.render(self.selection, self.shards, 0, 1, 4, 8, max_chunks=1)
            self.assertEqual(renderer.call_count, 1)
        self.assertFalse(list((self.root / "cache").glob("*.rejected.json")))
        self.assertFalse(list(self.shards.glob("*.receipt.json")))

        def recover(args, **kwargs):
            lazy = json.loads(Path(args[-2]).read_text())
            row = self.materialize({"task_source": "deepcad_simple_agent", "deepcad_lazy": lazy})
            Path(args[-1]).write_text(json.dumps(row))
            return SimpleNamespace(returncode=0)

        with patch.object(lazy_inputs.subprocess, "run", side_effect=recover) as renderer:
            module.render(self.selection, self.shards, 0, 1, 4, 8, max_chunks=1)
            self.assertEqual(renderer.call_count, 1)
        rows = [draw["row"] for draw in module.read_jsonl(self.shards / "chunk-000000.jsonl")]
        self.assertEqual([row["responses_create_params"]["temperature"] for row in rows], [0, 0.5] * 4)
        self.assertEqual([row["agent_ref"]["name"] for row in rows], ["agent-0", "agent-1"] * 4)

    def test_mixed_chunk_settings_refused(self):
        self.render_all()
        receipt_path = self.shards / "chunk-000001.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["render_overrides"] = {"seed": 999}
        module.write_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            module.assemble(self.selection, self.shards, self.release)

    def test_empty_and_invalid_base64_refused(self):
        example = next(module.read_jsonl(self.selection / "selected.jsonl"))["example"]
        for encoded in ("", "%%%"):
            row = self.materialize(example)
            row["responses_create_params"]["input"][0]["content"][0]["image_url"] = "data:image/png;base64," + encoded
            with self.assertRaises(ValueError):
                module.validate_materialized_row(row)

    def test_interrupted_selection_can_retry_without_manual_cleanup(self):
        output = self.root / "interrupted-selection"
        with patch.object(module, "write_json", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                module.select(self.manifest, output, self.sampler_config, 32, 8, 42)
        self.assertFalse(output.exists())
        module.select(self.manifest, output, self.sampler_config, 32, 8, 42)
        self.assertEqual(module.load_selection(output)[0]["num_samples"], 32)

    def test_release_publishes_only_after_validation_and_retries_interruption(self):
        self.render_all()
        validate = module.validate

        def interrupted(path):
            self.assertFalse(self.release.exists())
            validate(path)
            raise KeyboardInterrupt

        with patch.object(module, "validate", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                module.assemble(self.selection, self.shards, self.release)
        self.assertFalse(self.release.exists())
        module.assemble(self.selection, self.shards, self.release)
        module.validate(self.release)

    def test_nonimage_bytes_are_rejected(self):
        example = next(module.read_jsonl(self.selection / "selected.jsonl"))["example"]
        row = self.materialize(example)
        row["responses_create_params"]["input"][0]["content"][0]["image_url"] = "data:image/png;base64,dGVzdA=="
        with self.assertRaisesRegex(ValueError, "image bytes cannot be decoded"):
            module.validate_materialized_row(row)

    def test_resolve_propagates_unconfirmed_failures_and_provider_outages(self):
        example = next(module.read_jsonl(self.selection / "selected.jsonl"))["example"]
        for failure in (lazy_inputs.UnconfirmedReferenceFailure("unconfirmed"), RuntimeError("provider outage")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(module, "materialize_one", side_effect=failure) as renderer:
                    with self.assertRaises(type(failure)) as caught:
                        module._resolve(example)
                    self.assertIs(caught.exception, failure)
                    self.assertEqual(renderer.call_count, 1)


if __name__ == "__main__":
    unittest.main()
