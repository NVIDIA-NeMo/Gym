# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resources_servers.deepcad.dataset import lazy_inputs, sampled_lazy
from resources_servers.deepcad.dataset.source_operation_sampler import (
    SourceOperationSampler,
    SourceOperationSamplingConfig,
)
from resources_servers.deepcad.tests.test_lazy_render_timeout import worker
from resources_servers.deepcad.tests.test_reference_operation_fallback import examples


def test_export_preserves_draw_order_and_multiplicity(tmp_path):
    rows = []
    for index in range(8):
        rows.append(
            {
                "deepcad_lazy": {
                    "config": {
                        "dataset_root": str(tmp_path),
                        "cache_root": str(tmp_path / "cache"),
                        "renderer_python": "python",
                    },
                    "record": {"id": index, "dataset": "a" if index % 2 else "b", "reference_operations": ["box"]},
                }
            }
        )
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {"sources": ["a", "b"], "uniform_sources": [], "seed": 42}
    settings = tmp_path / "sampler.json"
    settings.write_text(json.dumps(config))
    output = tmp_path / "ordered"
    sampled_lazy.export(manifest, output, settings, None, 64, 8)
    native = list(
        SourceOperationSampler(
            manifest.read_text().splitlines(True), SourceOperationSamplingConfig(**config, samples_per_epoch=64), 8
        )
    )
    exported = list(sampled_lazy.read_range(output, 0, 64))
    assert [row["deepcad_lazy"]["record"]["id"] for row in exported] == native
    assert list(np.fromfile(output / "source-indices.i64", dtype="<i8")) == native
    assert all(row["deepcad_lazy"]["config"]["warm_start_cache"] for row in exported)
    assert not (tmp_path / "cache").exists()
    with pytest.raises(FileExistsError):
        sampled_lazy.export(manifest, output, settings, None, 64, 8)
    with pytest.raises(ValueError, match="outside"):
        list(sampled_lazy.read_range(output, 63, 2))
    with (output / "draws.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        list(sampled_lazy.read_range(output, 0, 1))


def test_partial_replacement_warms_once_and_replays_in_fresh_process(tmp_path, monkeypatch):
    row = examples(tmp_path, [("near", ["cut", "hole"], "source-a", "selected")])
    config = row["deepcad_lazy"]["config"]
    config["warm_start_cache"] = True
    row["responses_create_params"] = {"temperature": 0.42}
    draws = tmp_path / "draws.jsonl"
    draws.write_text((json.dumps(row) + "\n") * 8)
    (tmp_path / "draws-manifest.json").write_text(
        json.dumps({"draw_count": 8, "draws_sha256": sampled_lazy.sha256(draws)})
    )
    calls = []
    step = tmp_path / "references/near.step"
    step.parent.mkdir()
    run = sampled_lazy.subprocess.run

    def render(argv, **kwargs):
        if argv[1] == "-m":
            return run(argv, **kwargs)
        assert not Path(argv[-1]).is_relative_to(Path(config["cache_root"]))
        name = json.loads(Path(argv[-2]).read_text())["record"]["id"]
        calls.append(name)
        if name == "bad":
            Path(argv[-1]).with_suffix(".rejected.json").write_text(
                json.dumps({"reason": "slow program", "confirmation_key": "reference_execution_timeout"})
            )
            return SimpleNamespace(returncode=0)
        if name == "near":
            step.write_bytes(b"test reference")
            step.with_suffix(".json").write_text(
                json.dumps({"step_sha256": hashlib.sha256(step.read_bytes()).hexdigest()})
            )
        return worker(argv, **kwargs)

    monkeypatch.setattr(sampled_lazy.subprocess, "run", render)
    receipt = sampled_lazy.warm(tmp_path, 1, 6, 4)
    assert receipt == json.loads((tmp_path / "warm-1-6-ready.json").read_text())
    assert receipt["count"] == 6
    assert [entry["draw_index"] for entry in receipt["entries"]] == list(range(1, 7))
    assert receipt["cache_verification"]["fresh_process_cache_hits"]
    assert all(entry["reference_sha256"] == sampled_lazy.sha256(step) for entry in receipt["entries"])
    assert all(entry["cache_sha256"] == sampled_lazy.sha256(Path(entry["cache_path"])) for entry in receipt["entries"])
    first_path = tmp_path / "draw1-ready.json"
    first = first_path.read_bytes()
    assert json.loads(first)["entries"][0]["draw_index"] == 1
    assert json.loads(first)["cache_verification"]["count"] == 1
    sampled_lazy.warm(tmp_path, 1, 1, 1)
    assert first_path.read_bytes() == first
    assert calls == ["bad", "near"]
    # Only the pinned choice can satisfy the next requests after the catalog is removed.
    Path(config["reference_catalog"]).unlink()
    replay = lazy_inputs.materialize_one(row, cache_only=True)
    assert replay["verifier_metadata"]["reference_operation_fallback"]["missing_operations"] == ["loft"]
    assert replay["verifier_metadata"]["requested_task_id"] == "bad"
    assert replay["responses_create_params"]["temperature"] == 0.42
    strict = deepcopy(row)
    strict["deepcad_lazy"]["config"]["allow_partial_operation_replacements"] = False
    with pytest.raises(lazy_inputs.ReferenceRenderTimeout):
        lazy_inputs.materialize_one(strict, cache_only=True)
    step.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        sampled_lazy.verify(tmp_path, 1, 1)
    step.unlink()
    with pytest.raises(FileNotFoundError):
        sampled_lazy.verify(tmp_path, 1, 1)
    assert calls == ["bad", "near"]


def test_verify_cold_entry_never_starts_renderer(tmp_path, monkeypatch):
    row = examples(tmp_path, [])
    row["deepcad_lazy"]["config"]["warm_start_cache"] = True
    monkeypatch.setattr(
        lazy_inputs.subprocess, "run", lambda *a, **kw: pytest.fail("cold verification invoked renderer")
    )
    with pytest.raises(FileNotFoundError, match="Prompt cache entry missing"):
        lazy_inputs.materialize_one(row, cache_only=True)


@pytest.mark.parametrize("recover", [True, False])
def test_warm_outage_does_not_confirm_duplicate_draws(tmp_path, monkeypatch, recover):
    bad = examples(tmp_path, [("near", ["cut", "hole"], "source-a", "selected")])
    bad["deepcad_lazy"]["config"]["warm_start_cache"] = True
    good = deepcopy(bad)
    good["deepcad_lazy"]["record"].update(
        id="near",
        asset_id="near",
        task_id="near",
        source_sha256="near",
        reference_step="references/near.step",
        reference_operations=["cut", "hole"],
    )
    draws = tmp_path / "draws.jsonl"
    draws.write_text("".join(json.dumps(row) + "\n" for row in [good, *[bad] * 7]))
    (tmp_path / "draws-manifest.json").write_text(
        json.dumps({"draw_count": 8, "draws_sha256": sampled_lazy.sha256(draws)})
    )
    calls = []
    repaired = False
    run = sampled_lazy.subprocess.run

    def render(argv, **kwargs):
        if argv[1] == "-m":
            return run(argv, **kwargs)
        record = json.loads(Path(argv[-2]).read_text())["record"]
        calls.append(record["id"])
        if record["id"] == "bad" and not repaired:
            return SimpleNamespace(returncode=-11, stderr="broken renderer")
        step = tmp_path / record["reference_step"]
        step.parent.mkdir(exist_ok=True)
        step.write_bytes(record["id"].encode())
        step.with_suffix(".json").write_text(json.dumps({"step_sha256": sampled_lazy.sha256(step)}))
        return worker(argv, **{**kwargs, "recover": True})

    monkeypatch.setattr(lazy_inputs.subprocess, "run", render)
    with pytest.raises(lazy_inputs.UnconfirmedReferenceFailure):
        sampled_lazy.warm(tmp_path, 0, 8, 4)
    assert calls == ["near", "bad"]
    cache = tmp_path / "cache"
    assert len(list(cache.rglob("*.failure.json"))) == 1
    assert not list(cache.rglob("*.rejected.json"))
    assert not list(cache.rglob("*.resolved.json"))
    assert (tmp_path / "draw0-ready.json").exists()
    assert not (tmp_path / "warm-0-8-ready.json").exists()

    repaired = recover
    receipt = sampled_lazy.warm(tmp_path, 0, 8, 4)
    assert calls == ["near", "bad", "bad"]
    assert [entry["draw_index"] for entry in receipt["entries"]] == list(range(8))
    assert [entry["requested_task_id"] for entry in receipt["entries"]] == ["near", *["bad"] * 7]
    assert [entry["resolved_task_id"] for entry in receipt["entries"]] == ["near", *["bad" if recover else "near"] * 7]
    if recover:
        assert not list(cache.rglob("*.failure.json"))
        assert not list(cache.rglob("*.rejected.json"))
        assert not list(cache.rglob("*.resolved.json"))
    else:
        rejected = list(cache.rglob("*.rejected.json"))
        assert len(rejected) == 1
        assert json.loads(rejected[0].read_text())["confirmation_count"] == 2
