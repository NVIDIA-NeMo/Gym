# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared bundles preserve cached task semantics and publish only complete output."""

import base64
import json
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from resources_servers.deepcad.dataset import lazy_inputs, sampled_lazy
from resources_servers.deepcad.tests.test_reference_operation_fallback import examples


def image_url(color):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def export_draws(path, rows):
    path.mkdir()
    draws = path / "draws.jsonl"
    draws.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "draws-manifest.json").write_text(
        json.dumps({"draw_count": len(rows), "draws_sha256": sampled_lazy.sha256(draws)})
    )


@pytest.fixture
def cached_draws(tmp_path, monkeypatch):
    requested = examples(tmp_path, [("near", ["cut", "hole"], "source-a", "selected")])
    requested["deepcad_lazy"]["config"]["warm_start_cache"] = True
    requested["responses_create_params"] = {"temperature": 0.42, "max_output_tokens": 123}
    requested["agent_ref"] = {"name": "caller-route"}
    requested["caller_metadata"] = {"draw": 7}
    calls = []

    def render(argv, **kwargs):
        payload = json.loads(Path(argv[-2]).read_text())
        record, config = payload["record"], payload["config"]
        calls.append(record["id"])
        if record["id"] == "bad":
            Path(argv[-1]).with_suffix(".rejected.json").write_text(json.dumps({"reason": "Invalid CAD solid"}))
            return SimpleNamespace(returncode=0)
        reference = tmp_path / record["reference_step"]
        reference.parent.mkdir(exist_ok=True)
        reference.write_bytes(b"real reference bytes")
        reference.with_suffix(".json").write_text(json.dumps({"step_sha256": sampled_lazy.sha256(reference)}))
        views = 1 if config["seed"] == 42 else 3
        row = {
            "verifier_metadata": {
                "task_id": record["task_id"],
                "source_sha256": record["source_sha256"],
                "reference_step": reference.name,
                "difficulty": record["difficulty"],
                "view_mode": "diagonal" if views == 1 else "axis",
                "reference_operations": record["reference_operations"],
            },
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Preserve exact CAD instructions.\n"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Preserve operations and view instructions.\n"},
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": image_url((config["seed"], i, 90)),
                                    "detail": "high",
                                }
                                for i in range(views)
                            ],
                        ],
                    },
                ]
            },
        }
        Path(argv[-1]).write_text(json.dumps(row))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(lazy_inputs.subprocess, "run", render)
    variant = deepcopy(requested)
    variant["deepcad_lazy"]["config"]["seed"] = 43
    caller = deepcopy(requested)
    caller["responses_create_params"]["temperature"] = 0.9
    caller["agent_ref"] = {"name": "other-route"}
    draws = [requested, caller, requested, variant]
    warm = [lazy_inputs.materialize_one(draw) for draw in draws]
    assert calls == ["bad", "near", "bad", "near"]
    monkeypatch.setattr(lazy_inputs.subprocess, "run", lambda *a, **kw: pytest.fail("cache-only export rendered"))
    output = tmp_path / "ordered"
    export_draws(output, draws)
    return output, draws, warm


def test_prepare_preserves_rows_pixels_references_duplicates_and_variants(tmp_path, cached_draws):
    output, draws, warm = cached_draws
    before = {path: path.read_bytes() for path in (tmp_path / "cache").rglob("*.json")}
    original = deepcopy(draws)
    destination = tmp_path / "prepared"
    receipt = sampled_lazy.prepare(output, destination, start=0, count=4, cache_only=True)
    assert {path.name for path in destination.iterdir()} == {"rows.jsonl", "assets"}
    rows = [json.loads(line) for line in (destination / "rows.jsonl").read_text().splitlines()]
    assert receipt == json.loads((destination / "assets/manifest.json").read_text())
    assert sampled_lazy.sha256(destination / "rows.jsonl") == receipt["rows_sha256"]
    assert rows[0] == rows[2]
    assert rows[0]["agent_ref"] != rows[1]["agent_ref"]
    assert [row["responses_create_params"]["temperature"] for row in rows] == [0.42, 0.9, 0.42, 0.42]
    all_images = []
    for row, expected in zip(rows, warm, strict=True):
        assert "deepcad_lazy" not in row
        reference = (destination / "assets" / row["verifier_metadata"]["reference_step"]).resolve()
        assert reference.is_relative_to(destination / "assets")
        assert reference.read_bytes() == b"real reference bytes"
        restored = deepcopy(row)
        restored["verifier_metadata"]["reference_step"] = expected["verifier_metadata"]["reference_step"]
        for actual, previous in zip(
            restored["responses_create_params"]["input"][-1]["content"][1:],
            expected["responses_create_params"]["input"][-1]["content"][1:],
            strict=True,
        ):
            path = Path(actual["image_url"])
            assert path.is_absolute() and path.is_relative_to(destination / "assets") and path.exists()
            raw = base64.b64decode(previous["image_url"].split(",")[1])
            assert path.read_bytes() == raw
            with Image.open(path) as current, Image.open(BytesIO(raw)) as cached:
                assert current.tobytes() == cached.tobytes()
            all_images.append(str(path))
            actual["image_url"] = previous["image_url"]
        assert restored == expected
        assert row["verifier_metadata"]["reference_operation_fallback"]["missing_operations"] == ["loft"]
        assert row["verifier_metadata"]["requested_task_id"] == "bad"
    assert len(set(all_images)) == 4
    assert draws == original
    assert {path: path.read_bytes() for path in before} == before
    for name, digest in receipt["assets_sha256"].items():
        assert sampled_lazy.sha256(destination / "assets" / name) == digest
    # Warm caches still satisfy their original embedded-image invariant.
    assert [lazy_inputs.materialize_one(draw, cache_only=True) for draw in draws] == warm


def test_prepare_failure_never_publishes_or_replaces_output(tmp_path, cached_draws):
    output, draws, _ = cached_draws
    destination = tmp_path / "prepared"
    cold = deepcopy(draws[0])
    cold["deepcad_lazy"]["config"]["seed"] = 987
    export_draws(tmp_path / "cold", [draws[0], cold])
    with pytest.raises(FileNotFoundError, match="Prompt cache entry missing"):
        sampled_lazy.prepare(tmp_path / "cold", destination, start=0, count=2, cache_only=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cad-prepare-*"))
    sampled_lazy.prepare(output, destination, start=0, count=1, cache_only=True)
    before = {path: path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        sampled_lazy.prepare(output, destination, start=0, count=1, cache_only=True)
    assert {path: path.read_bytes() for path in before} == before
    with pytest.raises(ValueError, match="outside"):
        sampled_lazy.prepare(output, tmp_path / "invalid", start=3, count=2, cache_only=True)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("unsupported", ["opencode", "target_images", "cached_target_images"])
def test_prepare_rejects_unsupported_workspace_image_seeding(tmp_path, cached_draws, unsupported):
    _, draws, _ = cached_draws
    if unsupported == "opencode":
        draws[0]["deepcad_lazy"]["config"]["harness"] = "opencode"
    elif unsupported == "target_images":
        draws[0]["target_images"] = [image_url("white")]
    else:
        lazy = deepcopy(draws[0]["deepcad_lazy"])
        lazy["record"] = json.loads(lazy_inputs.resolution_target(lazy).read_text())["record"]
        cache = lazy_inputs.cache_target(lazy)
        row = json.loads(cache.read_text())
        row["target_images"] = [image_url("white")]
        cache.write_text(json.dumps(row))
    export_draws(tmp_path / "unsupported", draws[:1])
    destination = tmp_path / "prepared"
    with pytest.raises(ValueError, match="simple harness without target_images"):
        sampled_lazy.prepare(tmp_path / "unsupported", destination, start=0, count=1, cache_only=True)
    assert not destination.exists()


@pytest.mark.parametrize("failure", ["base64", "image", "reference", "short_range"])
def test_prepare_rejects_corrupt_or_incomplete_range_atomically(tmp_path, cached_draws, monkeypatch, failure):
    output, draws, _ = cached_draws
    if failure == "reference":
        (tmp_path / "references/near.step").write_bytes(b"corrupt STEP")
    elif failure == "short_range":
        read_range = sampled_lazy.read_range
        monkeypatch.setattr(sampled_lazy, "read_range", lambda *a: iter(list(read_range(*a))[:1]))
    else:
        lazy = deepcopy(draws[-1]["deepcad_lazy"])
        lazy["record"] = json.loads(lazy_inputs.resolution_target(lazy).read_text())["record"]
        cache = lazy_inputs.cache_target(lazy)
        row = json.loads(cache.read_text())
        row["responses_create_params"]["input"][-1]["content"][1]["image_url"] = "data:image/png;base64," + (
            "!bad!" if failure == "base64" else base64.b64encode(b"not an image").decode()
        )
        cache.write_text(json.dumps(row))
    destination = tmp_path / "rejected"
    with pytest.raises(
        (ValueError, OSError),
        match={
            "base64": "base64",
            "image": "cannot identify image",
            "reference": "checksum mismatch",
            "short_range": "Incomplete draw range",
        }[failure],
    ):
        sampled_lazy.prepare(output, destination, start=0, count=4, cache_only=True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cad-prepare-*"))


def test_prepare_default_propagates_first_unconfirmed_failure(tmp_path, cached_draws, monkeypatch):
    output, draws, _ = cached_draws
    cold = deepcopy(draws[0])
    cold["deepcad_lazy"]["config"]["seed"] = 999
    export_draws(tmp_path / "cold", [cold, cold])
    calls = []

    def crash(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=-11, stderr="renderer crash")

    monkeypatch.setattr(lazy_inputs.subprocess, "run", crash)
    destination = tmp_path / "rejected"
    with pytest.raises(lazy_inputs.UnconfirmedReferenceFailure):
        sampled_lazy.prepare(tmp_path / "cold", destination, start=0, count=2)
    assert len(calls) == 1
    assert not destination.exists()
    assert not list((tmp_path / "cache").rglob("*.timeout-*.json"))
    assert not list((tmp_path / "cache").rglob("*.failure.building"))


@pytest.mark.asyncio
async def test_prepared_reference_uses_real_verifier_root_confinement(tmp_path, cached_draws):
    from unittest.mock import AsyncMock, MagicMock, patch

    from nemo_gym.client import ServerClient
    from nemo_gym.sandbox import SandboxExecResult
    from resources_servers.deepcad.app import DeepCADResourcesServer, DeepCADResourcesServerConfig

    output, _, _ = cached_draws
    destination = tmp_path / "prepared"
    sampled_lazy.prepare(output, destination, start=0, count=1, cache_only=True)
    row = json.loads((destination / "rows.jsonl").read_text())
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    server = DeepCADResourcesServer(
        config=DeepCADResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            name="deepcad",
            entrypoint="app.py",
            reference_root=str(destination / "assets"),
            sandbox_provider={"docker": {}},
        ),
        server_client=client,
    )
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("100", "", 0),
        SandboxExecResult('{"path":"/tmp/deepcad-sealed-test/prediction.step"}', "", 0),
    ]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)

    async def score(prediction, reference, **kwargs):
        assert reference.is_relative_to(destination / "assets")
        assert sampled_lazy.sha256(reference) == sampled_lazy.sha256(tmp_path / "references/near.step")
        return {"reward": 0.82}

    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context) as factory:
        with patch.object(DeepCADResourcesServer, "_score", AsyncMock(side_effect=score)) as scorer:
            result = await server._evaluate_once("result=1", row["verifier_metadata"]["reference_step"])
            assert result == {"reward": 0.82}
            factory.assert_called_once()
            scorer.assert_awaited_once()
            outside = tmp_path / "outside.step"
            outside.write_bytes(b"outside")
            (destination / "assets/escape.step").symlink_to(outside)
            for reference in ("../../outside.step", "escape.step"):
                factory.reset_mock()
                scorer.reset_mock()
                result = await server._evaluate_once("result=1", reference)
                assert result["failure_kind"] == "deepcad:reference" and result["mask_sample"]
                factory.assert_not_called()
                scorer.assert_not_awaited()
