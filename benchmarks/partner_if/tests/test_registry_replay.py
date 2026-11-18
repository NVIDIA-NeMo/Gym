# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A registry replay needs no source checkout, and corrupted assets stay unscored."""

import base64
import hashlib
import json

import pytest

from benchmarks.partner_if import prepare as registry
from benchmarks.partner_if.replay_assets import REPLAY_FILES, materialize_bundle, prepared_bundle_matches
from benchmarks.partner_if.tests.fixture_data import PARAMS, SOURCE, TASK
from resources_servers.partner_if.recorded_context import assets as replay


@pytest.fixture
def assets(tmp_path):
    bundle = {"schema_version": "partner-if-replay-assets-v2", "files": {}}
    for name in REPLAY_FILES | {"runtime.json"}:
        data = (
            (json.dumps(PARAMS, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            if name == "actor-params.json"
            else (SOURCE / "assets/ngc-recorded-context" / name).read_bytes()
        )
        bundle["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "data": base64.b64encode(data).decode()}
    path = tmp_path / "assets.json"
    path.write_text(json.dumps(bundle))
    return path


@pytest.mark.parametrize("fault", ["path", "digest", "encoding", "missing"])
def test_invalid_asset_bundle_does_not_create_runtime_files(tmp_path, assets, fault):
    bundle = json.loads(assets.read_text())
    if fault == "path":
        bundle["files"]["../../escape"] = bundle["files"].pop("case.json")
    elif fault == "digest":
        bundle["files"]["case.json"]["sha256"] = "0" * 64
    elif fault == "encoding":
        bundle["files"]["case.json"]["data"] = "not base64!"
    else:
        del bundle["files"]["case.json"]
    assets.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        materialize_bundle(assets, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_registry_only_assets_keep_original_actor_input_and_validate_all_members(tmp_path, assets):
    package = tmp_path / "package"
    materialize_bundle(assets, package)
    assert prepared_bundle_matches(assets, package)
    config = replay.ReplayConfig(package_dir=package, coverage_profile="expanded-v4", temporal_history=True)
    case, bundle = replay.load_inputs(config, PARAMS)
    assert case["messages"] and case["tools"]
    assert bundle["runtime"] and bundle["temporal_world"]
    assert replay.actor_identity(
        json.loads((package / "ngc-recorded-context/actor-params.json").read_text())
    ) == replay.actor_identity(PARAMS)
    (package / "ngc-recorded-context/actor-params.json").write_text("{}")
    assert not prepared_bundle_matches(assets, package)
    with pytest.raises(ValueError, match="actor-params"):
        replay.load_inputs(config, PARAMS)


@pytest.mark.parametrize("fault", [None, "digest", "row"])
def test_registry_prepare_verifies_assets_before_replacing_existing_data(tmp_path, assets, monkeypatch, fault):
    row = json.loads(json.dumps(TASK))
    row["task_name"] = "pif_029"
    raw = json.dumps(row) + "\n"
    if fault == "row":
        altered = json.loads(raw)
        altered["responses_create_params"]["input"][0]["content"] = "changed"
        raw = json.dumps(altered) + "\n"
    output = tmp_path / "partner_if_benchmark.jsonl"
    output.write_text("previous")
    for key, value in {
        "DATA_DIR": tmp_path,
        "OUTPUT_FPATH": output,
        "TOOLS_FPATH": tmp_path / "tools.jsonl",
        "REPLAY_FPATH": tmp_path / "ngc_replay/replay.jsonl",
        "REPLAY_PACKAGE_DIR": tmp_path / "ngc_replay/registry",
        "REPLAY_ASSET_FPATH": tmp_path / "downloaded-assets.json",
        "ROW_COUNT": 1,
        "SHA256": hashlib.sha256(raw.encode()).hexdigest(),
        "REPLAY_ASSET_SHA256": "0" * 64 if fault == "digest" else hashlib.sha256(assets.read_bytes()).hexdigest(),
    }.items():
        monkeypatch.setattr(registry, key, value)

    def download(config):
        data = assets.read_bytes() if config.artifact_fpath == registry.REPLAY_ASSET_ARTIFACT else raw.encode()
        from pathlib import Path

        Path(config.output_fpath).write_bytes(data)

    monkeypatch.setattr(registry, "download_jsonl_dataset", download)
    if fault:
        with pytest.raises(ValueError):
            registry.prepare()
        assert output.read_text() == "previous"
    else:
        assert registry.prepare().read_text() == raw
        assert registry.is_prepared_data_current(output)
        (registry.REPLAY_PACKAGE_DIR / "ngc-recorded-context/runtime.json").unlink()
        assert registry.REPLAY_FPATH.read_text() == raw
        assert not registry.is_prepared_data_current(output)


@pytest.mark.parametrize("fault", [None, "source", "assets", "version", "existing"])
def test_release_staging_changes_only_harness_metadata_and_never_publishes(tmp_path, assets, monkeypatch, fault):
    from benchmarks.partner_if.stage_recorded_context_release import HARNESS, stage

    # The second row's spacing and Unicode separator must survive byte for byte.
    other = {
        "responses_create_params": {"input": "preserve \u0085 spacing"},
        "verifier_metadata": {"partner_if": {"task_id": "pif_028"}},
    }
    untouched = json.dumps(other, ensure_ascii=False, indent=None) + "\n"
    original = json.dumps(TASK) + "\n" + untouched
    source = tmp_path / "source.jsonl"
    source.write_text(original)
    monkeypatch.setattr(registry, "SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(registry, "REPLAY_ASSET_SHA256", hashlib.sha256(assets.read_bytes()).hexdigest())
    monkeypatch.setattr(registry, "ROW_COUNT", 2)
    output = tmp_path / "candidate"
    if fault == "source":
        source.write_text(original + " ")
    elif fault == "assets":
        assets.write_text(assets.read_text() + " ")
    elif fault == "existing":
        output.mkdir()
    version = registry.VERSION if fault == "version" else "2.1.8"
    if fault:
        with pytest.raises((ValueError, FileExistsError)):
            stage(source, assets, output, version=version)
        assert not (output / registry.ARTIFACT).exists()
        return
    manifest = json.loads(stage(source, assets, output, version=version).read_text())
    candidate = (output / registry.ARTIFACT).read_text()
    assert candidate.endswith(untouched)
    row = json.loads(candidate.split("\n")[0])
    assert row.pop("recorded_context_harness") == HARNESS
    assert row == TASK
    assert (output / registry.REPLAY_ASSET_ARTIFACT).read_bytes() == assets.read_bytes()
    assert source.read_text() == original
    assert manifest["status"] == "local_candidate_not_published"
    assert manifest["artifacts"][registry.ARTIFACT] == hashlib.sha256(candidate.encode()).hexdigest()
