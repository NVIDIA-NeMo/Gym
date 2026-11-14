# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A registry replay needs no source checkout, and corrupted assets stay unscored."""

import base64
import copy
import hashlib
import json

import pytest

from benchmarks.partner_if import prepare as registry
from benchmarks.partner_if import prepare_ngc_replay
from benchmarks.partner_if.replay_assets import REPLAY_FILES, materialize_bundle, prepared_bundle_matches
from benchmarks.partner_if.tests.test_ordering_prefix import fields
from benchmarks.partner_if.tests.test_replay import PARAMS, SOURCE, TASK, draft, probe, response
from responses_api_agents.ngc_recorded_context import replay


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


@pytest.mark.asyncio
@pytest.mark.parametrize("passed", [True, False])
async def test_registry_only_assets_keep_original_requests_and_stop_at_the_ordering_event(tmp_path, assets, passed):
    package = tmp_path / "package"
    materialize_bundle(assets, package)
    assert prepared_bundle_matches(assets, package)
    calls = []

    async def provider(payload, timeout):
        calls.append(copy.deepcopy(payload))
        assert len(calls) == 1
        return 200, response(probe() if passed else draft())

    async def review(*args):
        return fields(passed)

    config = replay.ReplayConfig(
        package_dir=package,
        artifacts_dir=tmp_path / "captures",
        coverage_profile="expanded-v4",
        temporal_history=True,
        evaluation_boundary="ordering_prefix_v1",
    )
    _, _, _, diagnostics = await replay.execute(
        config, PARAMS, model="test", secret="", request_api=provider, review_prefix=review
    )
    assert diagnostics["termination"] == ("ordering_passed" if passed else "ordering_failed")
    case = json.loads((package / "ngc-recorded-context/case.json").read_text())
    assert calls[0]["messages"] == case["messages"]
    assert calls[0]["tools"] == case["tools"]
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
        assert prepare_ngc_replay.is_prepared_data_current(registry.REPLAY_FPATH)
        (registry.REPLAY_PACKAGE_DIR / "ngc-recorded-context/runtime.json").unlink()
        assert registry.REPLAY_FPATH.read_text() == raw
        assert not prepare_ngc_replay.is_prepared_data_current(registry.REPLAY_FPATH)
