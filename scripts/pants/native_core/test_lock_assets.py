# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from collect import pytest_sessionfinish, targets
from lock_assets import INPUTS, bundle_key, fingerprint, lock_assets, source_uv_version, uv_version_from_text
from prepare import validate_lock


def fixture_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in INPUTS:
        (source / name).write_text(name)
    setup = source / "scripts/ci/setup_dev.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text("curl https://astral.sh/uv/0.11.29/install.sh | sh")
    return source


def test_exact_bundle_selection_and_rejects_stale_or_corrupt(tmp_path):
    source = fixture_source(tmp_path)
    controller = tmp_path / "controller"
    inputs = fingerprint(source)
    key = bundle_key(inputs, source_uv_version(source))
    assets = controller / "lock-bundles" / key
    assets.mkdir(parents=True)
    resolves = {}
    for group in ("core", "sandbox"):
        (assets / f"{group}.requirements.txt").write_text("example==1\n")
        (assets / f"{group}.lock").write_text(
            json.dumps({"locked_resolves": [{"locked_requirements": [{"project_name": "example", "version": "1"}]}]})
        )
        resolves[group] = validate_lock(assets / f"{group}.requirements.txt", assets / f"{group}.lock")
    manifest = dict(
        inputs=inputs, uv_version="0.11.29", pants_version="2.30.2", pex_version="2.101.6", resolves=resolves
    )
    (assets / "lock-inputs.json").write_text(json.dumps(manifest))
    assert lock_assets(source, controller) == assets
    (source / "pyproject.toml").write_text("changed")
    with pytest.raises(ValueError, match="No validated dependency bundle"):
        lock_assets(source, controller)
    (source / "pyproject.toml").write_text("pyproject.toml")
    (assets / "sandbox.requirements.txt").write_text("example==2\n")
    with pytest.raises(ValueError, match="Version mismatch"):
        lock_assets(source, controller)


@pytest.mark.parametrize(
    "text",
    [
        "https://astral.sh/uv/install.sh",
        "https://astral.sh/uv/0.11.29/install.sh https://astral.sh/uv/0.13.4/install.sh",
    ],
)
def test_unpinned_or_ambiguous_toolchain_fails(text):
    with pytest.raises(ValueError, match="exactly one"):
        uv_version_from_text(text)


def test_legacy_collection_is_one_complete_shard(tmp_path, monkeypatch):
    import collect

    output = tmp_path / "collection.json"
    monkeypatch.setenv("GYM_NATIVE_COLLECTION_OUTPUT", str(output))
    monkeypatch.setenv("GYM_NATIVE_COLLECTION_MARKER", "not sandbox")
    monkeypatch.setattr(collect.subprocess, "check_output", lambda *a, **kw: "exact-sha\n")
    session = SimpleNamespace(
        config=SimpleNamespace(rootpath=tmp_path),
        items=[
            SimpleNamespace(path=tmp_path / "tests/test_first.py", nodeid="tests/test_first.py::test_ok"),
            SimpleNamespace(path=tmp_path / "tests/test_second.py", nodeid="tests/test_second.py::test_ok"),
        ],
    )
    pytest_sessionfinish(session, 0)
    manifest = json.loads(output.read_text())
    assert manifest["shards"] == 1
    assert manifest["durations_sha256"] is None
    assert manifest["testcase_count"] == 2
    assert len(targets(manifest, resolve="core", shard=0)) == 2
    with pytest.raises(ValueError, match="valid sampled shard"):
        targets(manifest, resolve="core", shard=1)
