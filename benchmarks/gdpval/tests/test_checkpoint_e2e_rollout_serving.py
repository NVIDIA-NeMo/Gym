# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e import rollout_serving as serving


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    local = tmp_path / "local"
    (local / "job").mkdir(parents=True)
    for variable in serving.CONTAINER_ENV:
        path = local / "ray" if variable == "RAY_TMPDIR" else local / "job" / variable.lower()
        path.mkdir(parents=True)
        monkeypatch.setenv(variable, str(path))
    shared = tmp_path / "shared"
    model = shared / "checkpoint"
    _write(model / "config.json", '{"architectures": ["ExampleModel"]}\n')
    _write(model / "modeling_example.py", "MODEL_CONSTANT = 42\n")
    _write(model / "tokenizer.json", '{"vocab": {"same": 1}}\n')
    _write(model / "nested" / "chat_template.jinja", "{{ messages }}\n")
    _write(model / "model.safetensors.index.json", '{"weight_map": {"layer": "model-00001.safetensors"}}\n')
    weight = _write(model / "model-00001.safetensors", "fixture weight bytes\n")
    image = _write(shared / "vllm-openai:v0.27.1___tomer_with_gym.sqsh", "fixture image bytes\n")
    parser = _write(shared / "parsers" / "reasoning.py", "PARSER_NAME = 'fixture'\n")
    return SimpleNamespace(
        local=local, root=local / "job" / "serving", image=image, model=model, parser=parser, weight=weight
    )


def _stage(inputs: SimpleNamespace, mounts: str | None = None) -> dict:
    return serving.stage(
        inputs.root,
        inputs.image,
        inputs.model,
        mounts if mounts is not None else f"{inputs.parser.parent}:/parsers:ro",
        runtime_root=inputs.root.parent,
        local_root=inputs.local,
    )


def test_serving_stage_copies_code_image_and_tokenizer_preserving_weight_links_and_mount_targets(inputs) -> None:
    source_bytes = {path: path.read_bytes() for path in inputs.model.rglob("*") if path.is_file()}
    manifest = _stage(inputs)
    assert serving.verify(inputs.root, local_root=inputs.local) == manifest
    assert (inputs.root / "vllm.sqsh").read_bytes() == inputs.image.read_bytes()
    assert (inputs.root / "mount_0/reasoning.py").read_bytes() == inputs.parser.read_bytes()
    for source, content in source_bytes.items():
        copy = inputs.root / "model" / source.relative_to(inputs.model)
        assert copy.read_bytes() == content
        assert copy.is_symlink() == (source == inputs.weight)
        if copy.is_symlink():
            assert copy.readlink() == source
        else:
            assert copy.resolve().is_relative_to(inputs.local)
        assert source.read_bytes() == content
    assert manifest["environment"]["EXTRA_MOUNTS"] == (
        f"{inputs.root}/mount_0:/parsers:ro,{inputs.root.parent}:{inputs.root.parent},{inputs.local}/ray:{inputs.local}/ray"
    )
    assert manifest["environment"]["PYXIS_CONTAINER_ENV"] == ",".join(serving.CONTAINER_ENV)
    image_record = next(record for record in manifest["files"] if record["path"] == "vllm.sqsh")
    assert image_record["sha256"] == hashlib.sha256(inputs.image.read_bytes()).hexdigest()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$CONTAINER_IMAGE" "$MODEL_PATH" "$EXTRA_MOUNTS" "$PYXIS_CONTAINER_ENV"',
            "fixture",
            str(inputs.root / "environment.sh"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.splitlines() == list(manifest["environment"].values())


@pytest.mark.parametrize(
    "asset", ["vllm.sqsh", "model/config.json", "model/modeling_example.py", "mount_0/reasoning.py"]
)
def test_verify_rejects_modified_local_assets(inputs, asset: str) -> None:
    _stage(inputs)
    path = inputs.root / asset
    path.chmod(0o600)
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local serving asset changed"):
        serving.verify(inputs.root, local_root=inputs.local)


@pytest.mark.parametrize("source_name", ["image", "parser", "weight"])
def test_verify_rejects_changed_source_identity(inputs, source_name: str) -> None:
    _stage(inputs)
    path = getattr(inputs, source_name)
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed after staging"):
        serving.verify(inputs.root, local_root=inputs.local)


def test_verify_rejects_new_source_files_and_wrong_weight_link(inputs) -> None:
    _stage(inputs)
    link = inputs.root / "model" / inputs.weight.name
    link.unlink()
    link.symlink_to(inputs.parser)
    with pytest.raises(ValueError, match="weight link changed"):
        serving.verify(inputs.root, local_root=inputs.local)
    link.unlink()
    link.symlink_to(inputs.weight)
    _write(inputs.model / "new_model_code.py", "changed = True\n")
    with pytest.raises(ValueError, match="source changed after staging"):
        serving.verify(inputs.root, local_root=inputs.local)


@pytest.mark.parametrize("target", ["/model", "/model/plugin", "/lustre", "/", "relative", "/parsers/../model"])
def test_stage_rejects_unsafe_or_shadowing_mount_targets(inputs, target: str) -> None:
    with pytest.raises(ValueError, match="mount target"):
        _stage(inputs, f"{inputs.parser.parent}:{target}:ro")
    assert not inputs.root.exists()


def test_stage_rejects_writable_and_overlapping_parser_mounts(inputs) -> None:
    with pytest.raises(ValueError, match="unsupported parser mount"):
        _stage(inputs, f"{inputs.parser.parent}:/parsers:rw")
    with pytest.raises(ValueError, match="overlapping parser mount"):
        _stage(inputs, f"{inputs.parser.parent}:/parsers:ro,{inputs.parser.parent}:/parsers/nested:ro")


def test_stage_rejects_symlinked_code_and_existing_destination(inputs) -> None:
    (inputs.model / "imported.py").symlink_to(inputs.parser)
    with pytest.raises(ValueError, match="symlink in serving input"):
        _stage(inputs)
    assert not (inputs.root / "environment.sh").exists()
    with pytest.raises(ValueError, match="existing serving stage"):
        _stage(inputs)


def test_stage_rejects_image_drift_during_staging(inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    original = serving._copy_file

    def copy_then_change(source, destination, signature):
        digest = original(source, destination, signature)
        if source == inputs.image:
            source.write_text("changed image\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(serving, "_copy_file", copy_then_change)
    with pytest.raises(ValueError, match="source changed after staging"):
        _stage(inputs)


def test_verify_rejects_added_shared_code_and_missing_environment(inputs) -> None:
    _stage(inputs)
    extra = inputs.root / "shared_code.py"
    extra.symlink_to(inputs.parser)
    with pytest.raises(ValueError, match="unexpected serving asset"):
        serving.verify(inputs.root, local_root=inputs.local)
    extra.unlink()
    (inputs.root / "environment.sh").unlink()
    with pytest.raises(ValueError, match="environment differs"):
        serving.verify(inputs.root, local_root=inputs.local)


def test_stage_rejects_a_destination_inside_a_mount_source(inputs) -> None:
    with pytest.raises(ValueError, match="destination is inside its source"):
        _stage(inputs, f"{inputs.local}:/recursive:ro")
    assert not inputs.root.exists()


@pytest.mark.parametrize(
    "weight", ["../outside.safetensors", "/outside.safetensors", "missing.safetensors", "code.py"]
)
def test_stage_rejects_unsafe_or_missing_indexed_weights(inputs, weight: str) -> None:
    (inputs.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": weight}}))
    with pytest.raises(ValueError, match="weight.*(path|missing)"):
        _stage(inputs)


@pytest.mark.parametrize("runtime_location", ["shared", "sibling", "serving"])
def test_stage_rejects_invalid_runtime_mount_root(inputs, tmp_path: Path, runtime_location: str) -> None:
    runtime = {
        "shared": tmp_path / "shared",
        "sibling": inputs.local / "sibling",
        "serving": inputs.root,
    }[runtime_location]
    runtime.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="(outside node-local|below the node-local runtime)"):
        serving.stage(inputs.root, inputs.image, inputs.model, runtime_root=runtime, local_root=inputs.local)


def test_stage_rejects_shared_cache_and_local_cache_outside_self_mount(inputs, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "shared"))
    with pytest.raises(ValueError, match="outside node-local"):
        _stage(inputs)
    path = inputs.local / "unmounted-cache"
    path.mkdir()
    monkeypatch.setenv("HF_HOME", str(path))
    with pytest.raises(ValueError, match="outside the mounted runtime"):
        _stage(inputs)


def test_verify_rejects_environment_drift_before_serving(inputs, monkeypatch) -> None:
    _stage(inputs)
    monkeypatch.setenv("HF_HOME", str(inputs.model))
    with pytest.raises(ValueError, match="container environment changed"):
        serving.verify(inputs.root, local_root=inputs.local)


@pytest.mark.parametrize("target", ["runtime", "ray", "ancestor"])
def test_stage_rejects_parser_mounts_shadowing_runtime_paths(inputs, target: str) -> None:
    destination = {"runtime": inputs.root.parent, "ray": inputs.local / "ray", "ancestor": inputs.local}[target]
    with pytest.raises(ValueError, match="overlaps the node-local runtime"):
        _stage(inputs, f"{inputs.parser.parent}:{destination}:ro")
