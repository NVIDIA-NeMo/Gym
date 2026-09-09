# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tempfile
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
    assert json.loads((inputs.root / "manifest.json").read_text()) == manifest
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
    "suffix",
    [":/model:ro", ":/lustre:ro", ":relative:ro", ":/parsers", ":/parsers:rw", ":/parsers:ro,other:/extra:ro"],
)
def test_stage_only_accepts_the_readonly_parser_mount_contract(inputs, suffix: str) -> None:
    with pytest.raises(ValueError, match="unsupported parser mount"):
        _stage(inputs, f"{inputs.parser.parent}{suffix}")
    assert not inputs.root.exists()


def test_stage_rejects_symlinked_code_and_existing_destination(inputs) -> None:
    (inputs.model / "imported.py").symlink_to(inputs.parser)
    with pytest.raises(ValueError, match="symlink in serving input"):
        _stage(inputs)
    assert not (inputs.root / "environment.sh").exists()
    with pytest.raises(ValueError, match="existing serving stage"):
        _stage(inputs)


def test_stage_reads_image_once_and_never_reopens_the_destination_for_hashing(inputs, monkeypatch) -> None:
    original_open = Path.open
    reads = []

    def counted_open(path, mode="r", *args, **kwargs):
        if path == inputs.image and mode == "rb":
            reads.append(path)
        if path == inputs.root / "vllm.sqsh" and "r" in mode:
            raise AssertionError("staging reread the copied image")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    _stage(inputs)
    assert reads == [inputs.image]


@pytest.mark.parametrize("asset", ["image", "parser"])
def test_stage_rejects_source_changes_during_copy(inputs, monkeypatch, asset) -> None:
    source = getattr(inputs, asset)
    original_open = Path.open

    class ChangingReader:
        def __enter__(self):
            self.stream = original_open(source, "rb")
            return self

        def __exit__(self, *_):
            self.stream.close()

        def read(self, size):
            chunk = self.stream.read(size)
            if chunk:
                with original_open(source, "ab") as output:
                    output.write(b"changed while copying\n")
                # Return one original chunk, then EOF, while the real file has
                # grown. The production source stat must catch that change.
                self.read = lambda _: b""
            return chunk

    def changing_open(path, mode="r", *args, **kwargs):
        if path == source and mode == "rb":
            return ChangingReader()
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changing_open)
    with pytest.raises(ValueError, match="source changed while copying"):
        _stage(inputs)
    assert not (inputs.root / "manifest.json").exists()
    assert not (inputs.root / "environment.sh").exists()


def test_stage_rejects_a_short_image_copy_without_rehashing(inputs, monkeypatch) -> None:
    original_open = Path.open
    truncated = inputs.image.read_bytes()[:-1]

    def truncated_open(path, mode="r", *args, **kwargs):
        if path == inputs.image and mode == "rb":
            return io.BytesIO(truncated)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", truncated_open)
    with pytest.raises(ValueError, match="serving copy differs from source"):
        _stage(inputs)
    assert not (inputs.root / "manifest.json").exists()


def test_stage_rejects_a_destination_inside_a_mount_source(inputs) -> None:
    with pytest.raises(ValueError, match="destination is inside its source"):
        _stage(inputs, f"{inputs.local}:/parsers:ro")
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


def test_long_runtime_uses_short_tmp_forwarded_through_ray_mount(inputs, monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="gdpval-", dir="/tmp") as temporary:
        local = Path(temporary).resolve()
        job = local / ("job-" + "x" * 100)
        ray = local / "r" / "1"
        paths = {name: job / name.lower() for name in serving.CONTAINER_ENV}
        paths.update(RAY_TMPDIR=ray, TMPDIR=ray / "tmp")
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv(name, str(path))
        root = job / "serving"
        manifest = serving.stage(root, inputs.image, inputs.model, runtime_root=job, local_root=local)
        assert len(str(job / "tmp" / ("a" * 36)).encode()) > 107
        assert len(str(paths["TMPDIR"] / ("a" * 36)).encode()) <= 107
        assert manifest["container_environment"]["TMPDIR"] == str(paths["TMPDIR"])
        assert "TMPDIR" in manifest["environment"]["PYXIS_CONTAINER_ENV"].split(",")
        assert manifest["runtime_mounts"] == [str(job), str(ray)]
        assert manifest["environment"]["EXTRA_MOUNTS"] == f"{job}:{job},{ray}:{ray}"


@pytest.mark.parametrize("location", ["outside", "ray-root", "symlink-escape"])
def test_stage_rejects_external_tmp_outside_ray_subdirectory(inputs, monkeypatch, location) -> None:
    outside = inputs.local / "ray-other"
    outside.mkdir()
    ray = inputs.local / "ray"
    if location == "symlink-escape":
        path = ray / "tmp"
        path.symlink_to(outside, target_is_directory=True)
    else:
        path = ray if location == "ray-root" else outside
    monkeypatch.setenv("TMPDIR", str(path))
    with pytest.raises(ValueError, match="outside the mounted runtime: TMPDIR"):
        _stage(inputs)
    assert not inputs.root.exists()
