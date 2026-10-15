# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
import os
import tarfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest


@pytest.fixture
def helper() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/save_warmed_vllm_container.py"
    spec = importlib.util.spec_from_file_location("save_warmed_vllm_container", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waits_for_all_replicas(helper: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    response = MagicMock()
    response.__enter__.return_value.status = 200
    opener = MagicMock()
    opener.open.side_effect = [response, URLError("still loading"), response, response]
    monkeypatch.setattr(helper.urllib.request, "build_opener", lambda *args: opener)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    helper.wait_for_engines(["http://node:8001", "http://node:8002"], 10)
    assert [call.args[0] for call in opener.open.call_args_list] == [
        "http://node:8001/health",
        "http://node:8002/health",
        "http://node:8001/health",
        "http://node:8002/health",
    ]


def test_readiness_timeout(helper: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.time, "monotonic", iter([0, 11]).__next__)
    with pytest.raises(TimeoutError, match="startup timed out"):
        helper.wait_for_engines(["http://node:8001"], 10)


def test_collect_requires_current_warmup(helper: ModuleType, tmp_path: Path) -> None:
    marker = tmp_path / ".warmup-start"
    marker.touch()
    os.utime(marker, (100, 100))
    directory = tmp_path / ("a" * 64)
    directory.mkdir()
    path = directory / "autotune_configs.json"
    contents = json.dumps({"_metadata": {"gpu": "GB200"}, "ssu|shape": ["runner", 1]})
    path.write_text(contents)
    os.utime(path, (90, 90))
    with pytest.raises(ValueError, match="No autotuning caches were written"):
        helper.collect_caches(tmp_path, allow_empty=False)
    os.utime(path, (110, 110))
    assert helper.collect_caches(tmp_path, allow_empty=False) == {"a" * 64: contents}


def test_rewarming_baked_image_accepts_reused_autotuning(helper: ModuleType, tmp_path: Path) -> None:
    (tmp_path / ".warmup-start").touch()
    (tmp_path / ".baked").touch()
    directory = tmp_path / ("a" * 64)
    directory.mkdir()
    path = directory / "autotune_configs.json"
    path.write_text('{"cached-tactic": 1}')
    os.utime(path, (1, 1))
    assert helper.collect_caches(tmp_path, allow_empty=False) == {"a" * 64: '{"cached-tactic": 1}'}
    path.unlink()
    with pytest.raises(ValueError, match="No autotuning caches"):
        helper.collect_caches(tmp_path, allow_empty=False)


def test_headless_node_may_have_no_cache(helper: ModuleType, tmp_path: Path) -> None:
    (tmp_path / ".warmup-start").touch()
    assert helper.collect_caches(tmp_path, allow_empty=True) == {}
    with pytest.raises(ValueError, match="No autotuning caches"):
        helper.collect_caches(tmp_path, allow_empty=False)


def test_install_combines_roles_and_preserves_complete_files(helper: ModuleType, tmp_path: Path) -> None:
    prefill = json.dumps({"_metadata": {"gpu": "GB200"}, "prefill|shape": ["runner", 1]})
    decode = json.dumps({"_metadata": {"gpu": "GB200"}, "decode|shape": ["runner", 2]})
    another_replica = json.dumps({"_metadata": {"gpu": "GB200"}, "prefill|shape": ["runner", 3]})
    helper.install_caches(
        tmp_path,
        [json.dumps({"a" * 64: prefill}), json.dumps({"b" * 64: decode}), json.dumps({"a" * 64: another_replica})],
    )
    assert (tmp_path / ("a" * 64) / "autotune_configs.json").read_text() == prefill
    assert (tmp_path / ("b" * 64) / "autotune_configs.json").read_text() == decode
    assert (tmp_path / ".baked").is_file()


@pytest.mark.parametrize("contents", ["{}", '{"_metadata": {}}', "[]", "invalid json"])
def test_rejects_invalid_cache(helper: ModuleType, tmp_path: Path, contents: str) -> None:
    with pytest.raises(ValueError):
        helper.install_caches(tmp_path, [json.dumps({"a" * 64: contents})])
    assert not (tmp_path / ".baked").exists()


def test_rejects_path_traversal_and_empty_collection(helper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid vLLM cache configuration hash"):
        helper.install_caches(tmp_path, [json.dumps({"../outside": '{"op": 1}'})])
    with pytest.raises(ValueError, match="No autotuning caches collected"):
        helper.install_caches(tmp_path, ["{}"])


def test_rejects_incompatible_replica_entries(helper: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="incompatible cache entry sets"):
        helper.install_caches(tmp_path, [json.dumps({"a" * 64: '{"op1": 1}'}), json.dumps({"a" * 64: '{"op2": 2}'})])
    assert not (tmp_path / ".baked").exists()


def test_warmup_archive_merges_kernels_and_startup_plans_from_all_nodes(helper: ModuleType, tmp_path: Path) -> None:
    stream = io.BytesIO()
    checkpoint = "checkpointing_ssu_s_f32_i_bf16_dt_bf16_w_bf16_a_f32_si_i32_d_64_ds_128_np_6_mw_16_hpg_16_ng_2"
    materialize = "replayssm_materialize_s_f32_i_bf16_a_f32_d_64_ds_128_hpg_16_mw_16_pr_0"
    for node, module in enumerate((checkpoint, materialize)):
        root = tmp_path / str(node)
        cache = root / "autotune"
        cache.mkdir(parents=True)
        (cache / ".warmup-start").touch()
        config = cache / (str(node) * 64)
        config.mkdir()
        (config / "autotune_configs.json").write_text(json.dumps({f"role-{node}": 1}))
        # Include the custom-image build namespace and GPU architecture.
        directory = (
            root / "jit" / "flashinfer-builds" / "build-hash" / ".cache/flashinfer/0.7.0/100a/cached_ops" / module
        )
        directory.mkdir(parents=True)
        (directory / f"{module}.so").write_bytes(b"compiled-" + bytes([node]))
        (directory / "kernel.cuda.o").write_bytes(b"do not export object files")
        startup = root / "startup"
        startup.mkdir()
        for fingerprint in (str(node) * 16, "a" * 16):
            (startup / f"startup_plan_{fingerprint}.json").write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "fingerprint": fingerprint,
                        "kv_cache_memory_bytes": 1000 - node,
                        "free_memory_baseline": 2000 + node,
                    }
                )
            )
        helper.dump_warmup(cache, root / "jit", startup, stream, allow_empty=False)

    stream.seek(0)
    target = tmp_path / "image"
    libraries = helper.install_warmup(target / "autotune", target / "aot", target / "startup", stream)
    assert {library.stem for library in libraries} == {checkpoint, materialize}
    assert (target / "aot" / checkpoint / f"{checkpoint}.so").read_bytes() == b"compiled-\x00"
    assert (target / "aot" / materialize / f"{materialize}.so").read_bytes() == b"compiled-\x01"
    assert not list(target.rglob("*.o"))
    assert len(list((target / "autotune").glob("*/autotune_configs.json"))) == 2
    assert len(list((target / "startup").glob("*.json"))) == 3
    shared_plan = json.loads((target / "startup" / f"startup_plan_{'a' * 16}.json").read_text())
    assert shared_plan["kv_cache_memory_bytes"] == 999
    assert shared_plan["free_memory_baseline"] == 2001
    assert (target / "autotune" / ".startup-plans-baked").is_file()


def test_compile_caches_include_both_roles_without_mixing_replicas(helper: ModuleType, tmp_path: Path) -> None:
    stream = io.BytesIO()
    for node, config in enumerate(("a", "b", "a")):
        root = tmp_path / str(node)
        cache = root / "cache"
        (cache / (config * 64)).mkdir(parents=True)
        (cache / ".warmup-start").touch()
        (cache / (config * 64) / "autotune_configs.json").write_text('{"tactic": 1}')
        for group in (f"{config * 10}/rank_0_0", f"torch_aot_compile/{config * 64}/rank_0_0"):
            directory = root / "torch_compile_cache" / group
            directory.mkdir(parents=True)
            (directory / "metadata").write_text(str(node))
            (directory / f"binary-{node}").write_text(str(node))
        helper.dump_warmup(cache, root / "jit", root / "startup", stream, allow_empty=False)
    stream.seek(0)
    target = tmp_path / "image"
    helper.install_warmup(target / "cache", target / "aot", target / "startup", stream)
    for node, config in enumerate(("a", "b")):
        for group in (f"{config * 10}/rank_0_0", f"torch_aot_compile/{config * 64}/rank_0_0"):
            directory = target / "torch_compile_cache" / group
            assert (directory / "metadata").read_text() == str(node)
            assert (directory / f"binary-{node}").read_text() == str(node)
            assert not (directory / "binary-2").exists()


@pytest.mark.parametrize("damaged", [None, "autotune", "startup", "jit", "compile"])
def test_manifest_checks_exported_artifacts(helper: ModuleType, tmp_path: Path, damaged: str | None) -> None:
    cache_root, aot_root, startup_root = (tmp_path / name for name in ("cache", "aot", "startup"))
    paths = {
        "autotune": cache_root / ("a" * 64) / "autotune_configs.json",
        "startup": startup_root / f"startup_plan_{'b' * 16}.json",
        "jit": aot_root / "checkpointing_ssu" / "checkpointing_ssu.so",
        "compile": tmp_path / "torch_compile_cache" / ("c" * 10) / "rank_0_0" / "model",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True)
        path.write_bytes(b"original")
    helper.write_manifest(cache_root, aot_root, startup_root, [paths["jit"]])
    if damaged:
        paths[damaged].write_bytes(b"changed artifact")
        with pytest.raises(ValueError, match="missing or changed"):
            helper.inspect_manifest(cache_root, aot_root, startup_root)
    else:
        assert helper.inspect_manifest(cache_root, aot_root, startup_root) == [paths["jit"]]
        original = json.loads((cache_root / "warmed-manifest.json").read_text())
        # Rewarming may reuse every AOT library without producing any new JIT files.
        helper.write_manifest(cache_root, aot_root, startup_root, [])
        rebuilt = json.loads((cache_root / "warmed-manifest.json").read_text())
        assert rebuilt["build_id"] != original["build_id"]
        assert rebuilt["jit_modules"] == original["jit_modules"]
        assert helper.inspect_manifest(cache_root, aot_root, startup_root) == [paths["jit"]]


def test_headless_warmup_exports_kernels_without_autotuning(helper: ModuleType, tmp_path: Path) -> None:
    (tmp_path / ".warmup-start").touch()
    library = tmp_path / "jit/version/100a/cached_ops/module/module.so"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"compiled")
    stream = io.BytesIO()
    helper.dump_warmup(tmp_path, tmp_path / "jit", tmp_path / "startup", stream, allow_empty=True)
    stream.seek(0)
    with tarfile.open(fileobj=stream) as archive:
        assert archive.getnames() == ["autotune.json", "jit/module.so"]


@pytest.mark.parametrize(
    "name",
    ["jit/../../outside.so", "startup/../../outside.json", "unexpected", "compile/../../outside"],
)
def test_warmup_archive_rejects_unexpected_paths(helper: ModuleType, tmp_path: Path, name: str) -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        entry = tarfile.TarInfo(name)
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    stream.seek(0)
    with pytest.raises(ValueError, match="Unexpected warmup archive entry|Invalid Torch compilation cache path"):
        helper.install_warmup(tmp_path / "cache", tmp_path / "aot", tmp_path / "startup", stream)
    assert not (tmp_path / "cache/.baked").exists()


def test_warmup_archive_rejects_links_and_invalid_plans(helper: ModuleType, tmp_path: Path) -> None:
    for is_link in (True, False):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            entry = tarfile.TarInfo(f"startup/startup_plan_{'a' * 16}.json")
            if is_link:
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/outside"
                archive.addfile(entry)
            else:
                entry.size = 2
                archive.addfile(entry, io.BytesIO(b"{}"))
        stream.seek(0)
        with pytest.raises(ValueError, match="Unexpected warmup archive entry|Invalid startup plan"):
            helper.install_warmup(tmp_path / "cache", tmp_path / "aot", tmp_path / "startup", stream)
        assert not (tmp_path / "cache/.baked").exists()


def test_baked_kernels_resolve_as_aot_before_load(
    helper: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "checkpointing_ssu.so"
    load = MagicMock()
    env = MagicMock()
    env.get_aot_path.return_value = library
    monkeypatch.setitem(helper.sys.modules, "tvm_ffi", MagicMock(load_module=load))
    monkeypatch.setitem(helper.sys.modules, "flashinfer", ModuleType("flashinfer"))
    monkeypatch.setitem(helper.sys.modules, "flashinfer.jit", MagicMock(env=env))
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "0")
    helper.verify_libraries([library])
    assert os.environ["FLASHINFER_DISABLE_JIT"] == "1"
    env.get_aot_path.assert_called_once_with("checkpointing_ssu")
    load.assert_called_once_with(str(library))

    env.get_aot_path.return_value = tmp_path / "wrong.so"
    load.reset_mock()
    with pytest.raises(ValueError, match="does not resolve baked module"):
        helper.verify_libraries([library])
    load.assert_not_called()
