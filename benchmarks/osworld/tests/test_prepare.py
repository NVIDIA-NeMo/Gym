# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from benchmarks.osworld import assets
from benchmarks.osworld.assets import asset_specs_from_task, ensure_osworld_assets
from benchmarks.osworld.prepare import (
    DEFAULT_INPUT,
    GYM_SANDBOX_CONFIG,
    POINTER_AGENT_CONFIG,
    main,
    prepare,
    select_config_paths,
    write_env,
    write_task_shard,
    write_vm_snapshot_manifest,
)


def test_prepare_validates_committed_example() -> None:
    assert prepare() == DEFAULT_INPUT.resolve()


@pytest.mark.parametrize(
    "shard_args",
    [[], ["--num-shards", "1", "--shard-index", "0"]],
    ids=["defaults", "explicit-single-shard"],
)
def test_main_single_shard_keeps_the_original_input(
    monkeypatch, tmp_path: Path, shard_args: list[str]
) -> None:
    output = tmp_path / "results" / "rollouts.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare.py",
            "--input",
            str(DEFAULT_INPUT),
            "--output",
            str(output),
            "--skip-assets",
            "--no-env",
            *shard_args,
        ],
    )

    main()

    assert not list(tmp_path.rglob("input-shard-*.jsonl"))
    assert not list(tmp_path.rglob("*.manifest.json"))


def test_write_env_is_private_and_preserves_existing_file(tmp_path: Path) -> None:
    env_path = tmp_path / "env.yaml"
    output_path = tmp_path / "results" / "rollouts.jsonl"
    written = write_env(
        env_path,
        config_path=tmp_path / "config.yaml",
        input_jsonl=DEFAULT_INPUT,
        output_jsonl=output_path,
        policy_base_url="http://model.test/v1",
        policy_api_key="test-key",  # pragma: allowlist secret
        policy_model_name="test-model",
        num_samples_in_parallel=5,
        max_output_tokens=4096,
        temperature=0.6,
        top_p=0.95,
    )

    assert written is True
    assert env_path.stat().st_mode & 0o777 == 0o600
    contents = env_path.read_text(encoding="utf-8")
    assert "agent_name: osworld_simple_agent" in contents
    assert 'policy_base_url: "http://model.test/v1"' in contents
    assert str(DEFAULT_INPUT.resolve()) in contents
    assert str(output_path.resolve()) in contents
    assert "setup_cache_dir:" in contents
    assert "asset_input_jsonl:" in contents
    assert "num_samples_in_parallel: 5" in contents
    assert "concurrency: 5" in contents
    assert "max_output_tokens: 4096" in contents
    assert "temperature: 0.6" in contents
    assert "top_p: 0.95" in contents
    assert 'host: "127.0.0.1"' in contents
    assert "port: 11000" in contents

    env_path.write_text("user-owned: true\n", encoding="utf-8")
    assert (
        write_env(
            env_path,
            config_path=tmp_path / "config.yaml",
            input_jsonl=DEFAULT_INPUT,
            output_jsonl=output_path,
            policy_base_url="unused",
            policy_api_key="unused",  # pragma: allowlist secret
            policy_model_name="unused",
        )
        is False
    )
    assert env_path.read_text(encoding="utf-8") == "user-owned: true\n"


def test_prepare_composes_profile_and_backend_for_gym_env() -> None:
    paths = select_config_paths(profile="pointer", execution_backend="gym_sandbox")

    assert POINTER_AGENT_CONFIG.resolve() in paths
    assert paths[-1] == GYM_SANDBOX_CONFIG.resolve()


def test_write_task_shards_are_disjoint_complete_and_manifested(tmp_path: Path) -> None:
    source = tmp_path / "tasks.jsonl"
    rows = [
        {
            "responses_create_params": {"input": [{"role": "user", "content": f"task {index}"}]},
            "verifier_metadata": {
                "task_id": f"task-{index}",
                "osworld_task": {"id": f"task-{index}"},
            },
        }
        for index in range(7)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    shard_paths = [
        write_task_shard(source, tmp_path / f"shard-{index}.jsonl", num_shards=2, shard_index=index)
        for index in range(2)
    ]
    shard_ids = [
        [json.loads(line)["verifier_metadata"]["task_id"] for line in path.read_text().splitlines()]
        for path in shard_paths
    ]

    assert shard_ids == [["task-0", "task-2", "task-4", "task-6"], ["task-1", "task-3", "task-5"]]
    assert set(shard_ids[0]).isdisjoint(shard_ids[1])
    assert set().union(*map(set, shard_ids)) == {f"task-{index}" for index in range(7)}
    for index, path in enumerate(shard_paths):
        manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_sha256"]
        assert manifest["num_shards"] == 2
        assert manifest["shard_index"] == index
        assert manifest["total_tasks"] == 7
        assert manifest["shard_tasks"] == len(shard_ids[index])


def test_write_task_shard_rejects_invalid_index(tmp_path: Path) -> None:
    source = tmp_path / "tasks.jsonl"
    source.write_text(
        json.dumps({"verifier_metadata": {"task_id": "task-0", "osworld_task": {"id": "task-0"}}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shard_index"):
        write_task_shard(source, tmp_path / "out.jsonl", num_shards=2, shard_index=2)


def test_write_env_pins_vm_path_for_sandbox(tmp_path: Path) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"qcow2-base")
    env_path = tmp_path / "run" / "env.yaml"

    assert write_env(
        env_path,
        config_path=tmp_path / "config.yaml",
        input_jsonl=DEFAULT_INPUT,
        output_jsonl=tmp_path / "rollouts.jsonl",
        policy_base_url="http://model.test/v1",
        policy_api_key="local",
        policy_model_name="model",
        execution_backend="gym_sandbox",
        vm_path=vm_path,
        head_port=21001,
    )

    contents = env_path.read_text(encoding="utf-8")
    assert f'vm_path: "{vm_path.resolve()}"' in contents
    assert "port: 21001" in contents


def test_write_env_rejects_sandbox_without_explicit_vm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit vm_path"):
        write_env(
            tmp_path / "env.yaml",
            config_path=tmp_path / "config.yaml",
            input_jsonl=DEFAULT_INPUT,
            output_jsonl=tmp_path / "rollouts.jsonl",
            policy_base_url="http://model.test/v1",
            policy_api_key="local",
            policy_model_name="model",
            execution_backend="gym_sandbox",
        )


def test_vm_snapshot_manifest_is_content_addressed(tmp_path: Path) -> None:
    vm_path = tmp_path / "Ubuntu.qcow2"
    vm_path.write_bytes(b"fixed-qcow2")
    manifest_path = tmp_path / "vm-snapshot.json"

    manifest = write_vm_snapshot_manifest(
        manifest_path,
        vm_path=vm_path,
        execution_backend="gym_sandbox",
    )

    assert manifest["snapshot_id"] == f"sha256:{manifest['sha256']}"
    assert manifest["mount_mode"] == "read-only"
    assert manifest["reset_semantics"] == "close-and-recreate-from-base"
    assert manifest["live_ram_snapshot_supported"] is False
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_asset_specs_cover_setup_postconfig_and_evaluator_cloud_files() -> None:
    task = {
        "config": [
            {"type": "download", "parameters": {"files": [{"url": "https://hf.test/input", "path": "/tmp/in"}]}}
        ],
        "evaluator": {
            "postconfig": [
                {
                    "type": "download",
                    "parameters": {"files": [{"url": "https://hf.test/post", "path": "/tmp/post.bin"}]},
                }
            ],
            "expected": {
                "type": "cloud_file",
                "multi": True,
                "path": ["https://hf.test/a", "https://hf.test/b"],
                "dest": ["gold/a.bin", "gold/b.bin"],
            },
        },
    }

    specs = {spec.url: spec.cache_names for spec in asset_specs_from_task(task)}
    assert len(specs) == 4
    assert specs["https://hf.test/a"] == ("gold/a.bin",)
    assert specs["https://hf.test/b"] == ("gold/b.bin",)
    assert specs["https://hf.test/input"][0].endswith("_in")
    assert specs["https://hf.test/post"][0].endswith("_post.bin")


def test_ensure_osworld_assets_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "hf-cache" / "input.bin"
    source.parent.mkdir()
    source.write_bytes(b"asset")
    task = {
        "id": "task-1",
        "config": [
            {
                "type": "download",
                "parameters": {"files": [{"url": "https://hf.test/input", "path": "/home/user/input.bin"}]},
            }
        ],
        "evaluator": {"expected": {"type": "cloud_file", "path": "https://hf.test/input", "dest": "gold.bin"}},
    }
    input_jsonl = tmp_path / "tasks.jsonl"
    input_jsonl.write_text(json.dumps({"verifier_metadata": {"osworld_task": task}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(assets, "_download_asset", lambda *_args, **_kwargs: source)

    cache_dir = tmp_path / "setup-cache"
    first = ensure_osworld_assets(input_jsonl, cache_dir)
    second = ensure_osworld_assets(input_jsonl, cache_dir)

    assert first.asset_count == 1
    assert first.materialized_count == 2
    assert second.materialized_count == 0
    assert (cache_dir / "task-1" / "gold.bin").read_bytes() == b"asset"
    assert len(list((cache_dir / "task-1").glob("*_input.bin"))) == 1
