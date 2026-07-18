# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from benchmarks.osworld import assets
from benchmarks.osworld.assets import asset_specs_from_task, ensure_osworld_assets
from benchmarks.osworld.prepare import DEFAULT_INPUT, prepare, write_env


def test_prepare_validates_committed_example() -> None:
    assert prepare() == DEFAULT_INPUT.resolve()


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
