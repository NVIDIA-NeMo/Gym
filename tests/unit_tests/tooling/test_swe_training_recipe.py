# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recipe contract checks without a model download, sandbox, Slurm job or training worker."""

import copy
import hashlib
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from scripts import check_swe_training_recipe as check
from scripts import prepare_swe_single_agent_turn_recipe as recipe

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from tests.unit_tests.tooling.test_swe_rows_to_harbor import REBENCH_ROW, parser_cache  # noqa: F401


def test_task_order_and_verifier_payload_are_preserved_without_prompt_leakage():
    rows = [copy.deepcopy(REBENCH_ROW) | {"instance_id": f"task-{index}"} for index in range(16)]
    original = copy.deepcopy(rows)
    harnesses = ["opencode", "hermes", "pi"]
    records = recipe.materialize_training_rows(rows, harnesses=harnesses, route="single-agent-turn")
    assert len(records) == 48
    assert [r["task_id"] for r in records] == [
        {"taskset": f"swe_rebench_{harness}", "task_id": row["instance_id"]} for row in rows for harness in harnesses
    ]
    for record, row in zip(records, (row for row in rows for _ in harnesses)):
        payload = record["task_input"]
        assert payload["responses_create_params"] == row["responses_create_params"]
        assert payload["task_data"] == {key: value for key, value in row.items() if key != "responses_create_params"}
        assert row["patch"] not in json.dumps(payload["responses_create_params"])
    assert rows == original


def test_old_run_uses_the_original_flat_agent_contract():
    row = copy.deepcopy(REBENCH_ROW)
    records = recipe.materialize_training_rows([row], harnesses=["opencode"], route="old-run")
    assert records == [
        row
        | {
            "agent_ref": {"type": "responses_api_agents", "name": "opencode_sandboxed_agent"},
            "task_source": "swe_rebench_opencode",
        }
    ]
    assert "task_input" not in records[0]


@pytest.mark.parametrize("route", ["old-run", "single-agent-turn", "harbor"])
def test_generated_gym_configuration_resolves_offline(tmp_path, route):
    config = OmegaConf.load(recipe.ROOT / "examples/training/swe_single_agent_turn.yaml")
    harnesses = ["opencode"] if route == "old-run" else list(recipe.HARNESSES)
    recipe.configure_gym(
        config,
        harnesses=harnesses,
        route=route,
        output=tmp_path,
        parser_cache=tmp_path / "parsers",
        prepared=SimpleNamespace(tasks=[], folder=tmp_path / "tasks"),
    )
    gym = OmegaConf.merge(
        config.env.nemo_gym,
        {
            "policy_model_name": "test-model",
            "policy_base_url": "http://127.0.0.1:9000/v1",
            "policy_api_key": "unused",
            "sandbox": {"opensandbox": {"connection": {"domain": "https://example.invalid", "api_key": "unused"}}},
        },
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=gym,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert resolved.environment_routing_mode == ("agent" if route == "old-run" else "taskset")
    if route != "harbor":
        assert resolved.swe_resources.resources_servers.swe_rebench.verification_log_dir == str(tmp_path / "verifier")
    if route == "old-run":
        agent = resolved.opencode_sandboxed_agent.responses_api_agents.opencode_sandboxed_agent
        assert agent.opencode_env.OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX == "1024"
        model = agent.opencode_config.provider.nemo_gym.models.dummy_model
        assert model.limit.output == 1024 and model.interleaved.field == "reasoning_content"
    for harness in harnesses:
        name = f"swe_rebench_{harness}_environment"
        impl = {"old-run": "legacy_agent", "single-agent-turn": "single_agent_turn", "harbor": "harbor"}[route]
        env = resolved[name].environment_servers[impl]
        assert env.default_episode_timeout_seconds == 1200
        assert env.agent_server.name == ("opencode_sandboxed_agent" if route == "old-run" else f"{harness}_agent")
        if route == "harbor":
            assert env.verifier_completion_key == "evaluation_completed"
            assert env.inconclusive_verification_retries == 1
            assert env.sandbox_resources_override.memory_mib == 30720


@pytest.fixture
def assets(tmp_path, parser_cache):
    prepared, root, model = (tmp_path / name for name in ("prepared", "source", "a" * 40))
    for directory in (prepared, root, model):
        directory.mkdir()
    (prepared / "config.yaml").write_text("seed: 42\n")
    (root / "capture.mjs").write_text("export const capture = true;\n")
    (model / "model.safetensors").write_bytes(b"fake weight bytes, never loaded")
    config = b'{"model_type":"test"}'
    (model / "config.json").write_bytes(config)
    parser_root = parser_cache / "SWE-rebench-V2"
    revision = subprocess.check_output(["git", "-C", str(parser_root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "files_sha256": {"config.yaml": check.file_sha256(prepared / "config.yaml")},
        "runtime_files_sha256": {"gym/capture.mjs": check.file_sha256(root / "capture.mjs")},
        "model": {
            "path": str(model),
            "revision": model.name,
            "files": {
                "model.safetensors": {"algorithm": "sha256", "digest": check.file_sha256(model / "model.safetensors")},
                "config.json": {
                    "algorithm": "git_blob_sha1",
                    "digest": hashlib.sha1(b"blob 21\0" + config).hexdigest(),
                },
            },
        },
        "parser": {"path": str(parser_root), "commit": revision},
    }
    return prepared, root, model, manifest


@pytest.mark.parametrize("mutation", [None, "config", "helper", "weights", "model-config", "missing-weights"])
def test_preflight_checks_source_helpers_and_actual_model_bytes(assets, mutation):
    prepared, root, model, manifest = assets
    if mutation == "config":
        (prepared / "config.yaml").write_text("seed: 0\n")
    elif mutation == "helper":
        (root / "capture.mjs").write_text("export const capture = false;\n")
    elif mutation == "weights":
        (model / "model.safetensors").write_bytes(b"different bytes")
    elif mutation == "model-config":
        (model / "config.json").write_text("{}")
    elif mutation == "missing-weights":
        (model / "model.safetensors").unlink()
    if mutation is None:
        check.check_files(prepared, {"gym": root}, manifest)
    else:
        with pytest.raises(ValueError):
            check.check_files(prepared, {"gym": root}, manifest)


def test_model_provenance_records_hub_hashes_without_downloading(monkeypatch, tmp_path):
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[
            SimpleNamespace(rfilename="config.json", lfs=None, blob_id="b" * 40),
            SimpleNamespace(rfilename="model.safetensors", lfs=SimpleNamespace(sha256="c" * 64)),
        ],
    )
    calls = []

    def metadata(repo, **kwargs):
        calls.append((repo, kwargs))
        return info

    monkeypatch.setattr(recipe, "HfApi", lambda: SimpleNamespace(model_info=metadata))
    manifest = recipe.model_manifest("example/model", info.sha, tmp_path)
    assert calls == [("example/model", {"revision": info.sha, "files_metadata": True})]
    assert manifest["files"]["config.json"] == {"algorithm": "git_blob_sha1", "digest": "b" * 40}
    assert manifest["files"]["model.safetensors"] == {"algorithm": "sha256", "digest": "c" * 64}
    assert not list(tmp_path.iterdir())


def test_launcher_dry_run_builds_one_node_submission_without_calling_slurm(tmp_path):
    # Bypass preflight only in this command-construction test; real launches always run it.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uv").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "uv").chmod(0o755)
    (bin_dir / "sbatch").write_text("#!/bin/sh\nexit 99\n")
    (bin_dir / "sbatch").chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "NEMO_RL_ROOT": str(tmp_path / "rl"),
        "CONFIG_PATH": str(tmp_path / "config.yaml"),
        "CONTAINER": str(tmp_path / "image.sqsh"),
        "CONTAINER_PROVENANCE": str(tmp_path / "image.json"),
        "SHARED_ROOT": "/",
        "SLURM_ACCOUNT": "test-account",
        "SLURM_PARTITION": "test-partition",
        "OPENSANDBOX_DOMAIN": "https://example.invalid",
        "OPENSANDBOX_API_KEY": "not-a-real-key",
    }
    result = subprocess.run(
        ["bash", str(recipe.ROOT / "scripts/launch_swe_single_agent_turn_training.sh"), "--dry-run"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--nodes=1" in result.stdout and "--exclusive" in result.stdout
    assert "run_grpo_single_controller.py" in result.stdout
    assert "check_swe_training_recipe.py" in result.stdout
    assert "not-a-real-key" not in result.stdout
