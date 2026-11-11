# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tomllib
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from responses_api_agents.codex_agent.app import CodexAgentConfig
from responses_api_agents.codex_agent.model_catalog import load_model_catalog
from responses_api_agents.codex_agent.prepare_model_catalog import (
    check_catalog_export,
    check_lock_change,
    reconcile_lock,
)
from responses_api_agents.codex_agent.tests.test_app import _config, _make_agent
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import close_body, seed
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import setup as setup


@pytest.fixture
def descriptor():
    # Synthetic contract data, deliberately unlike Codex's real prompts and capability defaults.
    return {
        "slug": "test-model",
        "display_name": "Test",
        "description": None,
        "supported_reasoning_levels": [{"effort": "custom", "description": "Test effort"}],
        "shell_type": "shell_command",
        "visibility": "none",
        "supported_in_api": True,
        "priority": 17,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "availability_nux": None,
        "upgrade": None,
        "base_instructions": "Synthetic instruction: preserve π and exact whitespace.\n",
        "model_messages": {"instructions_template": "Synthetic {{ personality }}", "instructions_variables": None},
        "include_skills_usage_instructions": True,
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "auto",
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 12345},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": False,
        "effective_context_window_percent": 93,
        "experimental_supported_tools": ["test"],
        "input_modalities": ["text"],
        "supports_search_tool": False,
        "use_responses_lite": False,
        "future_metadata": {"preserve": [1, "value"]},
    }


@pytest.fixture
def catalog(tmp_path, descriptor):
    path = tmp_path / "catalog.json"
    path.write_bytes((json.dumps({"models": [descriptor]}, ensure_ascii=False, indent=2) + "\r\n").encode())
    return path


@pytest.mark.parametrize("missing", ["base_instructions", "shell_type", "truncation_policy", "input_modalities"])
def test_rejects_incomplete_descriptor(catalog, descriptor, missing):
    del descriptor[missing]
    catalog.write_text(json.dumps({"models": [descriptor]}))
    with pytest.raises(ValidationError, match=missing):
        _config(model="test-model", model_catalog_json=catalog)


@pytest.mark.parametrize("model", [None, "test", "test-model-suffix", "backend/checkpoint"])
def test_requires_exact_cli_alias(catalog, model):
    with pytest.raises(ValidationError, match="exact entry"):
        _config(model=model, model_catalog_json=catalog)


def test_gym_default_alias_uses_cli_name_not_backend_name(catalog, descriptor):
    descriptor["slug"] = "gym-policy-model"
    catalog.write_text(json.dumps({"models": [descriptor]}))
    config = _config(model_server={"type": "responses_api_models", "name": "policy"}, model_catalog_json=catalog)
    assert config._model_catalog_unknown_context


@pytest.mark.parametrize(
    "field,value",
    [
        ("shell_type", "invented"),
        ("supports_parallel_tool_calls", "false"),
        ("context_window", True),
        ("truncation_policy", {"mode": "bytes"}),
    ],
)
def test_rejects_invalid_descriptor_types(catalog, descriptor, field, value):
    descriptor[field] = value
    catalog.write_text(json.dumps({"models": [descriptor]}))
    with pytest.raises(ValidationError, match=field):
        _config(model="test-model", model_catalog_json=catalog)


@pytest.mark.parametrize("data", [{"models": []}, {"models": [{"slug": "test-model"}]}, "bad json"])
def test_rejects_invalid_catalog(catalog, data):
    catalog.write_text(data if isinstance(data, str) else json.dumps(data))
    with pytest.raises(ValidationError):
        _config(model="test-model", model_catalog_json=catalog)


def test_rejects_duplicate_alias(catalog, descriptor):
    catalog.write_text(json.dumps({"models": [descriptor, descriptor]}))
    with pytest.raises(ValidationError, match="duplicate"):
        _config(model="test-model", model_catalog_json=catalog)


@pytest.mark.parametrize("name", ["model_context_window", "model_auto_compact_token_limit"])
@pytest.mark.parametrize("location", ["typed", "extra", "profile"])
def test_unknown_context_rejects_overrides(catalog, name, location):
    settings = {name: 32768}
    if location == "extra":
        settings = {"extra_config": settings}
    if location == "profile":
        settings = {"extra_config": {"profile": "selected", "profiles": {"selected": settings}}}
    with pytest.raises(ValidationError, match="Unknown catalog context conflicts"):
        _config(model="test-model", model_catalog_json=catalog, **settings)


@pytest.mark.parametrize("name", ["model", "model_catalog_json"])
def test_catalog_prevents_extra_config_from_changing_identity(catalog, name):
    with pytest.raises(ValidationError, match="typed fields"):
        _config(model="test-model", model_catalog_json=catalog, extra_config={name: "elsewhere"})


def test_unknown_context_rejects_catalog_threshold(catalog, descriptor):
    descriptor["auto_compact_token_limit"] = 1000
    catalog.write_text(json.dumps({"models": [descriptor]}))
    with pytest.raises(ValidationError, match="must also omit"):
        _config(model="test-model", model_catalog_json=catalog)


def test_finite_catalog_keeps_supported_overrides(catalog, descriptor):
    descriptor["max_context_window"] = 32768
    catalog.write_text(json.dumps({"models": [descriptor]}))
    config = _make_agent(model="test-model", model_catalog_json=catalog, model_context_window=16384)._build_config(
        "http://m"
    )
    assert config["model_context_window"] == 16384


def test_host_stages_exact_bytes_in_each_private_home(catalog, tmp_path):
    original = catalog.read_bytes()
    agent = _make_agent(model="test-model", model_catalog_json=catalog)
    catalog.write_text("changed after validation")
    with patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path):
        config = agent._build_config("http://model", developer_instructions="Keep the task prompt")
        first, second = agent._setup_codex_home(config), agent._setup_codex_home(config)
    assert first != second
    for home in (first, second):
        staged = tomllib.loads((home / "config.toml").read_text())
        assert staged["model_catalog_json"] == str(home / "models.json")
        assert (home / "models.json").read_bytes() == original
        assert staged["developer_instructions"] == "Keep the task prompt"
        assert "model_context_window" not in staged and "model_auto_compact_token_limit" not in staged
    assert "model_catalog_json" not in config


def test_host_catalog_write_failure_cleans_private_home(catalog, tmp_path, monkeypatch):
    agent = _make_agent(model="test-model", model_catalog_json=catalog)
    with patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path):
        monkeypatch.setattr(type(catalog), "write_bytes", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            agent._setup_codex_home(agent._build_config("http://model"))
    assert list((tmp_path / ".codex_agent").iterdir()) == []


def test_sandbox_stages_exact_catalog_and_preserves_prompt(setup, catalog, monkeypatch):
    agent, sandbox = setup
    original = catalog.read_bytes()
    uploaded_bytes = {}
    upload = sandbox.upload

    async def record_upload(source, destination):
        uploaded_bytes[destination] = Path(source).read_bytes()
        await upload(source, destination)

    monkeypatch.setattr(sandbox, "upload", record_upload)
    agent.config = CodexAgentConfig.model_validate(agent.config.model_dump() | {"model_catalog_json": catalog})
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        created.raise_for_status()
        session_id = created.json()["agent_session_id"]
        result = client.post(
            "/ng-rollout/codex-smoke-a2/v1/responses",
            json={"input": "Task stays unchanged", "instructions": "Developer text"},
        )
        result.raise_for_status()
        directory = agent._session_records[session_id].state.session.session_dir
        home = f"{directory}/home/.codex"
        config = tomllib.loads(sandbox.files[f"{home}/config.toml"])
        assert config["model_catalog_json"] == f"{home}/models.json"
        assert uploaded_bytes[config["model_catalog_json"]] == original
        assert config["developer_instructions"] == "Developer text"
        assert json.loads(sandbox.files[f"{directory}/input.json"])["prompt"] == "Task stays unchanged"
        assert not any(path.startswith("/app/") for path in sandbox.files)
        client.post("/v1/agent_sessions/close", json=close_body(session_id)).raise_for_status()
    sandbox.disconnect.assert_awaited_once()


def test_absent_catalog_preserves_existing_configuration(tmp_path):
    agent = _make_agent()
    assert agent.config.model_catalog_json is None
    with patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path):
        home = agent._setup_codex_home(agent._build_config("http://model"))
    assert not (home / "models.json").exists()
    assert "model_catalog_json" not in tomllib.loads((home / "config.toml").read_text())


def test_validation_retains_unknown_fields_and_original_bytes(catalog):
    text, unknown = load_model_catalog(catalog, model="test-model")
    assert unknown and text.encode() == catalog.read_bytes()
    assert json.loads(text)["models"][0]["future_metadata"] == {"preserve": [1, "value"]}


@pytest.fixture
def lockfile():
    return b"""version = 4
[[package]]
name = "codex-models-manager"
version = "0.0.0"
dependencies = ["serde"]
[[package]]
name = "serde"
version = "1.0.0"
source = "registry+https://example.invalid/index"
checksum = "original"
[[package]]
name = "git-package"
version = "0.0.0"
source = "git+https://example.invalid/repo#original"
"""


def test_reconciliation_only_changes_workspace_version(tmp_path, lockfile):
    (tmp_path / "Cargo.toml").write_text('[workspace.package]\nversion = "0.144.4"\n')
    package = tmp_path / "models-manager"
    package.mkdir()
    (package / "Cargo.toml").write_text('[package]\nname = "codex-models-manager"\nversion.workspace = true\n')
    (tmp_path / "Cargo.lock").write_bytes(lockfile)
    before, after = reconcile_lock(tmp_path)
    assert before != after
    updated = (tmp_path / "Cargo.lock").read_bytes()
    assert updated == lockfile.replace(b'version = "0.0.0"', b'version = "0.144.4"', 1)
    check_lock_change(lockfile, updated, local_packages={"codex-models-manager"})


@pytest.mark.parametrize(
    "old,new",
    [
        (b'checksum = "original"', b'checksum = "changed"'),
        (b'version = "1.0.0"', b'version = "2.0.0"'),
        (b"repo#original", b"repo#changed"),
        (b'dependencies = ["serde"]', b"dependencies = []"),
    ],
)
def test_reconciliation_rejects_dependency_changes(lockfile, old, new):
    changed = lockfile.replace(b'version = "0.0.0"', b'version = "0.144.4"', 1).replace(old, new)
    with pytest.raises(ValueError, match="dependency changes are forbidden"):
        check_lock_change(lockfile, changed, local_packages={"codex-models-manager"})


@pytest.mark.parametrize(
    "changed_field", [None, "base_instructions", "truncation_policy", "supports_parallel_tool_calls", "slug"]
)
def test_export_parity_allows_only_context_changes(descriptor, changed_field):
    baseline = descriptor | {"context_window": 272000, "max_context_window": 272000, "auto_compact_token_limit": None}
    modified = deepcopy(descriptor)
    if changed_field is not None:
        modified[changed_field] = "changed"
    args = (json.dumps(baseline).encode(), json.dumps({"models": [modified]}).encode())
    if changed_field is None:
        check_catalog_export(*args, model="test-model")
    else:
        with pytest.raises(ValueError):
            check_catalog_export(*args, model="test-model")
