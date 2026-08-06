# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import Request, Response
from omegaconf import OmegaConf
from pydantic import BaseModel

import nemo_gym.environment_scaffold as scaffold_module
from nemo_gym.environment_manifest import dump_manifest, load_manifest
from nemo_gym.environment_scaffold import ScaffoldConflictError, ScaffoldError, scaffold_environment
from nemo_gym.environment_validation import (
    infer_integration_profile,
    inspect_workload,
    manifest_composition_deltas,
    resolve_composition_mirror,
)
from nemo_gym.global_config import GlobalConfigDictParserConfig, StaticValidationConfigParser
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.verifier_fixture import VerifierFixtureError
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
)


def _add_reused_verifier_fixture(config_path: Path) -> Path:
    fixture_path = config_path.parent.parent / "tests" / "verifier_cases.jsonl"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        {
            "case": "full_reward",
            "request": {"responses_create_params": {"input": "correct"}},
            "expected_status": 200,
            "expected_reward": 1,
        },
        {
            "case": "zero_reward",
            "request": {"responses_create_params": {"input": "incorrect"}},
            "expected_status": 200,
            "expected_reward": 0,
        },
        {
            "case": "malformed",
            "request": {},
            "expected_status": 422,
        },
    ]
    fixture_path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    return fixture_path


@pytest.mark.parametrize(
    ("profile", "has_agent", "has_driver"),
    [
        ("stock-loop", False, False),
        ("measured-loop", True, False),
        ("external-loop", True, False),
        ("custom-driver", False, True),
    ],
)
def test_scaffolds_each_integration_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    has_agent: bool,
    has_driver: bool,
) -> None:
    name = profile.replace("-", "_")

    result = scaffold_environment(kind="environment", name=name, profile=profile, root=tmp_path)

    assert result.root == tmp_path.resolve()
    assert result.asset_dir == tmp_path / "environments" / name
    assert result.created
    manifest = load_manifest(result.asset_dir / "manifest.yaml")
    assert (result.asset_dir / "manifest.yaml").read_text() == dump_manifest(manifest)
    assert manifest.name == name
    assert manifest.kind.value == "environment"
    assert manifest.integration_profile.value == profile
    assert manifest.resources_server == name
    assert manifest.agent_server == (f"{name}_agent" if has_agent else "simple_agent")
    assert manifest.requires == ["text-model"]
    assert manifest.provides == ["verification"]
    assert manifest.datasets is not None
    assert manifest.datasets[0].jsonl_fpath == f"environments/{name}/data/example.jsonl"
    assert (tmp_path / "responses_api_agents" / f"{name}_agent").is_dir() is has_agent
    assert (result.asset_dir / "rollout_driver.py").is_file() is has_driver
    if has_agent:
        app_text = (tmp_path / "responses_api_agents" / f"{name}_agent" / "app.py").read_text()
        assert ("async def responses(" in app_text) is (profile == "measured-loop")
        assert ("async def run(" in app_text) is (profile == "external-loop")

    monkeypatch.chdir(tmp_path)
    config = OmegaConf.to_container(OmegaConf.load(result.asset_dir / "config.yaml"), resolve=False)
    assert isinstance(config, dict)
    assert bool(config.get("rollout_collection_driver")) is has_driver
    assert infer_integration_profile(config) == profile
    agent_config = next(iter(config[f"{name}_agent"]["responses_api_agents"].values()))
    assert "integration_profile" not in agent_config
    assert agent_config["requires"] == ["verification", "text-model"]

    resource_config = OmegaConf.load(tmp_path / "resources_servers" / name / "configs" / f"{name}.yaml")
    assert resource_config[f"{name}_resources_server"].resources_servers[name].provides == ["verification"]
    resource_app = (tmp_path / "resources_servers" / name / "app.py").read_text()
    assert "REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS" in resource_app

    generated_python = [path for path in result.files if path.suffix == ".py"]
    assert generated_python
    for path in generated_python:
        compile(path.read_text(), str(path), "exec")

    cases_path = tmp_path / "resources_servers" / name / "tests" / "verifier_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert [case["case"] for case in cases] == [
        "full_reward",
        "zero_reward",
        "malformed",
        "determinism_reseed",
    ]
    assert all(case["expected_status"] == "TODO" for case in cases)
    assert all("expected_reward" not in case for case in cases)
    assert cases[3]["reseed"] is True
    readme = (result.asset_dir / "README.md").read_text()
    assert readme.index(f"gym env test {name} --update-expected") < readme.index(
        f"gym env test {name}                    # read-only"
    )


@pytest.mark.parametrize(
    ("profile", "forged_profile"),
    [("measured-loop", "external-loop"), ("external-loop", "measured-loop")],
)
def test_generated_profile_inference_ignores_agent_config_relabel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    forged_profile: str,
) -> None:
    name = profile.replace("-", "_")
    result = scaffold_environment(kind="environment", name=name, profile=profile, root=tmp_path)
    monkeypatch.chdir(tmp_path)
    config = OmegaConf.to_container(OmegaConf.load(result.asset_dir / "config.yaml"), resolve=False)
    assert isinstance(config, dict)
    agent_config = next(iter(config[f"{name}_agent"]["responses_api_agents"].values()))
    agent_config["integration_profile"] = forged_profile

    assert infer_integration_profile(config) == profile


def test_scaffolds_benchmark_authoring_path_and_metadata(tmp_path: Path) -> None:
    result = scaffold_environment(
        kind="benchmark",
        name="answer_check",
        profile="stock-loop",
        root=tmp_path,
        metadata={
            "version": "1.2.3",
            "domain": "math",
            "description": "Checks concise arithmetic answers.",
            "licensing": "internal",
            "authors": ["Ada", "Grace"],
            "determinism": "seeded",
            "canonical_split": "held_out_test",
            "reward_range": [-1, 2],
        },
    )

    manifest = load_manifest(result.asset_dir / "manifest.yaml")
    assert manifest.version == "1.2.3"
    assert manifest.kind.value == "benchmark"
    assert manifest.canonical_split == "held_out_test"
    assert manifest.standard_prompt_config == "benchmarks/answer_check/prompt.yaml"
    assert manifest.datasets is not None
    assert manifest.datasets[0].type.value == "benchmark"
    assert manifest.datasets[0].prepare_script == "benchmarks/answer_check/prepare.py"
    assert manifest.datasets[0].prompt_config == "benchmarks/answer_check/prompt.yaml"
    assert manifest.reward.range == (-1.0, 2.0)

    source_row = json.loads((result.asset_dir / "data" / "source.jsonl").read_text())
    assert source_row == {"question": "What is 6 x 7?", "expected_answer": "42"}
    assert "responses_create_params" not in source_row
    assert "{question}" in (result.asset_dir / "prompt.yaml").read_text()
    assert (result.asset_dir / "prepare.py").is_file()

    resource_config = OmegaConf.load(tmp_path / "resources_servers" / "answer_check" / "configs" / "answer_check.yaml")
    assert resource_config.answer_check_resources_server.resources_servers.answer_check.domain == "math"
    assert (
        resource_config.answer_check_resources_server.resources_servers.answer_check.description
        == "Checks concise arithmetic answers."
    )
    cases_path = tmp_path / "resources_servers" / "answer_check" / "tests" / "verifier_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert all(case["expected_status"] == "TODO" for case in cases)


def test_generated_verifier_fixture_starts_failing_then_updates_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scaffold_environment(kind="environment", name="generated_fixture", root=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    generated_test = importlib.import_module("resources_servers.generated_fixture.tests.test_app")

    assert len(generated_test.CASES) == 4
    with pytest.raises(VerifierFixtureError, match="update-expected"):
        generated_test.test_verifier_fixture()
    monkeypatch.setenv("NEMO_GYM_UPDATE_EXPECTED", "1")
    generated_test.test_verifier_fixture()
    monkeypatch.setenv("NEMO_GYM_UPDATE_EXPECTED", "0")
    generated_test.test_verifier_fixture()
    response = generated_test.CASES[0]["request"]["response"]
    assert list(response["output"][0]) == ["id", "type", "role", "status", "content"]


async def test_default_scaffold_completes_one_rollout_through_agent_run(tmp_path: Path, monkeypatch) -> None:
    scaffold_environment(kind="environment", name="rollout_ready", root=tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    verifier_module = importlib.import_module("resources_servers.rollout_ready.app")
    verifier_app = verifier_module.create_offline_verifier_app(
        server_config={"entrypoint": "app.py"},
        instance_name="rollout_ready_resources_server",
    )

    class ClientResponse:
        def __init__(self, body: bytes | dict, *, status: int = 200, cookies=None) -> None:
            self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.ok = 200 <= status < 400
            self.cookies = dict(cookies or {})
            self.content = self

        async def read(self) -> bytes:
            return self._body

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=8081,
            entrypoint="app.py",
            name="rollout_ready_agent",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="rollout_ready_resources_server",
            ),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=server_client,
    )
    model_response = {
        "id": "stub-response",
        "created_at": 0,
        "model": "stub-model",
        "object": "response",
        "output": [
            {
                "id": "stub-message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "42", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }

    async def post(*, server_name, url_path, json: object, cookies=None):
        payload = json.model_dump(mode="json") if isinstance(json, BaseModel) else json
        if server_name == "rollout_ready_resources_server":
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=verifier_app),
                base_url="http://verifier",
            ) as client:
                result = await client.post(url_path, json=payload, cookies=cookies)
            return ClientResponse(result.content, status=result.status_code, cookies=result.cookies)
        if server_name == "rollout_ready_agent":
            assert url_path == "/v1/responses"
            cookie_header = "; ".join(f"{key}={value}" for key, value in dict(cookies or {}).items()).encode()
            request = Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/responses",
                    "path_params": {},
                    "query_string": b"",
                    "headers": [(b"cookie", cookie_header)] if cookie_header else [],
                }
            )
            result = await agent.responses(
                request,
                Response(),
                NeMoGymResponseCreateParamsNonStreaming.model_validate(payload),
            )
            return ClientResponse(result.model_dump(mode="json"), cookies=cookies)
        assert server_name == "policy_model"
        assert url_path == "/v1/responses"
        return ClientResponse(model_response)

    server_client.post = AsyncMock(side_effect=post)

    task = json.loads((tmp_path / "environments" / "rollout_ready" / "data" / "example.jsonl").read_text())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent.setup_webserver()),
        base_url="http://agent",
    ) as client:
        result = await client.post("/run", json=task)

    assert result.status_code == 200
    assert result.json()["reward"] == 1.0
    assert [call.kwargs["server_name"] for call in server_client.post.await_args_list] == [
        "rollout_ready_resources_server",
        "rollout_ready_agent",
        "policy_model",
        "rollout_ready_resources_server",
    ]


def test_scaffold_derives_model_capability_from_open_modality(tmp_path: Path) -> None:
    result = scaffold_environment(
        kind="environment",
        name="multimodal_fixture",
        root=tmp_path,
        metadata={"modality": "vision language"},
    )

    assert load_manifest(result.asset_dir / "manifest.yaml").requires == ["vision-language-model"]
    config = OmegaConf.load(result.asset_dir / "config.yaml")
    agent_config = config.multimodal_fixture_agent.responses_api_agents.simple_agent
    assert list(agent_config.requires) == ["verification", "vision-language-model"]


def test_generated_fixture_respects_lower_is_better_reward_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scaffold_environment(
        kind="environment",
        name="lower_is_better",
        root=tmp_path,
        metadata={"reward_range": [-1, 2], "higher_is_better": False},
    )

    cases_path = tmp_path / "resources_servers" / "lower_is_better" / "tests" / "verifier_cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert all(case["expected_status"] == "TODO" for case in cases)
    assert (
        "reward = -1 if output_text == body.expected_answer else 2"
        in (tmp_path / "resources_servers" / "lower_is_better" / "app.py").read_text()
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    generated_test = importlib.import_module("resources_servers.lower_is_better.tests.test_app")
    with pytest.raises(VerifierFixtureError, match="update-expected"):
        generated_test.test_verifier_fixture()
    monkeypatch.setenv("NEMO_GYM_UPDATE_EXPECTED", "1")
    generated_test.test_verifier_fixture()
    monkeypatch.setenv("NEMO_GYM_UPDATE_EXPECTED", "0")
    generated_test.test_verifier_fixture()
    updated = [json.loads(line) for line in cases_path.read_text().splitlines()]
    assert updated[0]["expected_reward"] == -1
    assert updated[1]["expected_reward"] == 2


def test_reuses_existing_verifier_without_scaffolding_a_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared_config = tmp_path / "resources_servers" / "shared" / "configs" / "strict.yaml"
    shared_config.parent.mkdir(parents=True)
    (shared_config.parent.parent / "app.py").write_text("# reused verifier entrypoint\n")
    shared_config.write_text(
        """\
shared_resources:
  resources_servers:
    exact_match:
      entrypoint: app.py
      domain: knowledge
      provides: [verification]
shared_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: shared_resources
      model_server:
        type: responses_api_models
        name: policy_model
      datasets: []
"""
    )
    _add_reused_verifier_fixture(shared_config)

    result = scaffold_environment(
        kind="benchmark",
        name="shared_consumer",
        profile="stock-loop",
        reuse_verifier="shared/strict",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )

    assert not (tmp_path / "resources_servers" / "shared_consumer").exists()
    assert all("resources_servers/shared_consumer" not in str(path) for path in result.created)
    manifest = load_manifest(result.asset_dir / "manifest.yaml")
    assert manifest.resources_server == "exact_match"
    assert manifest.agent_server == "simple_agent"
    config_text = (result.asset_dir / "config.yaml").read_text()
    assert "resources_servers/shared/configs/strict.yaml" in config_text
    assert "_inherit_from: shared_agent" in config_text
    assert "name: shared_resources" in config_text
    readme = (result.asset_dir / "README.md").read_text()
    assert "gym env test shared_consumer" in readme
    assert "--update-expected" not in readme

    # Inherited server configs are structured. The recipe must not inject a new
    # profile key into that structured node; the stock-loop classifier already
    # derives the profile from simple_agent.
    monkeypatch.chdir(tmp_path)
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(result.asset_dir / "config.yaml")]}),
    )
    resolved = StaticValidationConfigParser().parse_no_environment(initial)
    assert infer_integration_profile(resolved) == "stock-loop"


@pytest.mark.parametrize("profile", ["stock-loop", "measured-loop", "external-loop", "custom-driver"])
def test_reused_mcqa_composition_is_isolated_and_runnable_for_every_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    shared_config = tmp_path / "resources_servers" / "mcqa" / "configs" / "mcqa.yaml"
    shared_config.parent.mkdir(parents=True)
    (shared_config.parent.parent / "app.py").write_text("# reused verifier entrypoint\n")
    shared_config.write_text(
        """\
mcqa:
  resources_servers:
    mcqa:
      entrypoint: app.py
      provides: [verification]
      domain: knowledge
    unused_scorer:
      entrypoint: unused.py
mcqa_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      requires: [verification, text-model]
      resources_server:
        type: resources_servers
        name: mcqa
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
        - name: inherited_train
          type: train
          jsonl_fpath: resources_servers/mcqa/data/train.jsonl
unused_resources:
  resources_servers:
    unrelated_scorer:
      entrypoint: app.py
bundled_model:
  responses_api_models:
    bundled_model:
      entrypoint: app.py
rollout_collection_driver: legacy_mcqa.driver:run
"""
    )
    _add_reused_verifier_fixture(shared_config)
    name = f"reuse_{profile.replace('-', '_')}"

    result = scaffold_environment(
        kind="benchmark",
        name=name,
        profile=profile,
        reuse_verifier="mcqa",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )

    monkeypatch.chdir(tmp_path)
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(result.asset_dir / "config.yaml")]}),
    )
    resolved = StaticValidationConfigParser().parse_no_environment(initial)
    manifest = load_manifest(result.asset_dir / "manifest.yaml")
    mirror = resolve_composition_mirror(resolved)
    inspection = inspect_workload(resolved, strict_missing_datasets=True, manifest=manifest)

    expected_agent = f"{name}_agent" if profile in {"measured-loop", "external-loop"} else "simple_agent"
    assert infer_integration_profile(resolved) == profile
    assert mirror.resources_server == "mcqa"
    assert mirror.agent_server == expected_agent
    assert [dataset.name for dataset in mirror.datasets] == [name]
    assert manifest_composition_deltas(manifest, mirror) == ()
    assert {
        component.implementation for component in inspection.components if component.role == "resources_server"
    } == {"mcqa"}
    assert {component.implementation for component in inspection.components if component.role == "agent_server"} == {
        expected_agent
    }
    agent_component = next(component for component in inspection.components if component.role == "agent_server")
    assert agent_component.requires == ("verification", "text-model")
    assert resolved.get("rollout_collection_driver") == (
        f"benchmarks.{name}.rollout_driver:run_rollout_collection" if profile == "custom-driver" else None
    )
    cleanup = OmegaConf.to_container(OmegaConf.load(result.asset_dir / ".reuse_cleanup.yaml"), resolve=False)
    assert isinstance(cleanup, dict)
    assert cleanup["bundled_model"]["_delete_key"] == "responses_api_models"
    assert cleanup["unused_resources"]["_delete_key"] == "resources_servers"


def test_reused_verifier_from_another_search_root_uses_portable_config_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin_root = tmp_path / "plugin_root"
    scaffold_root = tmp_path / "consumer_root"
    scaffold_root.mkdir()
    shared_config = plugin_root / "resources_servers" / "portable_shared" / "configs" / "strict.yaml"
    shared_config.parent.mkdir(parents=True)
    shared_config.write_text(
        """\
shared_resources:
  resources_servers:
    exact_match:
      entrypoint: app.py
      domain: knowledge
      provides: [verification]
"""
    )
    _add_reused_verifier_fixture(shared_config)
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(plugin_root))

    result = scaffold_environment(
        kind="benchmark",
        name="portable_consumer",
        reuse_verifier="portable_shared/strict",
        metadata={"canonical_split": "test"},
        root=scaffold_root,
    )

    config_text = (result.asset_dir / "config.yaml").read_text()
    assert "  - resources_servers/portable_shared/configs/strict.yaml" in config_text
    assert str(plugin_root.resolve()) not in config_text

    # The generated root-relative reference remains loadable through Gym's
    # ordinary component search after changing into the consumer checkout.
    monkeypatch.chdir(scaffold_root)
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(result.asset_dir / "config.yaml")]}),
    )
    resolved = StaticValidationConfigParser().parse_no_environment(initial)
    assert resolved.shared_resources.resources_servers.exact_match.domain == "knowledge"


def test_repeated_scaffold_is_idempotent_and_conflicts_are_atomic(tmp_path: Path) -> None:
    first = scaffold_environment(kind="environment", name="safe_eval", root=tmp_path)
    second = scaffold_environment(kind="environment", name="safe_eval", root=tmp_path)

    assert second.created == ()
    assert set(second.existing) == set(first.files)

    readme = first.asset_dir / "README.md"
    manifest = first.asset_dir / "manifest.yaml"
    readme.unlink()
    manifest.write_text("# contributor-owned content\n")

    with pytest.raises(ScaffoldConflictError) as error:
        scaffold_environment(kind="environment", name="safe_eval", root=tmp_path)

    assert error.value.paths == (manifest,)
    assert manifest.read_text() == "# contributor-owned content\n"
    assert not readme.exists(), "preflight must detect every conflict before creating a missing file"


@pytest.mark.parametrize("symlink_kind", ["parent", "target"])
def test_scaffold_preflight_rejects_symlinked_write_paths_atomically(tmp_path: Path, symlink_kind: str) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    if symlink_kind == "parent":
        (root / "environments").symlink_to(outside, target_is_directory=True)
        protected = outside
    else:
        target_dir = root / "environments" / "symlink_guard"
        target_dir.mkdir(parents=True)
        protected = outside / "manifest.yaml"
        protected.write_text("outside sentinel\n")
        (target_dir / "manifest.yaml").symlink_to(protected)

    before = {path: path.read_bytes() for path in outside.rglob("*") if path.is_file()}
    with pytest.raises(ScaffoldError, match="symlink"):
        scaffold_environment(kind="environment", name="symlink_guard", root=root)

    assert {path: path.read_bytes() for path in outside.rglob("*") if path.is_file()} == before
    assert not (root / "resources_servers" / "symlink_guard").exists()
    if symlink_kind == "parent":
        assert not (protected / "symlink_guard").exists()


def test_scaffold_write_plan_rejects_outside_target_before_any_write(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    safe_target = root / "environments" / "safe" / "manifest.yaml"
    outside_target = tmp_path / "outside" / "escaped.txt"

    with pytest.raises(ScaffoldError, match="outside root"):
        scaffold_module._write_plan(
            root,
            root / "environments" / "safe",
            {safe_target: "safe\n", outside_target: "escaped\n"},
        )

    assert not safe_target.exists()
    assert not outside_target.exists()


def test_scaffold_rejects_symlink_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real_root"
    linked_root = tmp_path / "linked_root"
    real_root.mkdir()
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ScaffoldError, match="root must not be a symlink"):
        scaffold_environment(kind="environment", name="root_guard", root=linked_root)

    assert not (real_root / "environments" / "root_guard").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "recipe", "name": "valid"},
        {"kind": "environment", "name": "../escape"},
        {"kind": "environment", "name": "Uppercase"},
        {"kind": "environment", "name": "not-importable", "profile": "custom-driver"},
        {"kind": "environment", "name": "valid", "profile": "unknown-loop"},
        {"kind": "environment", "name": "valid", "metadata": {"typo": "value"}},
        {"kind": "environment", "name": "valid", "metadata": {"reward_range": [1, 1]}},
    ],
)
def test_rejects_invalid_scaffold_requests(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ScaffoldError):
        scaffold_environment(root=tmp_path, **kwargs)


def test_missing_reused_verifier_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldError, match="gym list resources-servers"):
        scaffold_environment(
            kind="environment",
            name="consumer",
            reuse_verifier="definitely_missing_scaffold_fixture",
            root=tmp_path,
        )


def test_reuse_verifier_rejects_legacy_resource_without_verification_capability(tmp_path: Path) -> None:
    legacy_config = tmp_path / "resources_servers" / "legacy_scorer" / "configs" / "legacy_scorer.yaml"
    legacy_config.parent.mkdir(parents=True)
    legacy_config.write_text(
        """\
legacy_resources:
  resources_servers:
    legacy_scorer:
      entrypoint: app.py
      domain: knowledge
"""
    )

    with pytest.raises(ScaffoldError, match=r"gym list components --provides verification"):
        scaffold_environment(
            kind="environment",
            name="broken_consumer",
            reuse_verifier="legacy_scorer",
            root=tmp_path,
        )

    assert not (tmp_path / "environments" / "broken_consumer").exists()


def test_reuse_verifier_rejects_provider_without_canonical_fixture(tmp_path: Path) -> None:
    config_path = tmp_path / "resources_servers" / "fixtureless_scorer" / "configs" / "fixtureless_scorer.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """\
fixtureless_resources:
  resources_servers:
    fixtureless_scorer:
      entrypoint: app.py
      domain: knowledge
      provides: [verification]
"""
    )

    with pytest.raises(
        ScaffoldError,
        match=r"canonical verifier fixture is missing.*tests/verifier_cases\.jsonl.*gym env test --resources-server",
    ):
        scaffold_environment(
            kind="environment",
            name="fixtureless_consumer",
            reuse_verifier="fixtureless_scorer",
            root=tmp_path,
        )

    assert not (tmp_path / "environments" / "fixtureless_consumer").exists()


def test_invalid_manifest_metadata_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldError, match="generated manifest is invalid"):
        scaffold_environment(
            kind="environment",
            name="invalid_metadata",
            metadata={"version": "not-semver"},
            root=tmp_path,
        )

    assert not (tmp_path / "environments" / "invalid_metadata").exists()
    assert not (tmp_path / "resources_servers" / "invalid_metadata").exists()


def test_benchmark_scaffold_requires_explicit_canonical_split(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldError, match="metadata.canonical_split"):
        scaffold_environment(kind="benchmark", name="missing_split", root=tmp_path)

    assert not (tmp_path / "benchmarks" / "missing_split").exists()
    assert not (tmp_path / "resources_servers" / "missing_split").exists()


def test_reuse_verifier_does_not_accidentally_reuse_a_measured_agent(tmp_path: Path) -> None:
    shared_config = tmp_path / "resources_servers" / "shared" / "configs" / "shared.yaml"
    shared_config.parent.mkdir(parents=True)
    shared_config.write_text(
        """\
shared_resources:
  resources_servers:
    exact_match:
      entrypoint: app.py
      domain: knowledge
      provides: [verification]
measured_agent:
  responses_api_agents:
    custom_harness:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: shared_resources
"""
    )
    _add_reused_verifier_fixture(shared_config)

    result = scaffold_environment(
        kind="environment",
        name="stock_consumer",
        profile="stock-loop",
        reuse_verifier="shared",
        root=tmp_path,
    )

    config = OmegaConf.to_container(OmegaConf.load(result.asset_dir / "config.yaml"), resolve=False)
    assert isinstance(config, dict)
    agent = config["stock_consumer_agent"]["responses_api_agents"]
    assert list(agent) == ["simple_agent"]
    assert load_manifest(result.asset_dir / "manifest.yaml").agent_server == "simple_agent"
