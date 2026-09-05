# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest

from nemo_gym.environment.manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment.validation import (
    EnvironmentValidationError,
    ResolvedComponent,
    ResolvedComposition,
    _base_name,
    _infer_profile,
    _only_delegates_to_super,
    _only_raises_not_implemented,
    _with_component_root,
    validate_environment,
)


def _manifest(*, kind: str = "environment", profile: str = "custom-gym-verifier") -> dict:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = {
        "name": "example",
        "type": "benchmark" if kind == "benchmark" else "example",
        "jsonl_fpath": f"{root}/data/example.jsonl",
        "num_repeats": 1,
    }
    manifest = {
        "name": "demo",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "math",
        "description": "A small exact-match evaluation.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["contributor"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "demo",
        "agent_server": "simple_agent",
        "datasets": [dataset],
        "grading_mode": "exact",
    }
    if profile in {"custom-gym-verifier", "custom-gym-agent-loop"}:
        manifest["model_server"] = "policy_model"
    if kind == "benchmark":
        dataset["prepare_script"] = f"{root}/prepare.py"
        manifest.update(canonical_split="test", standard_prompt_config=f"{root}/prompts/default.yaml")
    if profile == "external-rollout-driver":
        manifest["rollout_driver"] = "environments.demo.rollout_driver:run_rollout_collection"
    return manifest


def test_ast_helpers_and_config_outside_catalog(tmp_path: Path) -> None:
    tree = ast.parse(
        "class Agent(pkg.SimpleAgent):\n"
        "    async def responses(self):\n"
        "        'unavailable'\n"
        "        raise NotImplementedError()\n"
        "    async def run(self):\n"
        "        'fallback'\n"
        "        return await super().run()\n"
    )
    server = tree.body[0]
    assert isinstance(server, ast.ClassDef)
    responses, run = server.body

    assert _base_name(server.bases[0]) == "SimpleAgent"
    assert _only_raises_not_implemented(responses)
    assert _only_delegates_to_super(run)
    with _with_component_root(tmp_path / "config.yaml"):
        pass


def _config(*, kind: str = "environment", profile: str = "custom-gym-verifier") -> str:
    root = f"{'benchmarks' if kind == 'benchmark' else 'environments'}/demo"
    dataset = f"""      - name: example
        type: {"benchmark" if kind == "benchmark" else "example"}
        jsonl_fpath: {root}/data/example.jsonl
"""
    if kind == "benchmark":
        dataset += f"        prepare_script: {root}/prepare.py\n"
    model_server = (
        ""
        if profile == "external-rollout-driver"
        else """      model_server:
        type: responses_api_models
        name: policy_model
"""
    )
    rollout_driver = (
        "rollout_collection_driver: environments.demo.rollout_driver:run_rollout_collection\n"
        if profile == "external-rollout-driver"
        else ""
    )
    return f"""demo_resources:
  resources_servers:
    demo:
      entrypoint: app.py
      domain: math
      grading_mode: exact
demo_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: demo_resources
{model_server.rstrip()}
      datasets:
{dataset}{rollout_driver}"""


def _asset(tmp_path: Path, *, kind: str = "environment", profile: str = "custom-gym-verifier") -> Path:
    directory = tmp_path / ("benchmarks" if kind == "benchmark" else "environments") / "demo"
    (directory / "data").mkdir(parents=True)
    manifest_path = directory / "manifest.yaml"
    manifest_path.write_text(dump_manifest(_manifest(kind=kind, profile=profile)), encoding="utf-8")
    (directory / "config.yaml").write_text(_config(kind=kind, profile=profile), encoding="utf-8")

    if kind == "benchmark":
        row = {"question": "What is 1 + 1?", "expected_answer": "2"}
        (directory / "prompts").mkdir()
        (directory / "prompts/default.yaml").write_text("user: '{question}'\n", encoding="utf-8")
        (directory / "prepare.py").write_text(
            "from pathlib import Path\n\ndef prepare(output: Path) -> Path:\n    return output\n",
            encoding="utf-8",
        )
    else:
        row = {"responses_create_params": {"input": "What is 1 + 1?"}, "expected_answer": "2"}
    (directory / "data/example.jsonl").write_text(f"{json.dumps(row)}\n", encoding="utf-8")

    if profile == "external-rollout-driver":
        (directory / "rollout_driver.py").write_text(
            "def run_rollout_collection():\n    pass\n",
            encoding="utf-8",
        )
    return manifest_path


def _replace_manifest(path: Path, **changes: object) -> EnvironmentManifest:
    manifest = load_manifest(path)
    updated = manifest.model_copy(update=changes)
    path.write_text(dump_manifest(updated), encoding="utf-8")
    return manifest


def _custom_agent_asset(tmp_path: Path, *, profile: str, source: str) -> Path:
    manifest_path = _asset(tmp_path, profile=profile)
    changes: dict[str, object] = {"agent_server": "custom_agent"}
    if profile == "external-agent-loop":
        changes["model_server"] = "policy_model"
    _replace_manifest(manifest_path, **changes)

    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("    simple_agent:\n", "    custom_agent:\n"),
        encoding="utf-8",
    )
    agent_dir = tmp_path / "responses_api_agents/custom_agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "app.py").write_text(source, encoding="utf-8")
    return manifest_path


def test_reports_resolved_composition_and_declared_profile(tmp_path: Path) -> None:
    report = validate_environment(_asset(tmp_path))

    assert report.name == "demo"
    assert report.declared_profile == "custom-gym-verifier"
    assert report.inferred_profile == "custom-gym-verifier"
    assert report.profile_evidence == "selected agent implementation is simple_agent"
    assert report.warnings == ()
    assert "integration_profile" not in report.to_dict()
    assert report.datasets[0].rows == 1
    assert report.grading_mode == "exact"
    assert report.rollout_driver is None
    assert [(item.role, item.implementation) for item in report.components] == [
        ("resources_server", "demo"),
        ("agent_server", "simple_agent"),
        ("model_server", "runtime-selected"),
    ]
    assert report.components[0].entrypoint == "app.py"


@pytest.mark.parametrize(
    ("profile", "responses_body", "evidence"),
    [
        (
            "custom-gym-agent-loop",
            "return None",
            "agent overrides responses() with measured behavior",
        ),
        (
            "external-agent-loop",
            "raise NotImplementedError",
            "agent responses() raises NotImplementedError",
        ),
    ],
)
def test_infers_custom_agent_profile_from_entrypoint_ast(
    tmp_path: Path,
    profile: str,
    responses_body: str,
    evidence: str,
) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile=profile,
        source=(
            "raise RuntimeError('must not import')\n\n"
            "class CustomAgent:\n"
            "    async def responses(self):\n"
            f"        {responses_body}\n\n"
            "    async def run(self):\n"
            "        pass\n\n"
            "CustomAgent.run_webserver()\n"
        ),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == profile
    assert report.profile_evidence == evidence
    assert report.warnings == ()


@pytest.mark.parametrize(
    ("implementation", "profile"),
    [
        ("browsecomp_agent", "custom-gym-agent-loop"),
        ("harbor_agent", "external-agent-loop"),
        ("labbench2_vlm_agent", "custom-gym-verifier"),
    ],
)
def test_infers_representative_builtin_agent_profiles(implementation: str, profile: str) -> None:
    component = ResolvedComponent(
        role="agent_server",
        name=implementation,
        implementation=implementation,
        boundary="responses_api_agents",
        entrypoint=str(Path(__file__).parents[2] / "responses_api_agents" / implementation / "app.py"),
    )
    composition = ResolvedComposition(
        resources_server=None,
        agent_server=implementation,
        model_server=None,
        datasets=(),
        rollout_driver=None,
        grading_mode=None,
        components=(component,),
    )

    assert _infer_profile(composition)[0] == profile


def test_profile_mismatch_is_reported_as_a_warning(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-verifier",
        source=("class CustomAgent:\n    async def responses(self):\n        pass\n\nCustomAgent.run_webserver()\n"),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "custom-gym-agent-loop"
    assert report.warnings == (
        "Declared integration_profile 'custom-gym-verifier' does not match inferred profile 'custom-gym-agent-loop' "
        "(agent overrides responses() with measured behavior).",
    )


def test_inconclusive_profile_is_unknown_and_warns(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-agent-loop",
        source="class CustomAgent:\n    pass\n\nCustomAgent.run_webserver()\n",
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "unknown"
    assert report.profile_evidence == "agent server does not declare an inspectable responses() behavior"
    assert report.warnings == (
        "Could not infer integration_profile (agent server does not declare an inspectable responses() behavior); "
        "declared profile is 'custom-gym-agent-loop'.",
    )


def test_conditionally_external_agent_profile_is_unknown(tmp_path: Path) -> None:
    manifest_path = _custom_agent_asset(
        tmp_path,
        profile="custom-gym-agent-loop",
        source=(
            "class CustomAgent:\n"
            "    async def responses(self):\n"
            "        if self.external:\n"
            "            raise NotImplementedError\n"
            "        return None\n\n"
            "CustomAgent.run_webserver()\n"
        ),
    )

    report = validate_environment(manifest_path)

    assert report.inferred_profile == "unknown"
    assert report.profile_evidence == "agent responses() behavior depends on runtime configuration"


def test_multiple_dataset_agents_are_actionable(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + """
second_demo_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: demo_resources
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: duplicate
        type: example
        jsonl_fpath: environments/demo/data/example.jsonl
""",
        encoding="utf-8",
    )

    with pytest.raises(
        EnvironmentValidationError,
        match=r"exactly one dataset-bearing agent instance.*demo_agent \(simple_agent\).*second_demo_agent",
    ):
        validate_environment(manifest_path)


def test_benchmark_uses_root_prompt_without_executing_prepare(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    prepare_path = manifest_path.parent / "prepare.py"
    prepare_path.write_text(
        "from pathlib import Path\n\ndef prepare(source: Path) -> Path:\n    raise RuntimeError('must not execute')\n",
        encoding="utf-8",
    )

    report = validate_environment(manifest_path)

    assert report.datasets[0].type == "benchmark"
    assert report.datasets[0].prompt_config.endswith("prompts/default.yaml")


def test_malformed_benchmark_prompt_is_an_actionable_validation_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prompts/default.yaml").write_text("user: [broken\n", encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match="Could not materialize benchmark dataset"):
        validate_environment(manifest_path)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("async def prepare() -> Path:\n    pass\n", "synchronous prepare"),
        ("def prepare(\n", "Could not parse dataset prepare script"),
    ],
)
def test_prepare_contract_is_checked_statically(source: str, message: str, tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text(source, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_prepare_annotations_are_not_a_runtime_requirement(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, kind="benchmark")
    manifest_path.parent.joinpath("prepare.py").write_text("def prepare():\n    return 'output'\n", encoding="utf-8")

    validate_environment(manifest_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "other"}, "identity"),
        ({"kind": "benchmark", "canonical_split": "test", "standard_prompt_config": "prompt.yaml"}, "kind"),
    ],
)
def test_manifest_identity_matches_its_catalog_path(changes: dict, message: str, tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    raw = load_manifest(manifest_path).model_dump(mode="json", exclude_none=True)
    raw.update(changes)
    if changes.get("kind") == "benchmark":
        raw["datasets"][0].update(type="benchmark", prepare_script="prepare.py")
    manifest_path.write_text(dump_manifest(raw), encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_slash_qualified_name_matches_nested_catalog_path(tmp_path: Path) -> None:
    config_manifest_path = _asset(tmp_path)
    nested_manifest_path = tmp_path / "environments/acme/demo/manifest.yaml"
    nested_manifest_path.parent.mkdir(parents=True)
    manifest = load_manifest(config_manifest_path).model_copy(update={"name": "acme/demo"})
    nested_manifest_path.write_text(dump_manifest(manifest), encoding="utf-8")

    report = validate_environment(nested_manifest_path, config_manifest_path.with_name("config.yaml"))

    assert report.name == "acme/demo"


def test_config_defaults_to_sibling_and_must_exist(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    validate_environment(manifest_path)
    config_path.unlink()

    with pytest.raises(EnvironmentValidationError, match="config.yaml"):
        validate_environment(manifest_path)


def test_stale_mirrors_are_reported_and_sync_changes_only_them(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    original = load_manifest(manifest_path)
    stale_dataset = original.datasets[0].model_copy(update={"jsonl_fpath": "wrong.jsonl"})
    _replace_manifest(
        manifest_path,
        resources_server="wrong",
        agent_server="wrong",
        datasets=[stale_dataset],
    )

    with pytest.raises(EnvironmentValidationError, match="resources_server") as caught:
        validate_environment(manifest_path)
    assert '"jsonl_fpath": "wrong.jsonl"' in str(caught.value)

    report = validate_environment(manifest_path, sync=True)
    synchronized = load_manifest(manifest_path)
    assert report.synchronized_fields == ("resources_server", "agent_server", "datasets")
    assert synchronized.resources_server == "demo"
    assert synchronized.agent_server == "simple_agent"
    assert synchronized.description == original.description


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "is empty"),
        ("[broken\n", "Malformed JSON"),
        ("[]\n", "must contain a JSON object"),
        ('{"responses_create_params": {"input": 42}}\n', "responses_create_params.input"),
    ],
)
def test_dataset_row_errors_are_actionable(tmp_path: Path, content: str, message: str) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_text(content, encoding="utf-8")

    with pytest.raises(EnvironmentValidationError, match=message):
        validate_environment(manifest_path)


def test_invalid_dataset_encoding_is_an_actionable_error(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path)
    manifest_path.parent.joinpath("data/example.jsonl").write_bytes(b"\xff\n")

    with pytest.raises(EnvironmentValidationError, match="Could not read dataset"):
        validate_environment(manifest_path)


def test_custom_driver_is_checked_without_scaffolding(tmp_path: Path) -> None:
    manifest_path = _asset(tmp_path, profile="external-rollout-driver")
    report = validate_environment(manifest_path)
    assert report.rollout_driver == "environments.demo.rollout_driver:run_rollout_collection"
    assert report.inferred_profile == "external-rollout-driver"
    assert report.profile_evidence == "rollout_collection_driver is configured"
    assert report.warnings == ()

    driver_path = manifest_path.parent / "rollout_driver.py"
    driver_path.write_text("def other_function():\n    pass\n", encoding="utf-8")
    with pytest.raises(EnvironmentValidationError, match="Rollout driver.*was not found"):
        validate_environment(manifest_path)

    driver_path.unlink()
    with pytest.raises(EnvironmentValidationError, match="Rollout driver module was not found"):
        validate_environment(manifest_path)
