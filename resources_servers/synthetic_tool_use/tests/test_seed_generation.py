# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from resources_servers.synthetic_tool_use.assets import generation_asset_hashes, load_generation_assets
from resources_servers.synthetic_tool_use.common.artifacts import RunArtifactStore
from resources_servers.synthetic_tool_use.common.clients import GeneratedText
from resources_servers.synthetic_tool_use.common.models import (
    DomainApplication,
    DomainCandidate,
    DomainStageConfig,
    GenerationMetadata,
    ModelRoleConfig,
    PolicyToolsStageConfig,
    ScenarioStageConfig,
    SeedGenerationConfig,
    StageState,
    normalize_domain_name,
)
from resources_servers.synthetic_tool_use.common.parsing import (
    extract_tag,
    parse_json_or_jsonl,
    parse_json_value,
)
from resources_servers.synthetic_tool_use.common.quality import (
    ArtifactValidationError,
    validate_tools,
)
from resources_servers.synthetic_tool_use_domain_generation.assets import DOMAIN_PROMPT_PATH, load_domain_prompt
from resources_servers.synthetic_tool_use_domain_generation.stage import DomainGenerationStage
from resources_servers.synthetic_tool_use_policy_tool_generation import profiles as policy_profiles
from resources_servers.synthetic_tool_use_policy_tool_generation.stage import PolicyToolsGenerationStage
from resources_servers.synthetic_tool_use_scenario_generation import assets as scenario_assets
from resources_servers.synthetic_tool_use_scenario_generation.schema import (
    generated_scenario_schema_json,
    scenario_schema_json,
)
from resources_servers.synthetic_tool_use_scenario_generation.stage import (
    ScenarioGenerationStage,
    scope_schedule,
)
from resources_servers.synthetic_tool_use_simulation.scripts.build_synthetic_tool_use_dataset import (
    build_sample_dataset,
)


POLICY_PROMPTS_DIR = policy_profiles.PACKAGE_DIR / "prompts"
GOLDEN_REFERENCES_DIR = policy_profiles.PACKAGE_DIR / "references" / "golden_policies"
SCENARIO_PROMPTS_DIR = scenario_assets.PROMPTS_DIR
PIPELINE_DIR = Path(__file__).resolve().parents[1]
RESOURCE_SERVERS_DIR = PIPELINE_DIR.parent

PROMPT_SHA256 = {
    "customer_scenario_collection_schema.json": "b0a4d8385fbda3b77d8d9626bc5998f1d5e62a2f75a225c3f6198a663aa5991e",
    "domain_generation.txt": "f90c8b57ed564fb8c918b4d2c2d9dc4da537285fe8bcc56500db168a54200211",
    "cohesion_judge.txt": "a0070c4d9688df277c5f65e2b3a22112a1dea2b4f02c173ff4758529f4d912d9",
    "general_policy.txt": "4da0ed416c152dffd975b46e480c3e83843b6bfe037c2222dc70cb5e471bad88",
    "general_policy_refine.txt": "cbe16bf60332a12a03083758fe31dafd5cb779abebf29214ef1315de526b8c1b",
    "general_tools.txt": "5b8af59584760ce76523286ef17865f5b195385afeb3494fcd867cd41d90b17a",
    "golden_judge.txt": "ca5ce1481ff65e3f913dce4f225598f50180241edac0f6317dfaa6a896441977",
    "proactive_policy.txt": "00c93189b9411fabde7439a5fdf36d14412e473e6b8851824043a7e4c74109be",
    "proactive_policy_refine.txt": "544fce4d3fc534634f04db0e2e01d7a5691612d01b6f3abd9bc27d039e88d9d5",
    "proactive_tools.txt": "2fd77edd5d5fdc99beceb58aede864b2a41ef3fecef102eb5b0a3c7bc1a7895d",
    "tools_refine.txt": "38965e0f6909b3799863153aab6980ef694da5128145a5aa4ff88bbbe2782738",
    "scenario_system.txt": "9b799cf3bd1110e9d637ca4db718f68d626c90943ecf5bcbf2ad808a511f3d90",
    "scenario_user.txt": "cad5e03e2ad1365c919da4520af339a0616ec0954a29e415779244df13b73cbd",
}


class QueueGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages: list[dict[str, str]]) -> GeneratedText:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("fake generator has no responses left")
        text = self.responses.pop(0)
        return GeneratedText(text=text, raw_response={"text": text}, provider_attempts=1)


def role(model: str) -> ModelRoleConfig:
    return ModelRoleConfig(model=model, base_url="http://model.invalid/v1", api_key_env="TEST_API_KEY")


def config(tmp_path: Path) -> SeedGenerationConfig:
    return SeedGenerationConfig(
        run_name="test-run",
        output_dir=tmp_path / "run",
        generation_profile="proactive",
        source_name="test_source",
        random_seed=7,
        domain_model=role("domain-model"),
        policy_tools_model=role("policy-model"),
        judge_model=role("judge-model"),
        scenario_model=role("scenario-model"),
        domains=DomainStageConfig(request_count=1, semantic_attempts=1),
        policy_tools=PolicyToolsStageConfig(
            semantic_attempts=1,
            refine=True,
            judge_enabled=True,
            judge_votes=1,
            golden_comparison_enabled=False,
        ),
        scenarios=ScenarioStageConfig(
            scenarios_per_request=1,
            request_count_per_domain=2,
            outside_policy_scope_fraction=0.5,
            semantic_attempts=1,
            scenarios_per_file=10,
        ),
    )


def tool() -> dict:
    return {
        "name": "lookup_account",
        "doc": "Look up an account.",
        "params": {
            "type": "object",
            "properties": {"account_id": {"type": "string"}},
            "required": ["account_id"],
        },
        "returns": {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        },
    }


def scenario(reason: str) -> dict:
    return {
        "customer_persona": "A concise customer",
        "reason_for_contact": reason,
        "customer_details": "Account ID A-123",
        "unknown_info": "Current account status",
        "task_instructions": "Provide the account ID when asked and seek a resolution.",
    }


def domain_candidate(index: int) -> DomainCandidate:
    return DomainCandidate(
        name=f"Domain {index}",
        applications=[DomainApplication(function=f"Handle task {index}")],
        generation_profile="proactive",
        request_index=0,
        candidate_index=index,
        generation=GenerationMetadata(model="domain-model"),
    )


def expected_timestamp(rng: random.Random) -> str:
    timezones = [
        ("America/New_York", 0.47),
        ("America/Chicago", 0.33),
        ("America/Denver", 0.06),
        ("America/Los_Angeles", 0.13),
        ("America/Anchorage", 0.003),
        ("Pacific/Honolulu", 0.004),
        ("America/Phoenix", 0.01),
    ]
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 12, 31, 23, 59, 59)
    timestamp = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))
    timezone_name = rng.choices(
        [name for name, _ in timezones],
        weights=[weight for _, weight in timezones],
        k=1,
    )[0]
    return timestamp.replace(tzinfo=ZoneInfo(timezone_name)).strftime("%Y-%m-%d %H:%M:%S %Z")


def expected_pair(policy: str, tools: str, index: int) -> str:
    return f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>\n<tools_{index}>\n{tools}\n</tools_{index}>"


def test_parsing_supports_tags_fences_arrays_and_jsonl() -> None:
    assert extract_tag("before <policy>Policy text</policy> after", "policy") == "Policy text"
    assert parse_json_value('```json\n{"accepted": true}\n```')["accepted"] is True
    assert parse_json_or_jsonl(json.dumps([tool()])) == [tool()]
    second_tool = {**tool(), "name": "second"}
    assert parse_json_or_jsonl(json.dumps(tool()) + "\n" + json.dumps(second_tool))[1]["name"] == "second"


def test_tool_validation_rejects_duplicate_names() -> None:
    with pytest.raises(ArtifactValidationError, match="duplicate tool name"):
        validate_tools([tool(), tool()])


def test_domain_name_normalization_produces_underscore_slug() -> None:
    assert normalize_domain_name("E-commerce") == "e_commerce"
    assert normalize_domain_name(" E commerce ") == "e_commerce"
    assert normalize_domain_name("General Customer Support") == "general_customer_support"


def test_generation_assets_are_stable_and_loadable() -> None:
    prompt_paths = {
        "customer_scenario_collection_schema.json": (
            SCENARIO_PROMPTS_DIR / "customer_scenario_collection_schema.json"
        ),
        "domain_generation.txt": DOMAIN_PROMPT_PATH,
        "scenario_system.txt": SCENARIO_PROMPTS_DIR / "scenario_system.txt",
        "scenario_user.txt": SCENARIO_PROMPTS_DIR / "scenario_user.txt",
        **{
            name: POLICY_PROMPTS_DIR / name
            for name in PROMPT_SHA256
            if name
            not in {
                "customer_scenario_collection_schema.json",
                "domain_generation.txt",
                "scenario_system.txt",
                "scenario_user.txt",
            }
        },
    }
    for name, expected_hash in PROMPT_SHA256.items():
        assert hashlib.sha256(prompt_paths[name].read_bytes()).hexdigest() == expected_hash

    digest = hashlib.sha256()
    for path in sorted(GOLDEN_REFERENCES_DIR.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    assert digest.hexdigest() == "4878085c98451c785b8f71793611c14992836403c74dd8aaf74d0ab8a1c83124"

    schema = scenario_schema_json()
    assert schema == generated_scenario_schema_json()
    assert len(schema.encode()) == 1877
    assert hashlib.sha256(schema.encode()).hexdigest() == PROMPT_SHA256["customer_scenario_collection_schema.json"]
    scenario_prompts = scenario_assets.load_scenario_prompts()
    rendered_user = scenario_prompts.user.format(
        scenario_count=80,
        scenarios_schema=schema,
    )
    assert hashlib.sha256(rendered_user.encode()).hexdigest() == (
        "75e348ec36ea5b3d321362279b5ebd0e9a2940a5c00a456ece1ebb514ca02b65"
    )

    assert load_domain_prompt() == DOMAIN_PROMPT_PATH.read_text().strip()
    assert scenario_prompts.system == (SCENARIO_PROMPTS_DIR / "scenario_system.txt").read_text().strip()
    assert scenario_prompts.user == (SCENARIO_PROMPTS_DIR / "scenario_user.txt").read_text().strip()

    shared_policy_assets = {
        "tools_refine_prompt": "tools_refine.txt",
        "cohesion_judge_prompt": "cohesion_judge.txt",
        "golden_judge_prompt": "golden_judge.txt",
    }
    for profile_name in ("general", "proactive"):
        profile = policy_profiles.load_profile(profile_name)
        profile_assets = {
            **shared_policy_assets,
            "policy_prompt": f"{profile_name}_policy.txt",
            "tools_prompt": f"{profile_name}_tools.txt",
            "policy_refine_prompt": f"{profile_name}_policy_refine.txt",
        }
        for field_name, asset_name in profile_assets.items():
            assert getattr(profile, field_name) == (POLICY_PROMPTS_DIR / asset_name).read_text().strip()

        hashes = generation_asset_hashes(profile_name)
        for asset_name, path in prompt_paths.items():
            if f"{'proactive' if profile_name == 'general' else 'general'}_" in asset_name:
                continue
            expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            assert expected_hash in hashes.values()


def test_generation_components_have_explicit_ownership_boundaries() -> None:
    component_names = (
        "synthetic_tool_use_domain_generation",
        "synthetic_tool_use_policy_tool_generation",
        "synthetic_tool_use_scenario_generation",
    )
    for component_name in component_names:
        component_dir = RESOURCE_SERVERS_DIR / component_name
        source = "\n".join(path.read_text(encoding="utf-8") for path in component_dir.glob("*.py"))
        for other_name in component_names:
            if other_name != component_name:
                assert f"resources_servers.{other_name}" not in source

    runtime_dir = RESOURCE_SERVERS_DIR / "synthetic_tool_use_simulation"
    assert not (runtime_dir / "seed_generation").exists()
    assert sorted(path.name for path in (PIPELINE_DIR / "configs").glob("*.yaml")) == [
        "general.yaml",
        "proactive.yaml",
    ]


def test_scope_schedule_is_deterministic() -> None:
    first = scope_schedule(10, 0.3, seed="test")
    second = scope_schedule(10, 0.3, seed="test")
    assert first == second
    assert first == [False, False, True, False, True, True, False, False, True, False]


@pytest.mark.parametrize("profile_name", ["general", "proactive"])
@pytest.mark.asyncio
async def test_policy_tool_request_rendering(profile_name: str, tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    generation_config.generation_profile = profile_name
    generation_config.policy_tools.judge_votes = 3
    generation_config.policy_tools.golden_comparison_enabled = True
    generation_config.policy_tools.golden_comparison_count = 4
    profile = policy_profiles.load_profile(profile_name)
    store = RunArtifactStore.create(generation_config, generation_asset_hashes(profile_name))
    domain = DomainCandidate(
        name="Help (& Care) / Café",
        applications=[DomainApplication(function="Resolve an account issue")],
        generation_profile=profile_name,
        request_index=0,
        candidate_index=0,
        generation=GenerationMetadata(model="domain-model"),
    )
    store.register_domains([domain])

    initial_policy = "Authenticate the customer before lookup."
    final_policy = "Authenticate the customer, then resolve the request."
    initial_tool = {**tool(), "doc": "Look up café account status."}
    final_tool = {**tool(), "doc": "Resolve café account status."}
    generator = QueueGenerator(
        [
            f"<policy>{initial_policy}</policy>",
            f"<tools>{json.dumps(initial_tool, ensure_ascii=False)}</tools>",
            f"<policy>{final_policy}</policy>",
            f"<tools>{json.dumps(final_tool, ensure_ascii=False)}</tools>",
        ]
    )
    judge = QueueGenerator(
        [
            "<judgment>true</judgment>",
            "<judgment>true</judgment>",
            "<judgment>true</judgment>",
            "<judgment>0</judgment>",
            "<judgment>0</judgment>",
            "<judgment>1</judgment>",
            "<judgment>1</judgment>",
        ]
    )
    golden_pairs = [(f"Golden policy {index}", json.dumps({"name": f"golden_tool_{index}"})) for index in range(4)]
    stage = PolicyToolsGenerationStage(generation_config, profile, store, generator, judge)
    stage.golden_pairs = golden_pairs
    await stage.run()
    assert store.load_manifest().domains[0].stages["policy_tools"].state == StageState.COMPLETE

    domain_name = "Help__Care_-_Café"
    rng = random.Random(f"{generation_config.random_seed}:{domain.domain_id}:1:policy-tools-v1")
    timestamp = expected_timestamp(rng)
    shuffled_pairs = list(golden_pairs)
    rng.shuffle(shuffled_pairs)
    policy_tool_references = "".join(
        expected_pair(policy, tools, index) for index, (policy, tools) in enumerate(shuffled_pairs)
    )
    expected_policy_prompt = (
        profile.policy_prompt.format(
            domain=domain_name,
            timestamp=timestamp,
        )
        + policy_tool_references
    )
    expected_tools_prompt = (
        profile.tools_prompt.format(domain=domain_name, policy=initial_policy)
        + policy_tool_references
        + f"\n\n<policy>{initial_policy}</policy>"
    )

    shuffled_policy_pairs = list(golden_pairs)
    rng.shuffle(shuffled_policy_pairs)
    policy_references = "".join(
        f"\n\n<policy_{index}>\n{policy}\n</policy_{index}>" for index, (policy, _) in enumerate(shuffled_policy_pairs)
    )
    expected_policy_refine_prompt = profile.policy_refine_prompt.format(
        domain=domain_name,
        policy=initial_policy,
        reference_policies=policy_references,
    )
    unused_tools_reference_shuffle = list(golden_pairs)
    rng.shuffle(unused_tools_reference_shuffle)
    initial_tools = json.dumps(initial_tool)
    expected_tools_refine_prompt = profile.tools_refine_prompt.format(
        domain=domain_name,
        policy=final_policy,
        tools=initial_tools,
    )
    assert generator.calls == [
        [{"role": "user", "content": expected_policy_prompt}],
        [{"role": "user", "content": expected_tools_prompt}],
        [{"role": "user", "content": expected_policy_refine_prompt}],
        [{"role": "user", "content": expected_tools_refine_prompt}],
    ]

    final_tools = json.dumps(final_tool)
    cohesion_prompt = profile.cohesion_judge_prompt.format(
        domain=domain_name,
        policy=final_policy,
        tools=final_tools,
    )
    shuffled_goldens = list(golden_pairs)
    rng.shuffle(shuffled_goldens)
    golden_prompts = [
        profile.golden_judge_prompt
        + expected_pair(golden_policy, golden_tools, 0)
        + expected_pair(final_policy, final_tools, 1)
        for golden_policy, golden_tools in shuffled_goldens[:2]
    ]
    assert judge.calls == [
        *[[{"role": "user", "content": cohesion_prompt}] for _ in range(3)],
        *[[{"role": "user", "content": prompt}] for prompt in [*golden_prompts, *golden_prompts]],
    ]


@pytest.mark.parametrize("outside_scope", [False, True])
@pytest.mark.asyncio
async def test_scenario_message_rendering(outside_scope: bool, tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    generation_config.scenarios.scenarios_per_request = 80
    assets = load_generation_assets("proactive")
    store = RunArtifactStore.create(generation_config, generation_asset_hashes("proactive"))
    domain = domain_candidate(0)
    store.register_domains([domain])
    generator = QueueGenerator([json.dumps({"scenarios": [scenario("Check account status")]})])
    stage = ScenarioGenerationStage(generation_config, assets.scenarios, store, generator)
    schema = scenario_schema_json()

    generated = await stage._run_request(
        domain_id=domain.domain_id,
        domain_name=domain.name,
        policy="Policy formatting remains here.\nSecond line.",
        schema=schema,
        request_index=0,
        outside_scope=outside_scope,
    )

    expected_system = assets.scenarios.system.format(
        domain_policy="Policy formatting remains here.\nSecond line.",
        policy_scope_instruction="does not cover" if outside_scope else "covers",
    )
    expected_user = assets.scenarios.user.format(
        scenario_count=80,
        scenarios_schema=schema,
    )
    assert generator.calls == [
        [
            {"role": "system", "content": expected_system},
            {"role": "user", "content": expected_user},
        ]
    ]
    assert generated is not None
    assert generated[0].representative_domain == domain.name
    assert generated[0].outside_policy_scope is outside_scope


def test_domain_directories_expand_to_maximum_index_width(tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    store = RunArtifactStore.create(generation_config, generation_asset_hashes("proactive"))

    store.register_domains([domain_candidate(index) for index in range(10)])
    assert sorted(path.name for path in store.domains_dir.iterdir()) == [str(index) for index in range(10)]

    store.register_domains([domain_candidate(index) for index in range(10, 50)])
    expected = [f"{index:02d}" for index in range(50)]
    assert sorted(path.name for path in store.domains_dir.iterdir()) == expected
    assert [entry.artifact_dir for entry in store.load_manifest().domains] == expected
    assert json.loads((store.domains_dir / "00" / "domain.json").read_text())["name"] == "Domain 0"


@pytest.mark.asyncio
async def test_end_to_end_seed_generation_and_materialization(tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    assets = load_generation_assets("proactive")
    store = RunArtifactStore.create(generation_config, generation_asset_hashes("proactive"))

    domains = QueueGenerator(
        [
            json.dumps([{"name": "Account Support", "applications": [{"function": "Look up account status"}]}]),
            "[]",
        ]
    )
    candidates = await DomainGenerationStage(generation_config, assets.domain_prompt, store, domains).run()
    assert len(candidates) == 1
    assert store.load_manifest().domains[0].source_index == 0
    assert domains.calls == [
        [{"role": "user", "content": assets.domain_prompt}],
        [
            {
                "role": "user",
                "content": (
                    assets.domain_prompt + "\n\nPreviously brainstormed domains: ['Account Support'].\n"
                    "Do not repeat these domains. Try looking for other domains or find specific sub-domains."
                ),
            }
        ],
    ]

    tools_json = json.dumps(tool())
    policy_tools = QueueGenerator(
        [
            "<policy>Authenticate before looking up account state.</policy>",
            f"<tools>{tools_json}</tools>",
            "<policy>Authenticate before looking up account state.</policy>",
            f"<tools>{tools_json}</tools>",
        ]
    )
    judge = QueueGenerator(['<judgment>{"accepted": true, "explanation": "coherent"}</judgment>'])
    await PolicyToolsGenerationStage(generation_config, assets.policy_tools, store, policy_tools, judge).run()
    entry = store.load_manifest().domains[0]
    assert entry.stages["policy_tools"].state == StageState.COMPLETE

    scenarios = QueueGenerator(
        [
            json.dumps({"scenarios": [scenario("Check account status")]}),
            json.dumps({"scenarios": [scenario("Request an unsupported legal review")]}),
        ]
    )
    await ScenarioGenerationStage(generation_config, assets.scenarios, store, scenarios).run()
    entry = store.load_manifest().domains[0]
    assert entry.stages["scenarios"].state == StageState.COMPLETE
    assert entry.scenario_count == 2

    output_path = tmp_path / "dataset.jsonl"
    report = build_sample_dataset(
        source_dirs=[store.domains_dir],
        output_path=output_path,
        report_path=tmp_path / "dataset.report.json",
        max_rows=None,
        dataset_name="generated_test",
        source_names=[generation_config.source_name],
        max_rows_per_domain=None,
        scan_domains_per_source=None,
    )
    assert report["rows_written"] == 2
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    expected_scope = scope_schedule(
        2,
        0.5,
        seed=f"{generation_config.random_seed}:{candidates[0].domain_id}",
    )
    assert sorted(row["customer_scenario"]["outside_policy_scope"] for row in rows) == sorted(expected_scope)
    checked = next(row for row in rows if row["customer_scenario"]["reason_for_contact"] == "Check account status")
    assert checked["customer_scenario"]["customer_details"] == "Account ID A-123"
    assert checked["customer_scenario"]["unknown_info"] == "Current account status"
    assert checked["customer_scenario"]["task_instructions"] == (
        "Provide the account ID when asked and seek a resolution."
    )
    assert rows[0]["metadata"]["seed_run_id"] == store.load_manifest().run_id
    assert rows[0]["metadata"]["domain_generator_model"] == "domain-model"
    assert rows[0]["source_artifacts"]["domain_id"] == candidates[0].domain_id


def test_manifest_omits_endpoint_and_credential_configuration(tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    store = RunArtifactStore.create(generation_config, generation_asset_hashes("proactive"))
    serialized = store.manifest_path.read_text(encoding="utf-8")
    assert "http://model.invalid" not in serialized
    assert "TEST_API_KEY" not in serialized


def test_run_identity_changes_with_profile_assets(tmp_path: Path) -> None:
    generation_config = config(tmp_path)
    store = RunArtifactStore.create(generation_config, {"policy_prompt": "hash-one"})
    first_run_id = store.load_manifest().run_id
    assert first_run_id
    with pytest.raises(ValueError, match="output directory belongs to run"):
        RunArtifactStore.create(generation_config, {"policy_prompt": "hash-two"})
