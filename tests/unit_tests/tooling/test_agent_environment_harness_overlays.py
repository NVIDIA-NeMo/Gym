# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment migration: agent environment harness overlays."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollouts.rollout_collection import _environment_servers_by_agent
from tests.helpers.agent_environment_migration import (
    AGENT_CONFIG,
    HARNESS_OVERRIDE,
    RELAY,
    _parent_and_child,
    _parse,
    _server_fronting,
    load_migration,
)


migration = load_migration()


def test_switch_keeps_a_relay_that_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A harness variant points the same server at an agent without sessions, so it must stay a relay.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    variant = tmp_path / "variant.yaml"
    variant.write_text(HARNESS_OVERRIDE.format(base=base))
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])

    assert migration.main(["--switch-simple-agent", str(base)]) == 0

    assert base.read_text() == AGENT_CONFIG + RELAY
    resolved = _parse(variant, strict=True)
    assert list(resolved["my_environment_server"]["environment_servers"]) == ["legacy_agent"]
    assert f"stays on the relay: my_environment_server in {base}: {variant} merges it with a legacy_agent server" in (
        capsys.readouterr().out
    )


def test_switch_keeps_every_declaration_of_a_relay_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A middle config overrides the relay under its own name, so two files declare the one instance.
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    middle = tmp_path / "middle.yaml"
    middle.write_text(
        f"config_paths:\n  - {base}\n"
        "my_environment_server:\n  environment_servers:\n    legacy_agent:\n      agent_server:\n"
        "        name: my_simple_agent\n"
    )
    variant = tmp_path / "variant.yaml"
    variant.write_text(HARNESS_OVERRIDE.format(base=middle))
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = (base.read_text(), middle.read_text())

    assert migration.main(["--switch-simple-agent", str(base), str(middle)]) == 0

    assert (base.read_text(), middle.read_text()) == before
    out = capsys.readouterr().out
    assert f"stays on the relay: my_environment_server in {base}: {variant} merges it" in out
    assert f"stays on the relay: my_environment_server in {middle}: {variant} merges it" in out
    assert list(_parse(variant, strict=True)["my_environment_server"]["environment_servers"]) == ["legacy_agent"]


def test_switch_refuses_a_child_relay_another_config_merges_with_a_legacy_agent_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The child cannot stay a relay while its parent switches, so the variant has to declare its own server.
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    variant.write_text(
        HARNESS_OVERRIDE.format(base=child)
        .replace("my_simple_agent: null", "child_agent: null")
        .replace("my_environment_server:", "child_environment_server:")
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = (base.read_text(), child.read_text())

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 2

    assert (base.read_text(), child.read_text()) == before
    error = capsys.readouterr().err
    assert "child_environment_server in a run of" in error and "my_environment_server, which switches" in error


def test_a_harness_with_its_own_relay_lets_the_benchmark_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The variant removes the benchmark's servers and fronts its harness with a relay of its own.
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    harness = (
        HARNESS_OVERRIDE.format(base=child)
        .split("my_simple_agent: null\n")[0]
        .replace("name: my_resources", "name: child_resources")
    )
    variant.write_text(
        harness
        + "my_simple_agent: null\nchild_agent: null\nmy_environment_server: null\nchild_environment_server: null\n"
        + _server_fronting("my_harness_agent", name="my_harness_environment_server", server_type="legacy_agent")
    )
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])

    assert migration.main(["--switch-simple-agent", str(base), str(child)]) == 0

    benchmark = _parse(child, strict=True)
    assert list(benchmark["child_environment_server"]["environment_servers"]) == ["single_agent_turn_legacy"]
    assert _environment_servers_by_agent(_parse(variant, strict=True)) == {
        "my_harness_agent": ["my_harness_environment_server"]
    }


def test_a_harness_can_override_a_relay_without_changing_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, child = _parent_and_child(tmp_path)
    variant = tmp_path / "variant.yaml"
    harness = (
        HARNESS_OVERRIDE.format(base=child)
        .split("my_simple_agent: null\n")[0]
        .replace("name: my_resources", "name: child_resources")
    )
    relay = _server_fronting("my_harness_agent", name="child_environment_server", server_type="legacy_agent")
    relay = relay.replace("  environment_servers:\n", "  _delete_key: _inherit_from\n  environment_servers:\n")
    relay = relay.replace("    legacy_agent:\n", "    _delete_key: single_agent_turn_legacy\n    legacy_agent:\n")
    variant.write_text(harness + "my_simple_agent: null\nchild_agent: null\nmy_environment_server: null\n" + relay)
    monkeypatch.setattr(migration, "repo_config_files", lambda: [variant])
    before = variant.read_text()
    routing = {"my_harness_agent": ["child_environment_server"]}
    assert _environment_servers_by_agent(_parse(variant, strict=True)) == routing

    assert migration.main(["--switch-simple-agent", str(base), str(child), str(variant)]) == 0

    assert variant.read_text() == before
    benchmark = _parse(child, strict=True)
    assert list(benchmark["child_environment_server"]["environment_servers"]) == ["single_agent_turn_legacy"]
    parsed = _parse(variant, strict=True)
    assert _environment_servers_by_agent(parsed) == routing
    assert list(parsed["child_environment_server"]["environment_servers"]) == ["legacy_agent"]


@pytest.mark.parametrize("present_before_inheritance", [False, True])
@pytest.mark.parametrize("directive", ["_inherit_from", "_copy"])
def test_resolved_instances_matches_gym_deletion_order(present_before_inheritance: bool, directive: str) -> None:
    document = {
        "base": {"setting": "inherited", "retained": True},
        "child": {directive: "base", "_delete_key": "setting"},
    }
    if present_before_inheritance:
        document["child"]["setting"] = "overridden"
    runtime = OmegaConf.create(document)
    GlobalConfigDictParser()._recursively_swap_keys(runtime)

    expected = {"retained": True}
    if present_before_inheritance:
        # Gym removes the local setting first, so inheritance supplies it again.
        expected["setting"] = "inherited"
    assert OmegaConf.to_container(runtime.child) == expected
    assert migration.resolved_instances([document])["child"] == expected


def test_resolved_instances_deletes_after_all_overlays_merge() -> None:
    documents = [
        {"instance": {"_delete_key": "obsolete", "retained": True}},
        {"instance": {"obsolete": {"setting": "later overlay"}}},
    ]
    runtime = OmegaConf.merge(*documents)
    GlobalConfigDictParser()._recursively_swap_keys(runtime)

    assert OmegaConf.to_container(runtime.instance) == {"retained": True}
    assert migration.resolved_instances(documents)["instance"] == {"retained": True}


@pytest.mark.parametrize(
    ("benchmark", "recipe", "resources", "dataset"),
    [
        ("hle", "pi", "equivalence_llm_judge", "hle"),
        ("hle", "opencode", "equivalence_llm_judge", "hle"),
        ("hle", "pi_search", "equivalence_llm_judge", "hle"),
        ("hle", "opencode_search", "equivalence_llm_judge", "hle"),
        ("hle_diamond", "pi", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "opencode", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "pi_search", "equivalence_llm_judge", "hle_diamond_text"),
        ("hle_diamond", "opencode_search", "equivalence_llm_judge", "hle_diamond_text"),
        ("apex_shortlist", "pi", "math_with_judge", "apex_shortlist"),
        ("apex_shortlist", "opencode", "math_with_judge", "apex_shortlist"),
    ],
)
def test_harness_recipes_preserve_routing_and_bindings_across_the_switch(
    benchmark: str, recipe: str, resources: str, dataset: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variable in ("PI_SANDBOX_IMAGE", "OPENCODE_SANDBOX_IMAGE", "OPENSANDBOX_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.setenv(variable, "test-only")
    parser = GlobalConfigDictParser()
    _, documents = parser.load_extra_config_paths([f"benchmarks/{benchmark}/{recipe}.yaml"])
    agent_name = f"{benchmark}_{recipe.removesuffix('_search')}_agent"
    server_name = f"{benchmark}_{resources}_environment_server"
    resources_name = f"{benchmark}_{resources}_resources_server"
    resolved = []
    for switched in (False, True):
        if switched:
            for document in documents:
                for name, instance in list(document.items_ex(resolve=False)):
                    if not isinstance(instance, DictConfig):
                        continue
                    relay = (instance.get("environment_servers") or {}).get("legacy_agent")
                    agent = (relay.get("agent_server") or {}).get("name", "") if relay else ""
                    if agent.endswith("_simple_agent"):
                        document[name] = OmegaConf.create(
                            migration.stanza(
                                name,
                                agent,
                                resources_name if name == server_name else resources,
                                instance.get("_inherit_from"),
                            )
                        )[name]
        parsed = parser.parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    *documents,
                    {"config_paths": [], "error_on_agent_without_environment_server": True},
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )
        assert _environment_servers_by_agent(parsed) == {agent_name: [server_name]}
        assert list(parsed[server_name].environment_servers) == ["legacy_agent"]
        agent = next(iter(parsed[agent_name].responses_api_agents.values()))
        assert agent.resources_server.name == resources_name
        assert agent.datasets[0].name == dataset
        if recipe.endswith("_search"):
            assert agent.tool_servers == [{"type": "resources_servers", "name": "tavily"}]
        participants = {}
        for server in parser.filter_for_server_instance_configs(parsed):
            config = OmegaConf.to_container(server.get_inner_run_server_config_dict(), resolve=True)
            participants[server.name] = {key: value for key, value in config.items() if key not in {"host", "port"}}
        resolved.append(participants)
    assert resolved[0] == resolved[1]
