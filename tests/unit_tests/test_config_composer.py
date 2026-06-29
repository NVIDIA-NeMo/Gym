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
from pathlib import Path

from omegaconf import OmegaConf
from pytest import raises

from nemo_gym.config_composer import (
    AgentNotComposableError,
    ComposeRequest,
    MandatoryPlaceholderError,
    NoComposableAgentBlockError,
    _validate_no_mandatory_placeholders,
    compose,
    find_agent_block_key,
    substitute_agent,
    substitute_dataset_params,
    substitute_model,
)


def _merged(*, agent_type: str = "simple_agent", resources_name: str = "bench_resources", num_repeats: int = 16):
    """A merged benchmark config: one resources block + one agent block carrying the benchmark."""
    return OmegaConf.create(
        {
            "policy_model": {"responses_api_models": {"vllm_model": {"entrypoint": "app.py"}}},
            resources_name: {
                "resources_servers": {
                    "bench": {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "judge_model_server": {"type": "responses_api_models", "name": "policy_model"},
                    }
                }
            },
            "bench_agent": {
                "responses_api_agents": {
                    agent_type: {
                        "entrypoint": "app.py",
                        "resources_server": {"type": "resources_servers", "name": resources_name},
                        "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        "datasets": [
                            {
                                "name": "bench",
                                "type": "benchmark",
                                "jsonl_fpath": "benchmarks/bench/data/bench.jsonl",
                                "prompt_config": None,
                                "prepare_script": "benchmarks/bench/prepare.py",
                                "num_repeats": num_repeats,
                            }
                        ],
                    }
                }
            },
        }
    )


def _fake_resolver(tmp_path: Path, agent_name: str, *, body: str):
    """Return a resolve_agent_config_path stub that writes ``body`` and returns its path."""
    config_path = tmp_path / f"{agent_name}.yaml"
    config_path.write_text(body)

    def _resolve(name, require_composable=False):
        if name == "swe_agents" and require_composable:
            raise AgentNotComposableError("Agent 'swe_agents' is self-contained")
        return str(config_path)

    return _resolve


def _new_agent_body(agent_name: str = "react_agent") -> str:
    # A composable agent config with its own (to-be-overridden) wiring.
    return (
        f"some_key:\n  responses_api_agents:\n    {agent_name}:\n      entrypoint: app.py\n"
        "      max_turns: 8\n"
        "      resources_server:\n        type: resources_servers\n        name: ???\n"
        "      model_server:\n        type: responses_api_models\n        name: policy_model\n"
    )


class TestFindAgentBlockKey:
    def test_finds_benchmark_carrying_block(self) -> None:
        assert find_agent_block_key(_merged()) == "bench_agent"

    def test_prefers_benchmark_block_over_auxiliary_agent(self) -> None:
        merged = _merged()
        merged["aux_agent"] = OmegaConf.create({"responses_api_agents": {"helper_agent": {"entrypoint": "app.py"}}})
        assert find_agent_block_key(merged) == "bench_agent"

    def test_single_agent_block_without_benchmark_is_accepted(self) -> None:
        merged = OmegaConf.create({"only_agent": {"responses_api_agents": {"simple_agent": {"entrypoint": "app.py"}}}})
        assert find_agent_block_key(merged) == "only_agent"

    def test_zero_agent_blocks_raises(self) -> None:
        merged = OmegaConf.create({"policy_model": {"responses_api_models": {"vllm_model": {}}}})
        with raises(NoComposableAgentBlockError, match="No `responses_api_agents` block"):
            find_agent_block_key(merged)

    def test_ambiguous_agent_blocks_without_benchmark_raises(self) -> None:
        merged = OmegaConf.create(
            {
                "a": {"responses_api_agents": {"x": {"entrypoint": "app.py"}}},
                "b": {"responses_api_agents": {"y": {"entrypoint": "app.py"}}},
            }
        )
        with raises(NoComposableAgentBlockError, match="Ambiguous composition target"):
            find_agent_block_key(merged)

    def test_multiple_benchmark_blocks_raises(self) -> None:
        merged = _merged()
        merged["bench_agent_2"] = OmegaConf.create(merged["bench_agent"])
        with raises(NoComposableAgentBlockError, match="Multiple agent blocks"):
            find_agent_block_key(merged)

    def test_block_with_multiple_agent_types_is_ignored(self) -> None:
        # A `responses_api_agents` mapping with two inner types is not a valid single-agent block.
        merged = OmegaConf.create(
            {"weird": {"responses_api_agents": {"a": {"entrypoint": "app.py"}, "b": {"entrypoint": "app.py"}}}}
        )
        with raises(NoComposableAgentBlockError, match="No `responses_api_agents` block"):
            find_agent_block_key(merged)


class TestSubstituteDatasetParams:
    def test_sets_num_repeats_and_prompt_config(self) -> None:
        merged = _merged(num_repeats=16)
        out = substitute_dataset_params(merged, "bench_agent", num_repeats=4, prompt_config="prompts/p.yaml")

        dataset = out["bench_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]
        assert dataset["num_repeats"] == 4
        assert dataset["prompt_config"] == "prompts/p.yaml"
        # Original is untouched (deep copy).
        assert merged["bench_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]["num_repeats"] == 16

    def test_none_fields_are_noop(self) -> None:
        merged = _merged(num_repeats=16)
        out = substitute_dataset_params(merged, "bench_agent")
        assert out["bench_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]["num_repeats"] == 16

    def test_missing_benchmark_dataset_raises(self) -> None:
        merged = _merged()
        merged["bench_agent"]["responses_api_agents"]["simple_agent"]["datasets"] = []
        with raises(NoComposableAgentBlockError, match="Expected exactly one"):
            substitute_dataset_params(merged, "bench_agent", num_repeats=2)


class TestSubstituteAgent:
    def test_swaps_agent_and_carries_over_wiring(self, tmp_path: Path) -> None:
        merged = _merged(agent_type="simple_agent")
        resolver = _fake_resolver(tmp_path, "react_agent", body=_new_agent_body("react_agent"))

        out = substitute_agent(merged, "bench_agent", "react_agent", resolve_agent_config_path=resolver)

        agents = out["bench_agent"]["responses_api_agents"]
        # Re-keyed to the new agent name; old type gone.
        assert list(agents.keys()) == ["react_agent"]
        block = agents["react_agent"]
        # New agent's own field is preserved.
        assert block["max_turns"] == 8
        # Env wiring wins over the agent config's values.
        assert block["resources_server"]["name"] == "bench_resources"
        assert block["model_server"]["name"] == "policy_model"
        assert block["datasets"][0]["name"] == "bench"

    def test_pattern_b_agent_propagates_not_composable(self, tmp_path: Path) -> None:
        merged = _merged()
        resolver = _fake_resolver(tmp_path, "swe_agents", body=_new_agent_body())
        with raises(AgentNotComposableError):
            substitute_agent(merged, "bench_agent", "swe_agents", resolve_agent_config_path=resolver)

    def test_agent_config_without_agent_block_raises(self, tmp_path: Path) -> None:
        merged = _merged()
        resolver = _fake_resolver(tmp_path, "broken", body="k:\n  not_an_agent: true\n")
        with raises(NoComposableAgentBlockError, match="has no `responses_api_agents` block"):
            substitute_agent(merged, "bench_agent", "broken", resolve_agent_config_path=resolver)


class TestValidateNoMandatoryPlaceholders:
    def test_passes_when_filled(self) -> None:
        _validate_no_mandatory_placeholders(_merged(), "bench_agent")

    def test_raises_on_missing_in_agent_block(self) -> None:
        merged = _merged()
        merged["bench_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] = "???"
        with raises(MandatoryPlaceholderError, match="resources_server.name"):
            _validate_no_mandatory_placeholders(merged, "bench_agent")

    def test_unresolvable_interpolation_does_not_crash(self) -> None:
        # A composable (Pattern A) agent may interpolate a key NO_MODEL doesn't inject (e.g. a
        # provider api key like ${nvidia_api_key}). Scanning for `???` must NOT resolve it — doing so
        # would raise InterpolationKeyError and block an otherwise-valid composition.
        merged = _merged()
        merged["bench_agent"]["responses_api_agents"]["simple_agent"]["api_key"] = "${nvidia_api_key}"
        _validate_no_mandatory_placeholders(merged, "bench_agent")  # must not raise

    def test_raises_on_missing_in_referenced_resources_block(self) -> None:
        merged = _merged()
        merged["bench_resources"]["resources_servers"]["bench"]["judge_model_server"]["name"] = "???"
        with raises(MandatoryPlaceholderError, match="judge_model_server.name"):
            _validate_no_mandatory_placeholders(merged, "bench_agent")

    def test_missing_resources_server_name_does_not_crash(self) -> None:
        merged = _merged()
        # If the resources_server reference itself is missing, validation skips it (the agent-block
        # walk still flags it as a placeholder).
        merged["bench_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] = "???"
        with raises(MandatoryPlaceholderError):
            _validate_no_mandatory_placeholders(merged, "bench_agent")

    def test_dangling_resources_reference_is_ignored(self) -> None:
        merged = _merged()
        merged["bench_agent"]["responses_api_agents"]["simple_agent"]["resources_server"]["name"] = "nonexistent"
        # No such top-level block; validation simply does not descend into it.
        _validate_no_mandatory_placeholders(merged, "bench_agent")


class TestSubstituteModel:
    def test_is_noop(self) -> None:
        merged = _merged()
        assert substitute_model(merged) is merged


class TestCompose:
    def test_empty_request_is_passthrough(self) -> None:
        merged = _merged()
        out = compose(merged, ComposeRequest(), resolve_agent_config_path=lambda *a, **k: "")
        assert OmegaConf.to_container(out, resolve=False) == OmegaConf.to_container(merged, resolve=False)
        assert out is not merged

    def test_compose_request_is_empty_property(self) -> None:
        assert ComposeRequest().is_empty is True
        assert ComposeRequest(num_repeats=2).is_empty is False

    def test_full_compose_swaps_agent_and_edits_dataset(self, tmp_path: Path) -> None:
        merged = _merged(agent_type="simple_agent", num_repeats=16)
        resolver = _fake_resolver(tmp_path, "react_agent", body=_new_agent_body("react_agent"))

        out = compose(
            merged,
            ComposeRequest(agent="react_agent", num_repeats=2, prompt_config="prompts/p.yaml"),
            resolve_agent_config_path=resolver,
        )

        block = out["bench_agent"]["responses_api_agents"]["react_agent"]
        assert block["resources_server"]["name"] == "bench_resources"
        assert block["datasets"][0]["num_repeats"] == 2
        assert block["datasets"][0]["prompt_config"] == "prompts/p.yaml"

    def test_compose_validates_placeholders(self, tmp_path: Path) -> None:
        merged = _merged()
        merged["bench_resources"]["resources_servers"]["bench"]["judge_model_server"]["name"] = "???"
        with raises(MandatoryPlaceholderError):
            compose(merged, ComposeRequest(num_repeats=2), resolve_agent_config_path=lambda *a, **k: "")

    def test_compose_propagates_not_composable(self, tmp_path: Path) -> None:
        merged = _merged()
        resolver = _fake_resolver(tmp_path, "swe_agents", body=_new_agent_body())
        with raises(AgentNotComposableError):
            compose(merged, ComposeRequest(agent="swe_agents"), resolve_agent_config_path=resolver)
