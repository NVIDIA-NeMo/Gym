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
"""Contracts between files that nothing imports across at runtime.

The shipped YAML overrides the class defaults; the agent's repair bound must equal the
server's; the example rows must be what the preparer emits; ``task_data.py`` must name
every field the server requires.
"""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml
from app import REQUIRED_TASK_FIELDS, VeriSoftBenchResourcesServerConfig
from task_data import TaskData


SERVER_DIR = Path(__file__).resolve().parents[1]
GYM_ROOT = SERVER_DIR.parents[1]
sys.path.insert(0, str(SERVER_DIR / "scripts"))
import prepare_verisoftbench as prep  # noqa: E402


def _config():
    return yaml.safe_load((SERVER_DIR / "configs" / "verisoftbench.yaml").read_text())


class TestShippedConfig:
    def test_yaml_agrees_with_class_defaults_for_every_execution_knob(self) -> None:
        server = _config()["verisoftbench"]["resources_servers"]["verisoftbench"]
        defaults = VeriSoftBenchResourcesServerConfig.model_fields
        for knob in (
            "max_repair_rounds",
            "max_concurrent_compilations",
            "conflict_attempts_initial",
            "conflict_attempts_repair",
            "require_container_at_startup",
        ):
            assert server[knob] == defaults[knob].default, knob

    def test_agent_repair_bound_matches_server(self) -> None:
        cfg = _config()
        server = cfg["verisoftbench"]["resources_servers"]["verisoftbench"]
        agent = cfg["verisoftbench_agent"]["responses_api_agents"]["verisoftbench_agent"]
        assert agent["max_repair_rounds"] == server["max_repair_rounds"] == 3
        standalone = yaml.safe_load(
            (GYM_ROOT / "responses_api_agents/verisoftbench_agent/configs/verisoftbench_agent.yaml").read_text()
        )
        assert (
            standalone["verisoftbench_agent"]["responses_api_agents"]["verisoftbench_agent"]["max_repair_rounds"] == 3
        )

    def test_benchmark_registration_uses_upstreams_draw_count(self) -> None:
        bench = yaml.safe_load((GYM_ROOT / "benchmarks/verisoftbench/config.yaml").read_text())
        agent = bench["verisoftbench_benchmark_agent"]
        assert agent["_inherit_from"] == "verisoftbench_agent"
        [dataset] = agent["responses_api_agents"]["verisoftbench_agent"]["datasets"]
        assert dataset["num_repeats"] == 8 and dataset["type"] == "benchmark" and dataset["license"] == "MIT"
        assert dataset["prepare_script"] == "benchmarks/verisoftbench/prepare.py"
        manifest = yaml.safe_load((GYM_ROOT / "benchmarks/verisoftbench/manifest.yaml").read_text())
        assert manifest["datasets"][0]["num_repeats"] == 8 and manifest["resources_server"] == "verisoftbench"

    @staticmethod
    def _collate_benchmark_rows(rows: list[dict], jsonl_fpath) -> list[dict]:
        """Run rows through Gym's registered benchmark collation (the path `gym eval run` prepares with).

        This is the path that calls ``validate_prompt_compatibility`` and ``apply_prompt_to_row``
        (nemo_gym/train_data_utils.py) with the shipped pass-through prompt config, exactly as the
        registered ``--split benchmark`` declaration does.
        """
        from omegaconf import DictConfig

        from nemo_gym.config_types import ResponsesAPIAgentServerInstanceConfig
        from nemo_gym.train_data_utils import TrainDataProcessor

        jsonl_fpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
        server_type_config_dict = {
            "responses_api_agents": {
                "verisoftbench_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "verisoftbench",
                            "type": "benchmark",
                            "jsonl_fpath": str(jsonl_fpath),
                            "prepare_script": "benchmarks/verisoftbench/prepare.py",
                            "prompt_config": str(GYM_ROOT / "benchmarks/verisoftbench/prompt.yaml"),
                        }
                    ],
                    "resources_server": {"type": "resources_servers", "name": "verisoftbench"},
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                }
            }
        }
        config = ResponsesAPIAgentServerInstanceConfig(
            name="verisoftbench_benchmark_agent",
            server_type_config_dict=DictConfig(server_type_config_dict),
            responses_api_agents=server_type_config_dict["responses_api_agents"],
        )
        [prepared] = TrainDataProcessor()._collate_samples_single_type(
            "benchmark", [config], task_data_validation="off"
        )
        return [json.loads(line) for line in prepared.read_text().splitlines()]

    def test_benchmark_form_rows_render_the_frozen_turns_through_gym_collation(self, tmp_path) -> None:
        frozen = [
            json.loads(line)
            for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()
            if line.strip()
        ]
        bench_rows = [
            prep.format_benchmark_row(
                {k: v for k, v in row["verifier_metadata"].items() if k != "context_mode"},
                row["system_prompt"],
                row["user_prompt"],
            )
            for row in frozen
        ]
        for row in bench_rows:
            assert "responses_create_params" not in row
        collated = self._collate_benchmark_rows(bench_rows, tmp_path / "bench.jsonl")
        assert [r["responses_create_params"]["input"] for r in collated] == [
            r["responses_create_params"]["input"] for r in frozen
        ], "the pass-through prompt config must render exactly the frozen prepared turns"
        assert all(r["task_source"] == "verisoftbench_benchmark_agent" for r in collated)
        bench = yaml.safe_load((GYM_ROOT / "benchmarks/verisoftbench/config.yaml").read_text())
        [dataset] = bench["verisoftbench_benchmark_agent"]["responses_api_agents"]["verisoftbench_agent"]["datasets"]
        assert dataset["prompt_config"] == "benchmarks/verisoftbench/prompt.yaml"

    def test_explicit_input_rows_are_rejected_by_gym_when_a_prompt_config_is_declared(self, tmp_path) -> None:
        # A row in the server split's explicit-input form (the frozen example rows carry
        # exactly that shape) must be refused by the compatibility guard, not rendered.
        frozen = json.loads((SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()[0])
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._collate_benchmark_rows([frozen], tmp_path / "explicit.jsonl")

    def test_benchmark_prepare_delegates_to_the_server_preparer_in_benchmark_form(self, monkeypatch) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "verisoftbench_benchmark_prepare", GYM_ROOT / "benchmarks" / "verisoftbench" / "prepare.py"
        )
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        captured = {}

        class FakeServerPrepare:
            @staticmethod
            def main(argv):
                captured["argv"] = argv

        monkeypatch.setattr(bench, "_load_server_prepare", lambda: FakeServerPrepare)
        bench.prepare()
        assert captured["argv"] == ["--output", str(bench.OUTPUT_FPATH), "--benchmark-form"]

    def test_resources_server_block_comes_first(self) -> None:
        # The README-table hook reads the first top-level key.
        assert next(iter(_config())) == "verisoftbench"


class TestExampleData:
    def test_example_rows_are_preparer_shaped_and_synthetic(self) -> None:
        rows = [
            json.loads(line)
            for line in (SERVER_DIR / "data" / "example.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 5
        for row in rows:
            # `gym dataset collate` rewrites the preparer's agent_ref into task_source; the tracked file carries that schema.
            assert row["task_source"] == prep.AGENT_REF["name"]
            meta = row["verifier_metadata"]
            for field in REQUIRED_TASK_FIELDS:
                assert field in meta, field
            TaskData.model_validate(meta)
            assert meta["thm_name"].startswith("GymExample."), "example theorems are authored here, not upstream's"
            assert meta["id"] >= 100_000, "example ids must not collide with upstream's 1..500"
            assert [m["role"] for m in row["responses_create_params"]["input"]] == ["system", "user"]

    def test_full_example_set_regenerates_and_carries_no_service_identifiers(self) -> None:
        # The committed example artifacts must be exactly what scripts/make_example_data.py
        # emits — no provider-collected bytes survive outside the generator. example.jsonl's
        # prompts need the pinned checkout, so that file is pinned down to its synthetic
        # source rows here; the four artifacts derived from it must rebuild byte-for-byte.
        import make_example_data as gen

        data = SERVER_DIR / "data"
        example_jsonl = (data / "example.jsonl").read_bytes()
        rows = [json.loads(line) for line in example_jsonl.decode("utf-8").splitlines()]
        assert [
            {k: v for k, v in row["verifier_metadata"].items() if k != "context_mode"} for row in rows
        ] == gen.example_rows()
        rollouts = gen.rollout_rows(rows)
        assert gen.jsonl_bytes(rollouts) == (data / "example_rollouts.jsonl").read_bytes()
        assert (
            gen.jsonl_bytes(gen.materialized_input_rows(rows))
            == (data / "example_rollouts_materialized_inputs.jsonl").read_bytes()
        )
        assert gen.aggregate_metrics_bytes(rollouts) == (data / "example_rollouts_aggregate_metrics.json").read_bytes()
        assert gen.example_metrics_bytes(example_jsonl) == (data / "example_metrics.json").read_bytes()
        # No provider identifiers anywhere: real runs carry opaque response/completion id
        # shapes, a provider model route and wall-clock timestamps; the committed set must
        # carry only the generator's fixed placeholders.
        service_markers = re.compile(
            rb"resp_[0-9A-Za-z_-]{8,}|chatcmpl-[0-9a-f-]{8,}|azure|anthropic|openai|claude|gpt", re.IGNORECASE
        )
        for fpath in sorted(data.glob("example*")):
            assert service_markers.search(fpath.read_bytes()) is None, fpath.name
        for row in [json.loads(line) for line in (data / "example_rollouts.jsonl").read_text().splitlines()]:
            for response in (row["response"], *(a["response"] for a in row["attempts"])):
                assert response["created_at"] == gen.EXAMPLE_CREATED_AT
                assert response["model"] == gen.EXAMPLE_MODEL


class TestTaskData:
    def test_every_field_the_server_requires_is_documented(self) -> None:
        for field in REQUIRED_TASK_FIELDS:
            assert field in TaskData.model_fields
