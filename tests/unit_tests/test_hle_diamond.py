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
"""HLE-Diamond benchmark: preparation, config composition and partition slices."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from datasets import Dataset
from omegaconf import OmegaConf

import benchmarks.hle_diamond.prepare as prepare_module
import benchmarks.hle_diamond.prepare_text as prepare_text_module
from nemo_gym.benchmarks import BENCHMARKS_DIR, BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths
from nemo_gym.config_types import ModelServerRef
from nemo_gym.discovery import _parse_no_environment_tolerating_unset_values
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.server_utils import ServerClient
from resources_servers.equivalence_llm_judge.app import LLMJudgeResourcesServer, LLMJudgeResourcesServerConfig


REPO_ROOT = Path(__file__).parents[2]
HLE_DIAMOND_DIR = REPO_ROOT / "benchmarks" / "hle_diamond"
HLE_JUDGE_PROMPT = REPO_ROOT / "benchmarks" / "hle" / "prompts" / "judge.txt"
IMAGE_DATA_URI = "data:image/png;base64,iVBORw0KGgo="
ROW_FIELDS = [
    "question",
    "expected_answer",
    "answer_type",
    "uuid",
    "partition",
    "category",
    "raw_subject",
    "has_image",
]


def _source_rows() -> list[dict]:
    """Four rows shaped like cais/hle-diamond: both partitions, both answer types, one image question."""
    common = {"author_name": "A. Author", "rationale": "Because.", "canary": "BENCHMARK DATA SHOULD NEVER APPEAR"}
    return [
        {
            "id": "q-reasoning-text",
            "question": "What is 2+2?",
            "image": "",
            "answer": "4",
            "answer_type": "exactMatch",
            "partition": "reasoning",
            "category": "Math",
            "raw_subject": "Arithmetic",
            **common,
        },
        {
            "id": "q-knowledge-mc",
            "question": "Which is a noble gas? A. Neon B. Iron",
            "image": "",
            "answer": "A",
            "answer_type": "multipleChoice",
            "partition": "knowledge",
            "category": "Chemistry",
            "raw_subject": "Inorganic Chemistry",
            **common,
        },
        {
            "id": "q-reasoning-image",
            "question": "How many squares are in the picture?",
            "image": IMAGE_DATA_URI,
            "answer": "9",
            "answer_type": "exactMatch",
            "partition": "reasoning",
            "category": "Chess/Logic/Puzzle",
            "raw_subject": "Puzzles",
            **common,
        },
        {
            "id": "q-knowledge-text",
            "question": "Who wrote the Aeneid?",
            "image": "",
            "answer": "Virgil",
            "answer_type": "exactMatch",
            "partition": "knowledge",
            "category": "Humanities/Social Science",
            "raw_subject": "Classics",
            **common,
        },
    ]


FIXTURE_EXPECTATION = prepare_module.CorpusExpectation(
    rows=4, partition_counts={"reasoning": 2, "knowledge": 2}, image_rows=1
)


@pytest.fixture
def prepared_env(monkeypatch, tmp_path):
    """Serve an in-memory arrow dataset instead of downloading; redirect outputs to tmp_path."""
    state = {"rows": _source_rows()}
    calls: list[dict] = []

    def fake_load_dataset(path, split=None, revision=None, token=None, **kwargs):
        calls.append({"path": path, "split": split, "revision": revision, "token": token})
        rows = state["rows"]
        return Dataset.from_dict({column: [row[column] for row in rows] for column in rows[0]})

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr("nemo_gym.global_config.maybe_get_global_config_dict", lambda: {"hf_token": "hf-fixture"})
    monkeypatch.setattr(prepare_module, "EXPECTED_CORPUS", FIXTURE_EXPECTATION)
    monkeypatch.setattr(prepare_module, "OUTPUT_FPATH", tmp_path / "hle_diamond_benchmark.jsonl")
    monkeypatch.setattr(prepare_module, "OUTPUT_TEXT_FPATH", tmp_path / "hle_diamond_benchmark_text.jsonl")
    return state, calls


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestPrepare:
    def test_full_set_materializes_prompt_and_image_blocks(self, prepared_env, capsys):
        _, calls = prepared_env

        output = prepare_module.prepare()

        assert output == prepare_module.OUTPUT_FPATH
        assert calls == [
            {
                "path": "cais/hle-diamond",
                "split": "test",
                "revision": prepare_module.REVISION,
                "token": "hf-fixture",
            }
        ]
        rows = _read_rows(output)
        assert [list(row) for row in rows] == [ROW_FIELDS + ["responses_create_params"]] * 4
        by_id = {row["uuid"]: row for row in rows}

        image_row = by_id["q-reasoning-image"]
        assert image_row["has_image"] is True
        system, user = image_row["responses_create_params"]["input"]
        assert system["role"] == "system" and "Explanation:" in system["content"]
        assert user == {
            "role": "user",
            "content": [
                {"type": "input_text", "text": image_row["question"]},
                {"type": "input_image", "image_url": IMAGE_DATA_URI, "detail": "high"},
            ],
        }

        text_row = by_id["q-knowledge-mc"]
        assert text_row["has_image"] is False
        assert text_row["responses_create_params"]["input"][1] == {"role": "user", "content": text_row["question"]}
        assert (text_row["answer_type"], text_row["partition"]) == ("multipleChoice", "knowledge")
        assert (text_row["expected_answer"], text_row["category"]) == ("A", "Chemistry")

        serialized = output.read_text(encoding="utf-8")
        for leaked in ("rationale", "author_name", "canary", "A. Author", "BENCHMARK DATA"):
            assert leaked not in serialized
        assert serialized.count(IMAGE_DATA_URI) == 1
        assert "Wrote 4 problems" in capsys.readouterr().out

    def test_text_only_drops_image_rows_and_keeps_raw_fields(self, prepared_env):
        output = prepare_text_module.prepare()

        assert output == prepare_module.OUTPUT_TEXT_FPATH
        rows = _read_rows(output)
        assert [row["uuid"] for row in rows] == ["q-reasoning-text", "q-knowledge-mc", "q-knowledge-text"]
        assert [list(row) for row in rows] == [ROW_FIELDS] * 3
        assert all(row["has_image"] is False for row in rows)
        assert IMAGE_DATA_URI not in output.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda rows: rows[:-1], "3 rows loaded, expected 4"),
            (
                lambda rows: [{**r, "partition": "reasoning"} if r["id"] == "q-knowledge-text" else r for r in rows],
                "partition split",
            ),
            (lambda rows: [{**r, "image": ""} for r in rows], "0 image rows, expected 1"),
            (
                lambda rows: [{**r, "id": "q-reasoning-text"} if r["id"] == "q-knowledge-text" else r for r in rows],
                "3 unique ids across 4 rows",
            ),
        ],
        ids=["row_count", "partition_split", "image_count", "duplicate_ids"],
    )
    def test_incomplete_corpus_fails_closed_before_writing(self, prepared_env, mutate, message):
        state, _ = prepared_env
        state["rows"] = mutate(_source_rows())

        with pytest.raises(ValueError, match=message):
            prepare_module.prepare()
        with pytest.raises(ValueError, match=message):
            prepare_text_module.prepare()

        assert not prepare_module.OUTPUT_FPATH.exists()
        assert not prepare_module.OUTPUT_TEXT_FPATH.exists()

    def test_limit_is_the_only_subset_mechanism_and_bypasses_the_check_loudly(self, prepared_env, capsys):
        state, _ = prepared_env
        state["rows"] = _source_rows()[:-1]  # incomplete corpus: only --limit may write it

        output = prepare_text_module.prepare(limit=2)

        assert [row["uuid"] for row in _read_rows(output)] == ["q-reasoning-text", "q-knowledge-mc"]
        assert "skipping the corpus completeness check" in capsys.readouterr().out

    @pytest.mark.parametrize("limit", [0, -1, True, 2.5, "3"], ids=["zero", "negative", "bool", "float", "str"])
    def test_invalid_limit_is_rejected_before_any_download(self, monkeypatch, limit):
        def refuse(*args, **kwargs):
            raise AssertionError("load_dataset must not be called for invalid arguments")

        monkeypatch.setattr("datasets.load_dataset", refuse)
        with pytest.raises(ValueError, match="positive integer"):
            prepare_module.prepare(limit=limit)

    @pytest.mark.parametrize("argv", [["--limit", "0"], ["--limit", "-1"], ["--limit", "abc"], ["--limit", "1.5"]])
    def test_cli_rejects_non_positive_limit_at_parse_time(self, argv, capsys):
        with pytest.raises(SystemExit):
            prepare_module.build_parser().parse_args(argv)
        assert "positive integer" in capsys.readouterr().err

    def test_cli_accepts_text_only_and_limit(self):
        args = prepare_module.build_parser().parse_args(["--text-only", "--limit", "3"])
        assert (args.text_only, args.limit) == (True, 3)
        assert prepare_module.build_parser().parse_args([]).limit is None


# benchmark name -> (dataset name, agent, prompt_config)
LISTED_BENCHMARKS = {
    "hle_diamond": ("hle_diamond", "hle_diamond_equivalence_llm_judge_simple_agent", None),
    "hle_diamond/text": (
        "hle_diamond_text",
        "hle_diamond_text_equivalence_llm_judge_simple_agent",
        "benchmarks/hle/prompts/default.yaml",
    ),
    "hle_diamond/opencode": ("hle_diamond_text", "hle_diamond_opencode_agent", "benchmarks/hle/prompts/opencode.yaml"),
    "hle_diamond/pi": ("hle_diamond_text", "hle_diamond_pi_agent", "benchmarks/hle/prompts/opencode.yaml"),
}
JUDGE_SERVER = "hle_diamond_equivalence_llm_judge_resources_server"


def _resolve(*config_files: str):
    """Resolve hle_diamond config files the way listing does (runtime-only values tolerated)."""
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [f"benchmarks/hle_diamond/{name}" for name in config_files]},
    )
    return _parse_no_environment_tolerating_unset_values(initial)


class TestConfigs:
    def test_listed_benchmarks_resolve_and_the_judge_override_is_not_one(self):
        paths = [p for p in _benchmark_config_paths(BENCHMARKS_DIR) if p.is_relative_to(HLE_DIAMOND_DIR)]
        names = {_benchmark_config_name(p.relative_to(BENCHMARKS_DIR)): p for p in paths}

        # The search presets declare no dataset of their own (as benchmarks/hle), so they are selectable by
        # path but not listed; judge_openai.yaml must never be listed.
        assert set(names) == set(LISTED_BENCHMARKS)
        for name, (dataset, agent, prompt_config) in LISTED_BENCHMARKS.items():
            benchmark = BenchmarkConfig.from_config_path(names[name], strict=False)
            resolved_prompt = benchmark.dataset.prompt_config and str(benchmark.dataset.prompt_config)
            assert (benchmark.name, benchmark.agent_name, resolved_prompt, benchmark.num_repeats) == (
                dataset,
                agent,
                prompt_config,
                1,
            )

    @pytest.mark.parametrize(
        ("variant", "agent"),
        [("opencode_search", "hle_diamond_opencode_agent"), ("pi_search", "hle_diamond_pi_agent")],
    )
    def test_search_presets_resolve_with_the_hle_diamond_exclusion_list(self, variant, agent):
        benchmark = BenchmarkConfig.from_config_path(HLE_DIAMOND_DIR / f"{variant}.yaml", strict=False)
        assert (benchmark.name, benchmark.agent_name, benchmark.num_repeats) == ("hle_diamond_text", agent, 1)

        tavily = get_first_server_config_dict(_resolve(f"{variant}.yaml"), "tavily")
        assert tavily["exclude_domains_file_path"] == "benchmarks/hle_diamond/search_exclusions.json"

    @pytest.mark.parametrize("config_file", ["config.yaml", "text.yaml", "opencode.yaml", "pi.yaml"])
    def test_every_preset_shares_the_hle_judge_prompt_labels_and_partition_slices(self, config_file):
        judge = get_first_server_config_dict(_resolve(config_file), JUDGE_SERVER)

        # judge_prompt_template_fpath is read relative to the resources server directory.
        prompt = (
            REPO_ROOT / "resources_servers" / "equivalence_llm_judge" / judge["judge_prompt_template_fpath"]
        ).resolve()
        assert prompt == HLE_JUDGE_PROMPT.resolve() and prompt.is_file()
        assert (judge["judge_equal_label"], judge["judge_not_equal_label"]) == ("Judgement: yes", "Judgement: no")
        assert judge["judge_model_server"]["name"] == "policy_model"
        assert list(judge["report_slice_fields"]) == ["partition"]

    @pytest.mark.parametrize("config_file", ["config.yaml", "text.yaml", "opencode_search.yaml"])
    def test_judge_openai_override_repoints_the_shared_judge_server(self, config_file, monkeypatch):
        monkeypatch.delenv("HLE_DIAMOND_JUDGE_MODEL", raising=False)
        resolved = _resolve(config_file, "judge_openai.yaml")

        assert (
            get_first_server_config_dict(resolved, JUDGE_SERVER)["judge_model_server"]["name"]
            == "hle_diamond_judge_model"
        )
        assert (
            get_first_server_config_dict(resolved, "hle_diamond_judge_model")["openai_model"] == "azure/openai/o3-mini"
        )

        monkeypatch.setenv("HLE_DIAMOND_JUDGE_MODEL", "vendor/other-judge")
        resolved = _resolve(config_file, "judge_openai.yaml")
        assert (
            get_first_server_config_dict(resolved, "hle_diamond_judge_model")["openai_model"] == "vendor/other-judge"
        )

    def test_slice_knob_defaults_off_and_only_hle_diamond_enables_it(self):
        assert LLMJudgeResourcesServerConfig.model_fields["report_slice_fields"].default_factory() == []
        base = yaml.safe_load(
            (REPO_ROOT / "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml").read_text()
        )
        assert "report_slice_fields" not in base["equivalence_llm_judge"]["resources_servers"]["equivalence_llm_judge"]
        shipped = yaml.safe_load((HLE_DIAMOND_DIR / "config.yaml").read_text())
        assert shipped[JUDGE_SERVER]["resources_servers"]["equivalence_llm_judge"]["report_slice_fields"] == [
            "partition"
        ]

    @pytest.mark.parametrize("manifest", ["manifest.yaml", "text/manifest.yaml"])
    def test_manifests_mirror_runtime_composition(self, manifest):
        path = HLE_DIAMOND_DIR / manifest
        loaded = load_manifest(path)
        composition = _resolve_manifest_composition(
            resolve_manifest_config_path(path, loaded), dataset_owner=loaded.dataset_owner
        )
        assert _mirror_differences(loaded, composition) == {}
        assert (loaded.licensing, loaded.authors) == ("MIT", ["rshahbazyan"])
        assert "1000" in loaded.description and "LLM-as-a-judge" in loaded.description


def _judge_server(report_slice_fields: list[str]) -> LLMJudgeResourcesServer:
    config = LLMJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=str(HLE_JUDGE_PROMPT),
        judge_equal_label="Judgement: yes",
        judge_not_equal_label="Judgement: no",
        report_slice_fields=report_slice_fields,
    )
    return LLMJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _rollout(task: int, repeat: int, reward: float, partition, issues=()) -> dict:
    row = {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": reward,
        "judge_evaluations": [{"judgement_parsing_issues": list(issues)}],
    }
    if partition is not None:
        row["partition"] = partition
    return row


class TestPartitionSlices:
    def test_partition_slices_reach_key_metrics_through_the_aggregate_path(self):
        server = _judge_server(["partition"])
        rows = [
            _rollout(0, 0, 1.0, "reasoning"),
            _rollout(1, 0, 0.0, "reasoning"),
            _rollout(2, 0, 1.0, "reasoning"),
            _rollout(3, 0, 0.0, "knowledge"),
            _rollout(4, 0, 1.0, "knowledge", issues=["no_verdict"]),
            _rollout(5, 0, 1.0, None),  # missing label: counted, never assumed
            {**_rollout(6, 0, 1.0, "knowledge"), "mask_sample": True},  # masked: excluded upstream of the hook
        ]

        result = compute_aggregate_metrics(
            rows, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )

        agent = result.agent_metrics
        assert agent["slices/partition/reasoning/mean/reward"] == pytest.approx(2 / 3)
        assert (
            agent["slices/partition/reasoning/question_count"],
            agent["slices/partition/reasoning/rollout_count"],
        ) == (3, 3)
        assert agent["slices/partition/knowledge/mean/reward"] == 0.5
        assert agent["slices/partition/knowledge/question_count"] == 2
        assert agent["slices/partition/missing_label_rollouts"] == 1

        assert result.key_metrics["mean/reward"] == pytest.approx(4 / 6)
        assert result.key_metrics["slices/partition/reasoning/mean/reward"] == pytest.approx(2 / 3)
        assert result.key_metrics["slices/partition/knowledge/mean/reward"] == 0.5
        assert result.key_metrics["judgement_parsing_issue_rate"] == pytest.approx(1 / 6)
        assert not any(key.endswith("_count") for key in result.key_metrics)

    def test_repeats_count_each_question_once(self):
        server = _judge_server(["partition"])
        rows = [_rollout(0, repeat, float(repeat == 0), "reasoning") for repeat in range(2)]
        rows += [_rollout(1, repeat, 1.0, "knowledge") for repeat in range(2)]

        agent = compute_aggregate_metrics(rows, compute_metrics_fn=server.compute_metrics).agent_metrics

        assert (
            agent["slices/partition/reasoning/question_count"],
            agent["slices/partition/reasoning/rollout_count"],
        ) == (1, 2)
        assert agent["slices/partition/reasoning/mean/reward"] == 0.5
        assert agent["slices/partition/knowledge/mean/reward"] == 1.0

    def test_empty_slice_fields_leave_metrics_unchanged(self):
        server = _judge_server([])
        rows = [_rollout(0, 0, 1.0, "reasoning"), _rollout(1, 0, 0.0, "knowledge")]

        result = compute_aggregate_metrics(
            rows, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )

        assert not any(key.startswith("slices/") for key in result.agent_metrics)
        assert result.key_metrics["mean/reward"] == 0.5
        assert result.key_metrics["judgement_parsing_issue_rate"] == 0.0
        # Unjudged rows still produce no custom metrics at all, exactly as before the hook.
        assert server.compute_metrics([[{"reward": 1.0, "partition": "reasoning"}]]) == {}
