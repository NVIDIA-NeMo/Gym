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
import re
import shlex
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from datasets import Dataset
from omegaconf import OmegaConf

import benchmarks.hle_diamond.prepare as prepare_module
import benchmarks.hle_diamond.prepare_text as prepare_text_module
from nemo_gym.benchmarks import BENCHMARKS_DIR, BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParserConfig, ModelServerRef, get_first_server_config_dict
from nemo_gym.config.config_types import is_almost_server, maybe_get_server_instance_config
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.discovery import _parse_no_environment_tolerating_unset_values
from nemo_gym.environment.manifest import load_manifest, resolve_manifest_config_path
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition
from nemo_gym.metrics import compute_aggregate_metrics
from nemo_gym.prompt import fill_prompt, load_prompt_config
from nemo_gym.sandbox import resolve_provider_config
from resources_servers.equivalence_llm_judge.app import (
    JudgeEvaluation,
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
)
from responses_api_agents.simple_agent.app import SimpleAgentVerifyResponse


REPO_ROOT = Path(__file__).parents[3]
HLE_DIAMOND_DIR = REPO_ROOT / "benchmarks" / "hle_diamond"
HLE_JUDGE_PROMPT = REPO_ROOT / "benchmarks" / "hle" / "prompts" / "judge.txt"
HLE_DIAMOND_PROMPTS = HLE_DIAMOND_DIR / "prompts"
# SYSTEM_PROMPT of hle_eval/run_model_predictions.py in centerforaisafety/hle at the commit named in
# prompts/default.yaml: the string the published table was measured with, single braces included.
UPSTREAM_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
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
    monkeypatch.setattr(
        "nemo_gym.config.global_config.maybe_get_global_config_dict", lambda: {"hf_token": "hf-fixture"}
    )
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
        state["rows"] = _source_rows()[:-1]  # incomplete corpus (2 text + 1 image rows): only --limit may write it

        output = prepare_text_module.prepare(limit=1)

        # The limit applies after the text filter, so the two text rows truncate to the first one.
        assert [row["uuid"] for row in _read_rows(output)] == ["q-reasoning-text"]
        out = capsys.readouterr().out
        assert "skipping the corpus completeness check" in out
        # Skipped images are counted from the source rows, not inflated by the limit-trimmed text row.
        assert "Wrote 1 problems" in out and "(skipped 1 image questions, limited to 1)" in out

    def test_limit_truncates_the_full_set_and_a_limit_above_the_corpus_writes_every_row(self, prepared_env, capsys):
        full = _read_rows(prepare_module.prepare(limit=1))
        assert [row["uuid"] for row in full] == ["q-reasoning-text"]
        assert list(full[0]) == ROW_FIELDS + ["responses_create_params"]
        assert "(0 with an image, limited to 1)" in capsys.readouterr().out

        text = _read_rows(prepare_text_module.prepare(limit=10))
        assert [row["uuid"] for row in text] == ["q-reasoning-text", "q-knowledge-mc", "q-knowledge-text"]
        out = capsys.readouterr().out
        assert (
            "writing a SUBSET" in out
            and "Wrote 3 problems" in out
            and "(skipped 1 image questions, limited to 10)" in out
        )

    def test_prepare_text_script_parses_limit_and_refuses_other_arguments(self, prepared_env, capsys):
        output = prepare_text_module.main(["--limit", "1"])
        assert [row["uuid"] for row in _read_rows(output)] == ["q-reasoning-text"]

        with pytest.raises(SystemExit):
            prepare_text_module.main(["--text-only"])
        assert "unrecognized arguments: --text-only" in capsys.readouterr().err

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


class TestPrompts:
    def test_default_prompt_renders_the_upstream_system_prompt_verbatim(self):
        prompt = load_prompt_config(str(HLE_DIAMOND_PROMPTS / "default.yaml"))

        system, user = fill_prompt(prompt, {"question": "What is 2+2?"})

        assert system == {"role": "system", "content": UPSTREAM_SYSTEM_PROMPT}
        assert user == {"role": "user", "content": "What is 2+2?"}
        # The file holds nothing but the two templates; its braces are doubled only for str.format_map.
        shipped = yaml.safe_load((HLE_DIAMOND_PROMPTS / "default.yaml").read_text(encoding="utf-8"))
        assert shipped == {
            "system": UPSTREAM_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}"),
            "user": "{question}",
        }

    def test_opencode_prompt_folds_the_verbatim_instruction_after_the_question(self):
        prompt = load_prompt_config(str(HLE_DIAMOND_PROMPTS / "opencode.yaml"))

        assert prompt.system is None
        (user,) = fill_prompt(prompt, {"question": "What is 2+2?"})
        assert user == {"role": "user", "content": "What is 2+2?\n\n" + UPSTREAM_SYSTEM_PROMPT}

    def test_prepare_bakes_exactly_the_upstream_system_prompt(self, prepared_env):
        assert prepare_module.PROMPT_CONFIG_FPATH == HLE_DIAMOND_PROMPTS / "default.yaml"

        rows = _read_rows(prepare_module.prepare())

        assert [row["responses_create_params"]["input"][0] for row in rows] == [
            {"role": "system", "content": UPSTREAM_SYSTEM_PROMPT}
        ] * 4


# benchmark name -> (dataset name, agent, prompt_config)
LISTED_BENCHMARKS = {
    "hle_diamond": ("hle_diamond", "hle_diamond_equivalence_llm_judge_simple_agent", None),
    "hle_diamond/text": (
        "hle_diamond_text",
        "hle_diamond_text_equivalence_llm_judge_simple_agent",
        "benchmarks/hle_diamond/prompts/default.yaml",
    ),
    "hle_diamond/opencode": (
        "hle_diamond_text",
        "hle_diamond_opencode_agent",
        "benchmarks/hle_diamond/prompts/opencode.yaml",
    ),
    "hle_diamond/pi": ("hle_diamond_text", "hle_diamond_pi_agent", "benchmarks/hle_diamond/prompts/opencode.yaml"),
}
JUDGE_SERVER = "hle_diamond_equivalence_llm_judge_resources_server"


def _resolve_paths(config_paths: list[str]):
    """Compose config paths in the given order the way listing does (runtime-only values tolerated)."""
    initial = OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, {"config_paths": config_paths})
    return _parse_no_environment_tolerating_unset_values(initial)


def _resolve(*config_files: str):
    """Resolve hle_diamond config files, in the given order, the way listing does."""
    return _resolve_paths([f"benchmarks/hle_diamond/{name}" for name in config_files])


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

    @pytest.mark.parametrize(
        "config_file",
        ["config.yaml", "text.yaml", "opencode.yaml", "opencode_search.yaml", "pi.yaml", "pi_search.yaml"],
    )
    def test_every_preset_shares_the_hle_judge_prompt_labels_and_partition_slices(self, config_file):
        judge = get_first_server_config_dict(_resolve(config_file), JUDGE_SERVER)

        # judge_prompt_template_fpath is read relative to the resources server directory.
        prompt = (
            REPO_ROOT / "resources_servers" / "equivalence_llm_judge" / judge["judge_prompt_template_fpath"]
        ).resolve()
        assert prompt == HLE_JUDGE_PROMPT.resolve() and prompt.is_file()
        assert (judge["judge_equal_label"], judge["judge_not_equal_label"]) == ("Judgement: yes", "Judgement: no")
        assert judge["judge_labels_case_insensitive"] is True  # D29: "Judgement: Yes" must not be no_verdict
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

    def test_judge_knobs_default_off_and_only_hle_diamond_enables_them(self):
        """The shared judge server's slice hook and case-insensitive labels stay off for hle and hle_verified."""
        fields = LLMJudgeResourcesServerConfig.model_fields
        assert fields["report_slice_fields"].default_factory() == []
        assert fields["judge_labels_case_insensitive"].default is False
        base = yaml.safe_load(
            (REPO_ROOT / "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml").read_text()
        )["equivalence_llm_judge"]["resources_servers"]["equivalence_llm_judge"]
        assert not {"report_slice_fields", "judge_labels_case_insensitive"} & set(base)
        for sibling in ("hle", "hle_verified"):
            sibling_text = (REPO_ROOT / "benchmarks" / sibling / "config.yaml").read_text(encoding="utf-8")
            assert "judge_labels_case_insensitive" not in sibling_text and "report_slice_fields" not in sibling_text

        shipped = yaml.safe_load((HLE_DIAMOND_DIR / "config.yaml").read_text())[JUDGE_SERVER]["resources_servers"][
            "equivalence_llm_judge"
        ]
        assert shipped["report_slice_fields"] == ["partition"]
        assert shipped["judge_labels_case_insensitive"] is True

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


class TestReasoningEffort:
    """D4 on the sandboxed path: Gym's ++responses_create_params.reasoning.effort decorates only Gym's run
    request, so each harness has to request high effort in its own model call."""

    @pytest.mark.parametrize("config_file", ["opencode.yaml", "opencode_search.yaml"])
    def test_opencode_presets_make_opencode_request_high_reasoning_effort(self, config_file):
        opencode_config = get_first_server_config_dict(_resolve(config_file), "hle_diamond_opencode_agent")[
            "opencode_config"
        ]

        assert opencode_config["provider"]["nemo_gym"]["models"]["dummy_model"]["options"]["reasoningEffort"] == "high"
        # Merged into the inherited OpenCode config, not substituted for it.
        assert list(opencode_config["instructions"]) == ["/tmp/nemo-gym-environment.txt"]
        assert opencode_config["permission"]["webfetch"] == "deny"

    @pytest.mark.parametrize("config_file", ["pi.yaml", "pi_search.yaml"])
    def test_pi_presets_pass_thinking_high(self, config_file):
        agent = get_first_server_config_dict(_resolve(config_file), "hle_diamond_pi_agent")

        # Pi's --thinking levels are off, minimal, low, medium, high, xhigh, max.
        assert agent["thinking"] == "high"
        assert agent["model_server"]["name"] == "policy_model"  # inherited preset keys survive


class TestLocalDockerOverride:
    """sandbox_docker_local.yaml: the sandboxed presets on the local Docker provider (comparison infrastructure)."""

    OVERRIDE = HLE_DIAMOND_DIR / "sandbox_docker_local.yaml"

    @pytest.mark.parametrize(
        ("config_file", "agent", "other_agent"),
        [
            ("opencode.yaml", "hle_diamond_opencode_agent", "hle_diamond_pi_agent"),
            ("opencode_search.yaml", "hle_diamond_opencode_agent", "hle_diamond_pi_agent"),
            ("pi.yaml", "hle_diamond_pi_agent", "hle_diamond_opencode_agent"),
            ("pi_search.yaml", "hle_diamond_pi_agent", "hle_diamond_opencode_agent"),
        ],
    )
    def test_override_resolves_to_exactly_the_docker_provider(self, config_file, agent, other_agent):
        resolved = _resolve(config_file, "sandbox_docker_local.yaml")
        agent_config = get_first_server_config_dict(resolved, agent)

        assert (agent_config["sandbox_provider"], agent_config["network_access"]) == ("sandbox_docker", "inherit")
        provider = resolve_provider_config(agent_config["sandbox_provider"], resolved)
        assert list(provider) == ["docker"]
        assert provider["docker"]["create"]["keepalive_shell"] == "/bin/sh"
        # The preset's OpenSandbox block is still composed, which is why the placeholder env vars exist.
        assert "opensandbox" in resolved["sandbox"]
        # The other harness's block is partial: not a server, not an almost-server, so never started.
        block = resolved[other_agent]
        instance, error = maybe_get_server_instance_config(name=other_agent, server_type_config_dict=block)
        assert instance is None and error is not None
        assert not is_almost_server(block)

    def test_override_tells_the_model_the_truth_about_egress_and_keeps_the_preset_knobs(self):
        opencode = get_first_server_config_dict(
            _resolve("opencode_search.yaml", "sandbox_docker_local.yaml"), "hle_diamond_opencode_agent"
        )
        note = opencode["sandbox_config"]["files"]["/tmp/nemo-gym-environment.txt"]
        assert "NOT restricted" in note and "disabled" not in note
        assert opencode["sandbox_config"]["ttl_s"] == 14400  # the rest of the inherited sandbox_config survives
        assert opencode["opencode_config"]["provider"]["nemo_gym"]["models"]["dummy_model"]["options"] == {
            "reasoningEffort": "high"
        }

        pi = get_first_server_config_dict(
            _resolve("pi_search.yaml", "sandbox_docker_local.yaml"), "hle_diamond_pi_agent"
        )
        assert "NOT restricted" in pi["system_prompt"] and "disabled" not in pi["system_prompt"]
        assert pi["thinking"] == "high"

    def test_override_is_not_a_benchmark_and_documents_the_local_recipe(self):
        text = self.OVERRIDE.read_text(encoding="utf-8")
        for documented in (
            "OPENSANDBOX_API_KEY",
            "OPENSANDBOX_DOMAIN",
            "# network: host",
            "ttl_s",
            "LABEL EVERY RESULT",
        ):
            assert documented in text
        assert "datasets" not in json.dumps(yaml.safe_load(text))

        discovered = {p.name for p in _benchmark_config_paths(BENCHMARKS_DIR) if p.is_relative_to(HLE_DIAMOND_DIR)}
        assert discovered == {"config.yaml", "text.yaml", "opencode.yaml", "pi.yaml"}


README = HLE_DIAMOND_DIR / "README.md"
OVERRIDE_FILES = ("judge_openai.yaml", "sandbox_docker_local.yaml")
CONFIG_PATHS_PREFIX = "+config_paths=["


def _documented_command_lines() -> list[str]:
    """Every documented `gym eval run` line (full commands and `...` fragments) in the README and in the
    two override headers, with backslash continuations joined and comment markers stripped."""
    lines: list[str] = []
    for path in (README, *(HLE_DIAMOND_DIR / name for name in OVERRIDE_FILES)):
        text = re.sub(r"\\\n#?[ \t]*", " ", path.read_text(encoding="utf-8"))
        stripped = (line.strip().lstrip("# ") for line in text.splitlines())
        lines += [line for line in stripped if line.startswith("gym eval run ")]
    return lines


def _documented_bash_commands(prefix: str) -> list[list[str]]:
    """The README's complete commands starting with ``prefix`` (fenced bash blocks, continuations joined,
    no `...` elision) as argv lists without the leading `gym`; placeholders become parseable tokens."""
    commands: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.strip()
            if line.startswith(prefix) and not line.endswith("..."):
                commands.append(shlex.split(line.replace("<gateway model id>", "policy"))[1:])
    return commands


def _documented_run_commands() -> dict[str, list[str]]:
    """The README's complete `gym eval run --benchmark` commands keyed by their --benchmark value."""
    return {
        argv[argv.index("--benchmark") + 1]: argv
        for argv in _documented_bash_commands("gym eval run ")
        if "--benchmark" in argv
    }


def _parse_cli(subcommand: list[str], argv: list[str]):
    """The argparse namespace, the Hydra tokens the CLI's flags translate to, and the raw Hydra overrides
    (including one argparse captured as the optional TARGET positional, which main() restores)."""
    from nemo_gym.cli.main import _is_positional_hydra_override, build_parser

    args, overrides = build_parser().parse_known_args([*subcommand, *argv])
    translated = [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]
    target = getattr(args, "target", None)
    if target and _is_positional_hydra_override(target):
        overrides = [target, *overrides]
    return args, translated, overrides


def _cli_config_paths(argv: list[str], subcommand: tuple[str, ...] = ("eval", "run")) -> tuple[list[str], object]:
    """The config_paths `gym <subcommand> <argv>` composes, produced by the CLI's own flag translation and
    coalescing (nemo_gym.cli.main.main), plus the parsed namespace."""
    from nemo_gym.cli.main import _merge_config_paths

    args, translated, overrides = _parse_cli(list(subcommand), argv)
    (token,) = [t for t in _merge_config_paths(translated + overrides) if t.startswith(CONFIG_PATHS_PREFIX)]
    return token[len(CONFIG_PATHS_PREFIX) : -1].split(","), args


class TestDocumentedCommands:
    """The README's run commands, composed exactly as `gym eval run` composes them (D31).

    The presets set `judge_model_server` (and the search presets `network_access` and the environment
    note) directly, and a later sibling in config_paths wins, so the overrides must merge after the
    preset. The raw `+config_paths=[...]` token is appended after every translated flag (--benchmark,
    --model-type) wherever it stands on the command line; `--config` is translated before --benchmark
    in this Gym version, so a `--config` override merges first and loses those keys.
    """

    COMMANDS = _documented_run_commands()

    @pytest.mark.parametrize("benchmark", sorted(COMMANDS))
    def test_readme_commands_compose_the_overrides_after_the_preset_so_they_win(self, benchmark, monkeypatch):
        monkeypatch.delenv("HLE_DIAMOND_JUDGE_MODEL", raising=False)
        paths, args = _cli_config_paths(self.COMMANDS[benchmark])

        assert args.split == "benchmark"
        names = [Path(p).name for p in paths]
        (preset,) = [
            p
            for p in paths
            if Path(p).resolve().parent == HLE_DIAMOND_DIR.resolve() and Path(p).name not in OVERRIDE_FILES
        ]
        assert names.index(Path(preset).name) < names.index("judge_openai.yaml")

        resolved = _resolve_paths(paths)
        assert (
            get_first_server_config_dict(resolved, JUDGE_SERVER)["judge_model_server"]["name"]
            == "hle_diamond_judge_model"
        )
        if "sandbox_docker_local.yaml" in names:
            assert names.index(Path(preset).name) < names.index("sandbox_docker_local.yaml")
            agent_name = BenchmarkConfig.from_config_path(Path(preset), strict=False).agent_name
            agent = get_first_server_config_dict(resolved, agent_name)
            assert (agent["sandbox_provider"], agent["network_access"]) == ("sandbox_docker", "inherit")
            assert list(resolve_provider_config(agent["sandbox_provider"], resolved)) == ["docker"]
            note = agent["sandbox_config"]["files"].get("/tmp/nemo-gym-environment.txt") or agent["system_prompt"]
            assert "NOT restricted" in note and "disabled" not in note

    @pytest.mark.parametrize(
        ("benchmark", "override"),
        [("hle_diamond", "judge_openai.yaml"), ("hle_diamond/opencode_search", "sandbox_docker_local.yaml")],
    )
    def test_config_flag_merges_before_the_preset_so_the_overrides_lose(self, benchmark, override, monkeypatch):
        """Why no documented command passes these files with --config. If this fails, the CLI's flag order
        changed: re-check the README's composition-order paragraph before deleting this case."""
        monkeypatch.delenv("HLE_DIAMOND_JUDGE_MODEL", raising=False)
        # --config placed last on the command line still lands first in config_paths.
        paths, _ = _cli_config_paths(
            ["--benchmark", benchmark, "--split", "benchmark", "--config", f"benchmarks/hle_diamond/{override}"]
        )
        assert Path(paths[0]).name == override

        resolved = _resolve_paths(paths)
        if override == "judge_openai.yaml":
            assert get_first_server_config_dict(resolved, JUDGE_SERVER)["judge_model_server"]["name"] == "policy_model"
        else:
            agent = get_first_server_config_dict(resolved, "hle_diamond_opencode_agent")
            assert agent["network_access"] == "model_and_tools"
            assert "disabled" in agent["sandbox_config"]["files"]["/tmp/nemo-gym-environment.txt"]

    def test_every_documented_run_line_carries_split_and_the_raw_override_token(self):
        lines = _documented_command_lines()
        assert len(lines) >= 5  # the README's full commands, fragment and --no-serve run, and both override headers

        for line in lines:
            if "--no-serve" in line:
                # Running the full set: the dataset is --input and --split is not read; the servers (and the
                # judge override, via `gym env start --config`) are already up.
                assert "--split" not in line and "--benchmark" not in line, line
                assert "--agent hle_diamond_equivalence_llm_judge_simple_agent" in line, line
                assert "++route_failures_to_sidecar=true" in line, line
                continue
            assert "--split benchmark" in line, line
            assert "--config " not in line, line  # merges before the preset: never for these overrides
            if any(name in line for name in OVERRIDE_FILES):
                assert f"'{CONFIG_PATHS_PREFIX}" in line, line
            if "sandbox_docker_local.yaml" in line:
                assert "judge_openai.yaml" in line, line  # otherwise a copied command self-judges

    def test_full_set_servers_compose_the_judge_override_after_the_preset(self, monkeypatch):
        """Running the full set: under `gym env start` every --config lands in command-line order."""
        monkeypatch.delenv("HLE_DIAMOND_JUDGE_MODEL", raising=False)
        (argv,) = _documented_bash_commands("gym env start ")

        paths, _ = _cli_config_paths(argv, subcommand=("env", "start"))
        _, _, overrides = _parse_cli(["env", "start"], argv)

        assert [Path(p).name for p in paths] == ["litellm_model.yaml", "config.yaml", "judge_openai.yaml"]
        assert "++policy_model_name=policy" in overrides and "++observability_enabled=true" in overrides
        resolved = _resolve_paths(paths)
        judge = get_first_server_config_dict(resolved, JUDGE_SERVER)
        assert judge["judge_model_server"]["name"] == "hle_diamond_judge_model"
        assert judge["judge_labels_case_insensitive"] is True
        assert (
            get_first_server_config_dict(resolved, "hle_diamond_judge_model")["openai_model"] == "azure/openai/o3-mini"
        )
        assert get_first_server_config_dict(resolved, "policy_model")["entrypoint"] == "app.py"

    def test_full_set_collection_runs_no_serve_with_sidecar_routing_and_only_the_effort_setting(self):
        (argv,) = [argv for argv in _documented_bash_commands("gym eval run ") if "--no-serve" in argv]

        args, translated, overrides = _parse_cli(["eval", "run"], argv)

        assert args.no_serve is True and args.split is None and args.limit is None
        assert (args.agent, args.input) == (
            "hle_diamond_equivalence_llm_judge_simple_agent",
            "benchmarks/hle_diamond/data/hle_diamond_benchmark.jsonl",
        )
        assert "+num_repeats=1" in translated
        # D4: the one sampling setting; D33: failures go to the sidecar instead of ending the run.
        assert {"++responses_create_params.reasoning.effort=high", "++route_failures_to_sidecar=true"} <= set(
            overrides
        )
        assert (args.temperature, args.top_p, args.max_output_tokens) == (None, None, None)


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
        # Unjudged rows still produce no custom metrics at all, exactly as before the hook, whether or
        # not slices are configured: the hook runs after the early return, so it never fights the
        # `if not judged` block that other metric hooks (PR #4658) also edit.
        assert server.compute_metrics([[{"reward": 1.0, "partition": "reasoning"}]]) == {}
        assert _judge_server(["partition"]).compute_metrics([[{"reward": 1.0, "partition": "reasoning"}]]) == {}


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                type="message",
                role="assistant",
                status="completed",
                content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestVerifyResponseWireFormat:
    def test_partition_survives_the_verify_response_and_reaches_the_slices(self, prepared_env):
        """Production row shape: /verify reply -> SimpleAgentVerifyResponse -> saved rollout -> /aggregate_metrics."""
        prepared = _read_rows(prepare_text_module.prepare())[0]  # q-reasoning-text, as the collector materializes it
        server = _judge_server(["partition"])
        request = LLMJudgeVerifyRequest(
            **prepared,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_response("Explanation: trivial.\nAnswer: 4\nConfidence: 100%"),
        )
        evaluation = JudgeEvaluation(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_response("Reasoning: same.\nJudgement: yes\nConfidence: 100"),
            verdict_label="Judgement: yes",
        )

        wire = server._make_response(request, prepared["expected_answer"], 1.0, [evaluation]).model_dump(mode="json")
        row = SimpleAgentVerifyResponse.model_validate(wire).model_dump()

        assert (row["uuid"], row["partition"], row["has_image"]) == ("q-reasoning-text", "reasoning", False)
        assert (row["expected_answer"], row["reward"]) == ("4", 1.0)
        row.update({"_ng_task_index": 0, "_ng_rollout_index": 0})
        result = compute_aggregate_metrics(
            [row], compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
        )
        assert result.key_metrics["slices/partition/reasoning/mean/reward"] == 1.0
        assert result.agent_metrics["slices/partition/missing_label_rollouts"] == 0
