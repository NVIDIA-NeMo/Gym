# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from benchmarks.indic.arena_proxy_v3 import prepare as module
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment.manifest import load_manifest


ROOT = Path(__file__).resolve().parents[3]


def _records(*, language: str = "hi", multi_turn: bool = False, creative: bool = False) -> tuple[dict, dict]:
    messages = [{"role": "user", "content": "पहला प्रश्न?"}]
    if multi_turn:
        messages.extend(
            [{"role": "assistant", "content": "Earlier assistant answer."}, {"role": "user", "content": "क्यों?"}]
        )
    question = (
        messages[0]["content"]
        if not multi_turn
        else "\n\n".join(f"[{message['role'].capitalize()}]: {message['content']}" for message in messages)
    )
    group = module.CREATIVE_GROUP if creative else "Information & advice"
    source = {
        "question_id": "question-1",
        "question": question,
        "responses_create_params": {"input": messages},
        "baseline_answer": "DO NOT COPY: original LMArena baseline",
        "other_answer": "DO NOT COPY: held-out comparison answer",
        "metadata": {"chat400": {"group": group, "num_user_turns": 2 if multi_turn else 1}},
    }
    if language != "en":
        source["metadata"]["chat400"]["translation"] = {"lang": language, "status": "ok"}
    reference = {
        "question_id": "question-1",
        "row_id": "question-1",
        "language": language,
        "category": group,
        "num_user_turns": 2 if multi_turn else 1,
        "question": question,
        "messages_sha256": module.messages_sha256(messages),
        "generation": "Frozen baseline response.",
        "status": "ok",
        "model": "openai/openai/gpt-5.4-mini",
    }
    return source, reference


@pytest.mark.parametrize("multi_turn", [False, True])
@pytest.mark.parametrize("creative", [False, True])
def test_preserves_entire_conversation_and_replaces_original_baseline(multi_turn: bool, creative: bool) -> None:
    source, reference = _records(multi_turn=multi_turn, creative=creative)
    rows, excluded = module.convert_language([source], [reference], language="hi", expected_rows=1)
    row = rows[0]
    assert excluded == []
    assert row["responses_create_params"] == source["responses_create_params"]
    assert row["responses_create_params"] is not source["responses_create_params"]
    assert row["question"] == source["question"]
    assert row["baseline_answer"] == reference["generation"]
    assert row["baseline_model"] == "gpt-5.4-mini"
    assert row["uid"] == "hi:question-1"
    assert row["category"] == ("creative_writing" if creative else "hard_prompt")
    assert row["num_user_turns"] == (2 if multi_turn else 1)
    assert "DO NOT COPY" not in json.dumps(row)


@pytest.mark.parametrize("status,kept", [("ok", True), ("low_score", True), ("hard_error", False)])
def test_records_failed_translations_and_retains_low_score(status: str, kept: bool) -> None:
    source, reference = _records()
    source["metadata"]["chat400"]["translation"]["status"] = status
    rows, excluded = module.convert_language([source], [reference], language="hi", expected_rows=1)
    assert len(rows) == int(kept)
    assert excluded == ([] if kept else ["question-1"])


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("question_id", "different", "reference IDs differ"),
        ("row_id", "different", "row_id or language"),
        ("language", "en", "row_id or language"),
        ("question", "Different conversation", "reference conversation"),
        ("messages_sha256", "stale-digest", "messages_sha256"),
        ("category", "Math / reasoning", "reference group"),
        ("num_user_turns", 3, "num_user_turns"),
        ("status", "failed", "reference status"),
        ("model", "another-model", "expected gpt-5.4-mini"),
        ("generation", "", "reference generation"),
    ],
)
def test_rejects_stale_or_failed_reference(field: str, value: object, error: str) -> None:
    source, reference = _records()
    reference[field] = value
    with pytest.raises(ValueError, match=error):
        module.convert_language([source], [reference], language="hi", expected_rows=1)


def test_requires_exact_counts_and_unique_source_and_reference_ids() -> None:
    source, reference = _records()
    with pytest.raises(ValueError, match="expected 400 source rows"):
        module.convert_language([source], [reference], language="hi")
    with pytest.raises(ValueError, match="duplicate question_id"):
        module.convert_language([source, source], [reference], language="hi", expected_rows=2)
    with pytest.raises(ValueError, match="duplicate question_id"):
        module.convert_language([source], [reference, reference], language="hi", expected_rows=1)
    with pytest.raises(ValueError, match="reference IDs differ"):
        module.convert_language([source], [], language="hi", expected_rows=1)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda s: s.update(question="Flattened last turn only"), "full conversation"),
        (lambda s: s["responses_create_params"].update(input=[]), "nonempty message list"),
        (lambda s: s["responses_create_params"]["input"].pop(), "end with a user"),
        (lambda s: s["responses_create_params"]["input"][1].update(role="user"), "must have role assistant"),
        (lambda s: s["responses_create_params"]["input"][1].update(content=""), "nonempty text"),
        (lambda s: s["metadata"]["chat400"].update(num_user_turns=1), "num_user_turns"),
        (lambda s: s["metadata"]["chat400"].update(group="unknown"), "unknown chat-400 group"),
        (lambda s: s["metadata"]["chat400"]["translation"].update(lang="en"), "translation language"),
        (lambda s: s["metadata"]["chat400"]["translation"].update(status="failed"), "unknown translation status"),
    ],
)
def test_rejects_corrupt_conversation_and_translation(mutation, error: str) -> None:
    source, reference = _records(multi_turn=True)
    mutation(source)
    with pytest.raises(ValueError, match=error):
        module.convert_language([source], [reference], language="hi", expected_rows=1)


def _write_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, languages: tuple[str, ...]) -> Path:
    source_dir = tmp_path / "source"
    checksums = {}
    for language in languages:
        source, reference = _records(language=language)
        second_source, second_reference = deepcopy(source), deepcopy(reference)
        second_source["question_id"] = "question-2"
        second_reference.update(question_id="question-2", row_id="question-2")
        for filename, rows in (
            (f"lmarena_v3_chat400_{language}.jsonl", [source, second_source]),
            (f"references/gpt-5.4-mini/{language}.jsonl", [reference, second_reference]),
        ):
            path = source_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            checksums[filename] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(module, "SOURCE_CHECKSUMS", checksums)
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    return source_dir


def test_prepare_reports_full_validation_before_limiting_and_hashes(tmp_path: Path, monkeypatch) -> None:
    source_dir = _write_source(tmp_path, monkeypatch, languages=("en", "hi"))
    output_dir = tmp_path / "prepared"
    output = module.prepare(source_dir=source_dir, languages=["en", "hi"], output_dir=output_dir, limit=1)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["uid"] for row in rows] == ["en:question-1", "hi:question-1"]
    assert json.loads((output_dir / "arena_proxy_v3_hi.jsonl").read_text()) == rows[1]
    report = json.loads((output_dir / "preparation_report.json").read_text())
    assert report["source_revision"] == module.SOURCE_REVISION
    assert report["coverage"]["hi"] == {
        "source_rows": 2,
        "eligible_rows": 2,
        "prepared_rows": 1,
        "excluded_untranslated": [],
    }
    assert report["prepared_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(report["files_sha256"]) == 4
    assert "DO NOT COPY" not in output.read_text()


def test_limit_matches_reference_source_prefix_before_translation_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = _write_source(tmp_path, monkeypatch, languages=("ta",))
    filename = "lmarena_v3_chat400_ta.jsonl"
    source_path = source_dir / filename
    sources = [json.loads(line) for line in source_path.read_text().splitlines()]
    sources[0]["metadata"]["chat400"]["translation"]["status"] = "hard_error"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in sources))
    monkeypatch.setitem(
        module.SOURCE_CHECKSUMS, filename, {"sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}
    )

    output = module.prepare(source_dir=source_dir, languages=["ta"], output_dir=tmp_path / "limited", limit=1)
    assert output.read_text() == ""
    report = json.loads((output.parent / "preparation_report.json").read_text())
    assert report["coverage"]["ta"]["eligible_rows"] == 1
    assert report["coverage"]["ta"]["prepared_rows"] == 0
    full_output = module.prepare(source_dir=source_dir, languages=["ta"], output_dir=tmp_path / "full")
    assert json.loads(full_output.read_text())["uid"] == "ta:question-2"


def test_source_content_change_rejected_before_replacing_prepared_output(tmp_path: Path, monkeypatch) -> None:
    source_dir = _write_source(tmp_path, monkeypatch, languages=("hi",))
    output_dir = tmp_path / "prepared"
    output = module.prepare(source_dir=source_dir, languages=["hi"], output_dir=output_dir)
    before = output.read_bytes()
    path = source_dir / "lmarena_v3_chat400_hi.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        module.prepare(source_dir=source_dir, languages=["hi"], output_dir=output_dir)
    assert output.read_bytes() == before


def test_downloads_use_frozen_hub_revision_and_original_filenames(monkeypatch, tmp_path: Path) -> None:
    import huggingface_hub

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / kwargs["filename"])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    paths = module._source_paths(source_dir=None, language="hi", revision=module.SOURCE_REVISION)
    assert len(module.SOURCE_REVISION) == 40
    assert calls == [
        {
            "repo_id": module.SOURCE_ID,
            "repo_type": "dataset",
            "revision": module.SOURCE_REVISION,
            "filename": filename,
            "token": True,
        }
        for filename in ("lmarena_v3_chat400_hi.jsonl", "references/gpt-5.4-mini/hi.jsonl")
    ]
    assert paths == tuple(tmp_path / call["filename"] for call in calls)


@pytest.mark.parametrize("languages", [[], ["hi", "hi"], ["as"], ["en,hi"]])
def test_rejects_bad_language_selection(languages: list[str], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="languages must be unique"):
        module.prepare(source_dir=tmp_path, languages=languages)


def test_conversation_prompts_reuse_in_repo_arena_templates() -> None:
    for filename in ("arena.yaml", "arena_creative.yaml"):
        original = yaml.safe_load((ROOT / "resources_servers/arena_judge/prompts" / filename).read_text())
        actual = yaml.safe_load((module.DIRECTORY / "prompts" / filename).read_text())
        expected = {
            key: value.replace("the user prompt displayed below", "the user conversation displayed below")
            .replace("your own answer to the prompt", "your own last answer to the conversation")
            .replace("<|User Prompt|>", "<|Conversation History|>")
            for key, value in original.items()
        }
        assert actual == expected
        assert "{question}" in actual["user"]


def test_manifest_and_config_keep_prepared_messages_and_shared_gym_components() -> None:
    manifest = load_manifest(module.DIRECTORY / "manifest.yaml")
    assert manifest.prompt_source == "prepared"
    assert manifest.resources_server == "arena_judge"
    assert manifest.agent_server == "simple_agent"
    assert manifest.licensing == "unknown"
    config = yaml.safe_load((module.DIRECTORY / "config.yaml").read_text())
    agent = config["indic_arena_proxy_v3_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["max_steps"] == 1
    assert "prompt_config" not in agent["datasets"][0]
    judge = yaml.safe_load((module.DIRECTORY / "judge.yaml").read_text())
    assert set(judge) == {"indic_arena_proxy_v3_judge_model", "indic_arena_proxy_v3_resources_server"}
    resources = judge["indic_arena_proxy_v3_resources_server"]["resources_servers"]["arena_judge"]
    assert resources["judge_chat_completions_create_params"] == {
        "messages": [],
        "max_tokens": 32768,
        "temperature": 0.0,
        "reasoning_effort": "low",
    }
    assert resources["strip_thinking"] and resources["sanitize_generations"]
    assert resources["arena_elo_bootstrap_rounds"] == 100


@pytest.mark.parametrize("judge_only", [False, True])
def test_config_resolves_without_policy_on_cpu(judge_only: bool) -> None:
    config_path = module.DIRECTORY / ("judge.yaml" if judge_only else "config.yaml")
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(config_path)], "error_on_agent_without_environment_server": True},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert "indic_arena_proxy_v3_resources_server" in resolved
    assert ("indic_arena_proxy_v3_agent" in resolved) is not judge_only
    if not judge_only:
        environment = resolved["indic_arena_proxy_v3_environment_server"]["environment_servers"][
            "single_agent_turn_legacy"
        ]
        assert environment["agent_server"]["name"] == "indic_arena_proxy_v3_agent"
        assert environment["resources_server"]["name"] == "indic_arena_proxy_v3_resources_server"
