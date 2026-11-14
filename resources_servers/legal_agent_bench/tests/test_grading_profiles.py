# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score/profile regressions use real scoring and file artifacts with an offline judge."""

import json
from pathlib import Path

import pytest

from resources_servers.legal_agent_bench import verifier
from resources_servers.legal_agent_bench.lab_harbor import scoring
from resources_servers.legal_agent_bench.lab_harbor.completion import CompletionState
from resources_servers.legal_agent_bench.lab_harbor.judge import VERDICT_SCHEMA
from resources_servers.legal_agent_bench.lab_harbor.judge_config import STANDARD_JUDGE_MODELS
from resources_servers.legal_agent_bench.lab_harbor.tools import get_all_tool_definitions


def _run_args(tmp_path: Path):
    run = tmp_path / "run"
    (run / "output").mkdir(parents=True)
    (run / "output" / "response.md").write_text("A supported legal answer.")
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "title": "Review",
                "instructions": "Write response.md.",
                "criteria": [
                    {
                        "id": "C-1",
                        "title": "Answer",
                        "match_criteria": "Correct answer",
                        "deliverables": ["response.md"],
                    }
                ],
                "metadata": {"lab_source_revision": "pinned-test-revision"},
            }
        )
    )
    return verifier.build_parser().parse_args(
        [
            "--task-json",
            str(task),
            "--run-dir",
            str(run),
            "--verifier-dir",
            str(tmp_path / "verification"),
            "--reward-json",
            str(tmp_path / "verification" / "reward.json"),
            "--judge-base-url",
            "http://judge.test/v1",
        ]
    )


@pytest.mark.parametrize(
    "verdicts, expected", [(["fail", "fail"], 0.0), (["pass", "fail"], 0.5), (["pass", "pass"], 1.0)]
)
@pytest.mark.parametrize("reward_mode", ["full_task", "criteria_pass_rate"])
def test_official_dual_averages_selected_reward_mode_and_retains_per_judge(
    tmp_path, monkeypatch, verdicts, expected, reward_mode
):
    monkeypatch.setenv("LEGAL_AGENT_BENCH_REWARD_MODE", reward_mode)
    args = _run_args(tmp_path)
    task_path = Path(args.task_json)
    task = json.loads(task_path.read_text())
    task["criteria"].append({**task["criteria"][0], "id": "C-2", "match_criteria": "Always-pass control"})
    task_path.write_text(json.dumps(task))
    answers = dict(zip(STANDARD_JUDGE_MODELS, verdicts, strict=True))

    class Judge:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, prompt, *_args, **_kwargs):
            verdict = "pass" if "Always-pass control" in prompt else answers[self.model]
            return {"reasoning": "Evidence checked.", "verdict": verdict}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": model} for model in STANDARD_JUDGE_MODELS]))
    result = verifier.run_verifier(args)
    reward = json.loads(Path(args.reward_json).read_text())
    criterion_average = (1 + expected) / 2
    assert reward["reward"] == (expected if reward_mode == "full_task" else criterion_average)
    assert reward["criteria_pass_rate"] == criterion_average
    assert result["reported_reward_mode"] == reward_mode
    assert result["dual_all_pass_rate"] == expected
    assert result["dual_criterion_pass"] == (1 + expected) / 2
    assert result["judge_profile"] == "lab-standard-dual-v1"
    assert result["source_revision"] == "pinned-test-revision"
    assert len(result["per_judge"]) == 2
    assert all((Path(args.verifier_dir) / f"judge-{index}" / "scores.json").is_file() for index in (1, 2))
    assert (Path(args.verifier_dir) / "scores_dual.json").is_file()


def test_single_entry_selects_requested_judge(tmp_path, monkeypatch):
    args = _run_args(tmp_path)

    class Judge:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "custom-policy-judge"
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"reasoning": "Checked.", "verdict": "pass"}

    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": "custom-policy-judge"}]))
    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    assert verifier.run_verifier(args)["score"] == 1.0
    assert not (Path(args.verifier_dir) / "scores_dual.json").exists()


def test_missing_pandoc_stops_before_any_judge_call(tmp_path, monkeypatch):
    args = _run_args(tmp_path)
    (Path(args.run_dir) / "output" / "memo.docx").write_bytes(b"unreadable document")
    monkeypatch.setattr(scoring, "pandoc_version", lambda: None)
    with pytest.raises(RuntimeError, match="pandoc"):
        scoring.score_rubric(
            criteria=[{"id": "C", "title": "Memo", "match_criteria": "Exists"}],
            run_dir=Path(args.run_dir),
            judge=object(),
            task_desc="Review",
        )


def test_corrupt_pptx_is_an_extraction_error(tmp_path):
    path = tmp_path / "deck.pptx"
    path.write_bytes(b"this is not a presentation")
    with pytest.raises(scoring.DeliverableReadError, match="not a zip archive"):
        scoring.read_deliverable_as_text(path)


def test_reasoning_precedes_verdict_and_tools_match_upstream():
    assert list(VERDICT_SCHEMA["properties"]) == ["reasoning", "verdict"]
    assert {tool["name"] for tool in get_all_tool_definitions()} == {
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "finish",
    }


def test_finish_gate_is_bounded_and_records_completion():
    state = CompletionState()
    for _ in range(2):
        assert state.finish(summary="done", missing=["memo.docx"]).startswith("Not finished")
        assert not state.finished
    assert state.finish(summary="done", missing=["memo.docx"]) == "Finished."
    assert state.finished and state.summary == "done"


@pytest.mark.parametrize("printed", [False, True])
def test_docx_comments_include_anchored_passage_without_duplication(tmp_path, monkeypatch, printed):
    import subprocess
    import zipfile

    path = tmp_path / "memo.docx"
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document = f'''<w:document xmlns:w="{namespace}"><w:body><w:p>
      <w:commentRangeStart w:id="1"/><w:r><w:t>Relevant clause</w:t></w:r>
      <w:commentRangeEnd w:id="1"/><w:r><w:commentReference w:id="1"/></w:r>
    </w:p></w:body></w:document>'''
    comments = f'''<w:comments xmlns:w="{namespace}">
      <w:comment w:id="1" w:author="Reviewer"><w:p><w:r><w:t>Fix this clause.</w:t></w:r></w:p></w:comment>
      <w:comment w:id="2"><w:p><w:r><w:t>Orphaned note.</w:t></w:r></w:p></w:comment>
    </w:comments>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/comments.xml", comments)
    output = 'Body {.comment-start id="1"} Fix this clause.' if printed else "Body"
    monkeypatch.setattr(scoring.subprocess, "run", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, output, ""))
    text = scoring.read_deliverable_as_text(path)
    assert text.count("Fix this clause.") == 1
    assert "Orphaned note." not in text
    if not printed:
        assert "Relevant clause" in text and "Reviewer" in text


@pytest.mark.parametrize(
    "selection",
    [
        "omitted",
        None,
        [],
        [{"model_name": "gpt-5.5"}] * 3,
        [{"model_name": "gpt-5.5"}] * 2,
        [{"model_name": "other-a"}, {"model_name": "other-b"}],
        [{"model_name": " "}],
        [{"model_name": 42}],
        [{"model_name": "gpt-5.5", "modle_name": "typo"}],
    ],
)
def test_invalid_judge_selection_fails_before_execution(tmp_path, monkeypatch, selection):
    from resources_servers.legal_agent_bench.app import LegalAgentBenchEnvironmentConfig

    args = _run_args(tmp_path)
    config = {
        "name": "lab",
        "host": "127.0.0.1",
        "port": 0,
        "entrypoint": "app.py",
        "agent_server": {"type": "responses_api_agents", "name": "agent"},
    }
    if selection != "omitted":
        config["judges"] = selection
    with pytest.raises(ValueError, match="judges|lab_judges"):
        LegalAgentBenchEnvironmentConfig.model_validate(config)

    # Legacy settings cannot silently supply an omitted list.
    monkeypatch.setenv("LAB_JUDGE_PROFILE", "single")
    monkeypatch.setenv("LAB_JUDGE_MODEL", "old-judge")
    if selection == "omitted":
        monkeypatch.delenv("LAB_JUDGES", raising=False)
    else:
        monkeypatch.setenv("LAB_JUDGES", json.dumps(selection))
    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", lambda **_kw: pytest.fail("judge must not be called"))
    verifier_dir = Path(args.verifier_dir)
    verifier_dir.mkdir()
    (verifier_dir / "scores_dual.json").write_text('{"score": 1}')
    Path(args.reward_json).write_text('{"reward": 1}')
    with pytest.raises(ValueError, match="lab_judges"):
        verifier.run_verifier(args)
    assert not Path(args.reward_json).exists()
    assert not (verifier_dir / "scores_dual.json").exists()


@pytest.mark.parametrize("dual", [False, True])
def test_judge_list_roundtrips_environment_and_standalone_verifier(tmp_path, monkeypatch, dual):
    from resources_servers.legal_agent_bench.app import LegalAgentBenchEnvironmentConfig, _build_verifier_env

    selection = [{"model_name": "route/gpt-5.5", "base_url": "https://one.example/v1", "api_key": ""}]
    if dual:
        selection.append({"model_name": "route/claude-sonnet-4-6", "reasoning_effort": "high"})
    config = LegalAgentBenchEnvironmentConfig(
        name="lab",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        agent_server={"type": "responses_api_agents", "name": "agent"},
        judges=selection,
        judge_base_url="https://shared.example/v1",
        judge_api_key="shared-test-key",
        judge_parallelism=1,
    )
    for key, value in _build_verifier_env(config).items():
        monkeypatch.setenv(key, value)
    calls = []

    class Judge:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"reasoning": "Checked output.", "verdict": "pass"}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    args = _run_args(tmp_path)
    result = verifier.run_verifier(args)
    assert [call["model"] for call in calls] == [judge["model_name"] for judge in selection]
    assert calls[0]["base_url"] == "https://one.example/v1" and calls[0]["api_key"] == ""
    assert config.judge_profile == ("official_dual" if dual else "single")
    assert result["judge_profile"] == ("lab-standard-dual-v1" if dual else "single")
    assert json.loads(Path(args.reward_json).read_text())["reward"] == 1
    if dual:
        assert calls[1]["base_url"] == "https://shared.example/v1"
        assert calls[1]["api_key"] == "shared-test-key"
        assert calls[1]["reasoning_effort"] == "high"
        assert set(result["per_judge"]) == set(STANDARD_JUDGE_MODELS)
    else:
        assert result["judge_model"] == "route/gpt-5.5"
        assert not (Path(args.verifier_dir) / "scores_dual.json").exists()


@pytest.mark.parametrize("dual", [False, True])
def test_regrading_failure_keeps_current_criteria_without_stale_success(tmp_path, monkeypatch, dual):
    args = _run_args(tmp_path)
    task_path = Path(args.task_json)
    task = json.loads(task_path.read_text())
    task["criteria"].append({**task["criteria"][0], "id": "C-2", "match_criteria": "Connection failure control"})
    task_path.write_text(json.dumps(task))
    failing = False

    class Judge:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, prompt, *_args, **_kwargs):
            if failing and self.model == "gpt-5.5" and "Connection failure control" in prompt:
                raise ConnectionError("judge unavailable")
            return {"reasoning": "Checked.", "verdict": "pass"}

    models = STANDARD_JUDGE_MODELS if dual else ("gpt-5.5",)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": name} for name in models]))
    monkeypatch.setenv("LAB_JUDGE_PARALLELISM", "1")
    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    assert verifier.run_verifier(args)["score"] == 1
    failing = True
    argv = []
    for key in ("task_json", "run_dir", "verifier_dir", "reward_json", "judge_base_url"):
        argv.extend(["--" + key.replace("_", "-"), getattr(args, key)])
    assert verifier.main(argv) == 1
    directory = Path(args.verifier_dir)
    assert not (directory / "scores_dual.json").exists()
    assert json.loads(Path(args.reward_json).read_text())["verifier_error"] == 1
    judge_dir = directory / "judge-2" if dual else directory
    report = json.loads((judge_dir / "scores.json").read_text())
    assert [criterion["verdict"] for criterion in report["criteria_results"]] == ["pass", "error"]
    assert report["judge_model"] == "gpt-5.5"
    assert report["source_revision"] == "pinned-test-revision"
    assert report["judge_error_count"] == 1
    events = [json.loads(line) for line in (judge_dir / "transcript.jsonl").read_text().splitlines()]
    assert sum(event["type"] == "criterion_complete" for event in events) == 2
    if dual:
        assert not (judge_dir / "reward.json").exists()
        assert json.loads((directory / "judge-1/reward.json").read_text())["reward"] == 1
    failing = False
    assert verifier.main(argv) == 0
    assert not (directory / "error.json").exists()
    assert json.loads(Path(args.reward_json).read_text())["reward"] == 1


def test_judges_overlap_within_the_shared_parallelism_budget(tmp_path, monkeypatch):
    import threading

    args = _run_args(tmp_path)
    barrier = threading.Barrier(2)
    calls = []

    class Judge:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            calls.append(self.model)
            barrier.wait(timeout=2)
            return {"reasoning": "Checked.", "verdict": "pass"}

    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": name} for name in STANDARD_JUDGE_MODELS]))
    monkeypatch.setenv("LAB_JUDGE_PARALLELISM", "2")
    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    result = verifier.run_verifier(args)
    assert set(calls) == set(STANDARD_JUDGE_MODELS)
    assert all(value["judge_parallelism"] == 1 for value in result["per_judge"].values())
    assert json.loads(Path(args.reward_json).read_text())["reward"] == 1


@pytest.mark.parametrize(
    "model", ["nvidia/foo", "anthropic/claude-sonnet-4-6", "vllm/local", "trtllm/custom", "custom/path/model"]
)
def test_explicit_judge_model_identifier_is_preserved(model):
    from resources_servers.legal_agent_bench.lab_harbor.judge import OpenAICompatibleJudge

    assert OpenAICompatibleJudge._normalize_api_model(model) == model
    assert OpenAICompatibleJudge._normalize_api_model("openai-compatible/" + model) == model


@pytest.mark.parametrize("reasoning", ["", "high"])
def test_optional_reasoning_template_is_omitted_when_unset(tmp_path, monkeypatch, reasoning):
    args = _run_args(tmp_path)
    received = []

    class Judge:
        def __init__(self, **kwargs):
            received.append(kwargs["reasoning_effort"])
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"reasoning": "Checked.", "verdict": "pass"}

    monkeypatch.setenv("LAB_JUDGE_REASONING_EFFORT", reasoning)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": "gpt-5.5"}]))
    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    assert verifier.run_verifier(args)["score"] == 1
    assert received and set(received) == {reasoning or None}
