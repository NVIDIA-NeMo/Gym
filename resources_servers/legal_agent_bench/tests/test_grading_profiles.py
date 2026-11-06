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
def test_official_dual_averages_task_scores_and_retains_per_judge(tmp_path, monkeypatch, verdicts, expected):
    args = _run_args(tmp_path)
    answers = dict(zip(STANDARD_JUDGE_MODELS, verdicts, strict=True))

    class Judge:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"reasoning": "Evidence checked.", "verdict": answers[self.model]}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": model} for model in STANDARD_JUDGE_MODELS]))
    result = verifier.run_verifier(args)
    reward = json.loads(Path(args.reward_json).read_text())
    assert reward["reward"] == expected
    assert result["dual_all_pass_rate"] == expected
    assert result["judge_profile"] == "lab-standard-dual-v1"
    assert result["source_revision"] == "pinned-test-revision"
    assert len(result["per_judge"]) == 2
    assert all((Path(args.verifier_dir) / f"judge-{index}" / "scores.json").is_file() for index in (1, 2))
    assert (Path(args.verifier_dir) / "scores_dual.json").is_file()


def test_incomplete_dual_grading_keeps_evidence_without_successful_aggregate(tmp_path, monkeypatch):
    args = _run_args(tmp_path)

    class Judge:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            if self.model == "gpt-5.5":
                raise ConnectionError("judge unavailable")
            return {"reasoning": "Checked.", "verdict": "pass"}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    monkeypatch.setenv("LAB_JUDGES", json.dumps([{"model_name": model} for model in STANDARD_JUDGE_MODELS]))
    assert (
        verifier.main(
            [
                "--task-json",
                args.task_json,
                "--run-dir",
                args.run_dir,
                "--verifier-dir",
                args.verifier_dir,
                "--reward-json",
                args.reward_json,
                "--judge-base-url",
                args.judge_base_url,
            ]
        )
        == 1
    )
    assert not (Path(args.verifier_dir) / "scores_dual.json").exists()
    failure = json.loads(Path(args.reward_json).read_text())
    assert failure["verifier_error"] == 1
    failed_judge = json.loads((Path(args.verifier_dir) / "judge-2" / "scores.json").read_text())
    assert failed_judge["criteria_results"][0]["verdict"] == "error"
    assert (Path(args.verifier_dir) / "judge-1" / "scores.json").exists()


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


def test_explicit_empty_judge_key_does_not_inherit_shared_credentials(tmp_path, monkeypatch):
    args = _run_args(tmp_path)
    keys = []

    class Judge:
        def __init__(self, **kwargs):
            keys.append(kwargs["api_key"])
            self.base_url = kwargs["base_url"]

        def evaluate_prompt(self, *_args, **_kwargs):
            return {"reasoning": "Checked.", "verdict": "pass"}

    monkeypatch.setattr(verifier, "OpenAICompatibleJudge", Judge)
    monkeypatch.setenv("LAB_JUDGE_API_KEY", "shared-test-key")
    verifier._run_single_verifier(
        args, judge_override={"model_name": "local", "base_url": "http://local.test/v1", "api_key": ""}
    )
    assert keys and set(keys) == {""}


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
    from resources_servers.legal_agent_bench.app import LegalAgentBenchResourcesServerConfig

    args = _run_args(tmp_path)
    config = {"name": "lab", "host": "127.0.0.1", "port": 0, "entrypoint": "app.py"}
    if selection != "omitted":
        config["judges"] = selection
    with pytest.raises(ValueError):
        LegalAgentBenchResourcesServerConfig.model_validate(config)

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
def test_judge_list_roundtrips_resources_and_standalone_verifier(tmp_path, monkeypatch, dual):
    from resources_servers.legal_agent_bench.app import LegalAgentBenchResourcesServerConfig, _build_verifier_env

    selection = [{"model_name": "route/gpt-5.5", "base_url": "https://one.example/v1", "api_key": ""}]
    if dual:
        selection.append({"model_name": "route/claude-sonnet-4-6", "reasoning_effort": "high"})
    config = LegalAgentBenchResourcesServerConfig(
        name="lab",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
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
