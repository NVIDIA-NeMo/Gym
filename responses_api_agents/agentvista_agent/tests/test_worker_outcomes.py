# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise worker outcomes with a real local child process and no upstream/network dependencies."""

import hashlib
import json
import sys
from types import ModuleType

import pytest

from responses_api_agents.agentvista_agent.runtime import default_worker, scoring, source, worker
from responses_api_agents.agentvista_agent.runtime.data import inventory


@pytest.mark.parametrize(
    "ending,masked,reward",
    [
        ("answer", False, 1.0),
        ("empty", False, 0.0),
        ("length", False, 0.0),
        ("answer-timeout", False, 1.0),
        ("reasoning-timeout", False, 0.0),
        ("silent-timeout", True, 0.0),
        ("no-mcp", True, 0.0),
        ("native-error", True, 0.0),
    ],
)
def test_model_outcomes_are_judged_but_missing_tools_and_infrastructure_are_masked(
    tmp_path, monkeypatch, ending, masked, reward
):
    root = tmp_path / "source"
    (root / "prompts").mkdir(parents=True)
    (root / "prompts/inference_prompts.yaml").write_text("system_prompts:\n  multi_tool_agent_search: Fixture\n")
    monkeypatch.setenv("AGENTVISTA_ROOT", str(root))
    for key in (
        "VERIFIER_API_KEY",
        "VERIFIER_END_POINT",
        "VERIFIER_MODEL_NAME",
        "SERPAPI_KEY",
        "AGENTVISTA_VISIT_API_KEY",
        "AGENTVISTA_VISIT_API_ENDPOINT",
        "AGENTVISTA_VISIT_MODEL_NAME",
    ):
        monkeypatch.setenv(key, "fixture")
    checked = []
    monkeypatch.setattr(worker, "verify_source", lambda path: checked.append(path) or {"revision": "fixture"})
    judged = []

    def judge(question, answer, texts):
        judged.append(texts)
        value = float(any("<answer>42</answer>" in text for text in texts))
        return {"reward": value, "status": "PASS" if value else "FAIL", "prediction": texts[-1] if texts else ""}

    monkeypatch.setattr(worker, "score", judge)
    cli = tmp_path / "cli"
    cli.write_text(f"""#!{sys.executable}
import json,os,sys,time
from pathlib import Path
if sys.argv[1] == '--version':
 print('1.17.11');sys.exit(0)
cfg=json.loads(Path(os.environ['OPENCODE_CONFIG']).read_text())
assert os.environ['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX']=='32768'
assert cfg['agent']['build']['temperature']==0.6
assert cfg['agent']['build']['top_p']==0.8
assert cfg['provider']['nemo']['models']['fixture']['temperature'] is True
ending={ending!r}
mcp=json.loads(Path(cfg['mcp']['agentvista']['command'][-1]).read_text())
if ending!='no-mcp':
 target=Path(mcp['output'])/'tools_listed';target.parent.mkdir(parents=True);target.write_text('listed')
if ending in ('answer','answer-timeout','no-mcp'):
 print(json.dumps({{'type':'text','part':{{'text':'<answer>42</answer>'}}}}),flush=True)
elif ending in ('empty','length','reasoning-timeout'):
 print(json.dumps({{'type':'reasoning','part':{{'text':'Still working'}}}}),flush=True)
elif ending=='native-error':
 print(json.dumps({{'type':'error','error':{{'name':'APIError'}}}}),flush=True);sys.exit(1)
if ending.endswith('timeout'):time.sleep(5)
print(json.dumps({{'type':'step_finish','part':{{'reason':'length' if ending=='length' else 'stop'}}}}),flush=True)
""")
    cli.chmod(0o755)
    data = tmp_path / "tasks.json"
    data.write_text(json.dumps([{"question_id": "fixture", "problem": "Compute 42", "solution": "42", "images": []}]))
    task = inventory(data, tmp_path)["fixture"]
    value = worker.run(
        {
            "output": str(tmp_path / "episode"),
            "source_root": str(root),
            "dataset": str(data),
            "image_folder": str(tmp_path),
            "task_id": "fixture",
            "source_sha256": task["source_sha256"],
            "command": str(cli),
            "endpoint": "http://127.0.0.1:1/v1",
            "model": "fixture",
            "context_tokens": 65536,
            "output_tokens": 32768,
            "temperature": 0.6,
            "top_p": 0.8,
            "max_steps": 4,
            "agent_timeout": 0.5,
        }
    )
    assert checked == [root]
    assert value["mask_sample"] is masked, value
    assert value["reward"] == reward
    assert bool(judged) is not masked
    assert (tmp_path / "episode/result.json").is_file()


@pytest.mark.parametrize(
    "reply",
    [
        "<think>draft <answer>wrong</answer></think><answer>right</answer>",
        "<think>draft <answer>wrong</answer>",
        "<think>reasoning only</think>",
    ],
)
def test_default_reasoning_is_not_a_candidate_answer(reply):
    original = {"role": "assistant", "content": reply, "function_call": None}
    projected = default_worker._native_reply(original)
    expected = "<answer>right</answer>" if reply.endswith("<answer>right</answer>") else ""
    assert projected["content"] == expected
    assert "function_call" not in projected
    assert original["content"] == reply and "function_call" in original
    assert default_worker._native_reply(reply) == expected


@pytest.mark.parametrize("part,reward", [("0", 0), ("1", 1), ("1.", 1), ("**1**", 1), ("0.", 0), ("1,", 1), ("10", 1)])
def test_upstream_score_words_and_empty_prediction_are_preserved(monkeypatch, part, reward):
    prompts = []
    module = ModuleType("utils.general_qa_tool")
    module.QUERY_PROMPT = "{question}|{ground_truth}|{prediction}"
    module.SYSTEM_PROMPT = ""

    class Judge:
        def query(self, prompt, system_prompt):
            prompts.append(prompt)
            return part, "Explanation: fixture"

    module.GPT4VisionClient = Judge
    monkeypatch.setitem(sys.modules, "utils", ModuleType("utils"))
    monkeypatch.setitem(sys.modules, "utils.general_qa_tool", module)
    for key in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"):
        monkeypatch.setenv(key, "fixture")
    verdict = scoring.score("question", "answer", [])
    assert prompts == ["question|answer|"]
    assert verdict["prediction"] == "" and verdict["reward"] == reward
    assert verdict["status"] == ("PASS" if reward else "FAIL")


def test_every_pinned_source_file_is_checked_and_mismatch_stops_before_actor(tmp_path, monkeypatch):
    path = tmp_path / "infer.py"
    path.write_text("original")
    monkeypatch.setattr(
        source,
        "source_pin",
        lambda: {
            "repository": "https://example.com/source.git",
            "revision": "fixture",
            "files": {"infer.py": hashlib.sha256(b"original").hexdigest()},
        },
    )
    assert source.verify_source(tmp_path)["files_verified"] == 1
    path.write_text("modified")
    monkeypatch.setattr(
        worker.subprocess,
        "check_output",
        lambda *args, **kwargs: pytest.fail("Actor started before source validation"),
    )
    result = worker.run({"output": str(tmp_path / "episode"), "source_root": str(tmp_path)})
    assert result["mask_sample"] and result["failure_kind"] == "agentvista:setup"
    assert "differs from pinned revision: infer.py" in result["error"]
    path.unlink()
    with pytest.raises(ValueError, match="Missing pinned"):
        source.verify_source(tmp_path)


@pytest.mark.parametrize("key", ["VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"])
def test_default_missing_judge_fails_before_loading_policy(tmp_path, monkeypatch, key):
    for name in ("VERIFIER_API_KEY", "VERIFIER_END_POINT", "VERIFIER_MODEL_NAME"):
        monkeypatch.setenv(name, "fixture")
    monkeypatch.delenv(key)
    monkeypatch.setattr(
        default_worker, "verify_source", lambda *args: pytest.fail("Loaded source before checking judge")
    )
    result = default_worker.run({"output": str(tmp_path / "episode")})
    assert result["mask_sample"] and key in result["error"]
