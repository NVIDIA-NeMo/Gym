# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest

from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_agents.hermes_agent.runtime import host_python
from responses_api_agents.hermes_agent.tests.test_sandbox_runner import _completion, _ModelServer


@pytest.mark.parametrize("delegated", [True, False])
@pytest.mark.parametrize("retry", [True, False])
def test_iteration_summary_retains_invocation_on_all_attempts(tmp_path, delegated, retry):
    """Exercise the pinned root/child summary API and validate its wire traffic with Gym."""
    responses = [_completion({"content": ""})] if retry else []
    responses.append(_completion({"content": "Partial findings."}))
    for i, response in enumerate(responses):
        response["id"] = f"summary-{i}"
    probe = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from run_agent import AIAgent
from tools.delegate_tool import _build_child_agent
from model_kwargs import configure_model_calls
from sandbox_observer import SandboxHermesObserver
parent = AIAgent(base_url=sys.argv[2], api_key='gym', model='policy_model', provider='custom',
    api_mode='chat_completions', quiet_mode=True, enabled_toolsets=[], skip_memory=True,
    skip_context_files=True, skip_background_review=True, max_iterations=1)
configure_model_calls(parent, preserve_reasoning_history=False)
def no_tool_calls(*_):
    raise AssertionError('this probe makes no tool call')
observer = SandboxHermesObserver(tool_recorder=no_tool_calls).instrument(parent)
agent = parent
if sys.argv[3] == 'True':
    agent = _build_child_agent(task_index=0, task_count=1, goal='Inspect the work', context=None,
        toolsets=[], model=None, max_iterations=1, parent_agent=parent)
messages = [
    {'role':'assistant', 'content':None, 'tool_calls':[{'type':'function','id':'t','call_id':'t',
     'response_item_id':'fc_1','function':{'name':'terminal','arguments':'{}'}}]},
    {'role':'tool','tool_call_id':'t','content':'partial work'},
]
agent._cached_system_prompt = 'Inspect the work.'
summary = agent._handle_max_iterations(messages, 1)
rows = observer.finish({'messages': [], 'completed': True}, None)['invocations']
pathlib.Path(sys.argv[4]).write_text(json.dumps([summary, rows, messages[0]]))
"""
    output = tmp_path / "result.json"
    with _ModelServer(responses) as model:
        completed = subprocess.run(
            [
                str(host_python()),
                "-I",
                "-c",
                probe,
                str(Path(__file__).parents[1]),
                model.base_url,
                str(delegated),
                str(output),
            ],
            env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    summary, rows, original = json.loads(output.read_text())
    assert summary == "Partial findings."
    assert original["tool_calls"][0]["response_item_id"] == "fc_1"
    assert len(model.requests) == len(responses)
    for body in model.requests:
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(body)
        assert not body.get("stream")
    owner = "root.child-1" if delegated else "root"
    assert [h.get("x-session-id") for h in model.request_headers] == [owner] * len(responses)
    by_id = {row["invocation_id"]: row for row in rows}
    assert by_id[owner]["model_response_ids"] == [response["id"] for response in responses]
    assert by_id[owner]["stop_reason"] == "max_iterations"
    assert by_id[owner]["status"] == "incomplete"
    if delegated:
        assert by_id["root"]["status"] == "completed"
        assert by_id["root"]["model_response_ids"] == []


def test_token_capture_survives_transport_rebuild(tmp_path):
    """Hermes clears its transport cache when it rebuilds the client after transient errors; capture must persist."""
    probe = """
import json, pathlib, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from openai.types.chat import ChatCompletion
from run_agent import AIAgent
from model_kwargs import configure_model_calls
agent = AIAgent(base_url='http://127.0.0.1:9/v1', api_key='gym', model='policy_model', provider='custom',
    api_mode='chat_completions', quiet_mode=True, enabled_toolsets=[], skip_memory=True,
    skip_context_files=True, skip_background_review=True, max_iterations=1)
configure_model_calls(agent, preserve_reasoning_history=False)
agent._transport_cache.clear()
response = ChatCompletion.model_validate(json.loads(sys.argv[2]))
normalized = agent._get_transport().normalize_response(response)
pathlib.Path(sys.argv[3]).write_text(json.dumps(agent._build_assistant_message(normalized, 'stop')))
"""
    completion = _completion(
        {"content": "done", "prompt_token_ids": [3, 4], "generation_token_ids": [5], "generation_log_probs": [-0.1]}
    )
    output = tmp_path / "message.json"
    completed = subprocess.run(
        [str(host_python()), "-I", "-c", probe, str(Path(__file__).parents[1]), json.dumps(completion), str(output)],
        env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    message = json.loads(output.read_text())
    assert message["prompt_token_ids"] == [3, 4]
    assert message["generation_token_ids"] == [5]
    assert message["generation_log_probs"] == [-0.1]


def test_summary_retry_attaches_only_the_final_attempt_tokens(tmp_path):
    """An empty first summary attempt is retried; the appended summary turn owns only the retry's token IDs."""
    responses = [
        _completion({"content": "", "prompt_token_ids": [1], "generation_token_ids": [2]}),
        _completion({"content": "Partial findings.", "prompt_token_ids": [7, 8], "generation_token_ids": [9]}),
    ]
    probe = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from run_agent import AIAgent
from model_kwargs import configure_model_calls
agent = AIAgent(base_url=sys.argv[2], api_key='gym', model='policy_model', provider='custom',
    api_mode='chat_completions', quiet_mode=True, enabled_toolsets=[], skip_memory=True,
    skip_context_files=True, skip_background_review=True, max_iterations=1)
configure_model_calls(agent, preserve_reasoning_history=False)
agent._cached_system_prompt = 'Inspect the work.'
messages = [{'role': 'user', 'content': 'Inspect the work.'}]
summary = agent._handle_max_iterations(messages, 1)
pathlib.Path(sys.argv[3]).write_text(json.dumps([summary, messages[-1]]))
"""
    output = tmp_path / "result.json"
    with _ModelServer(responses) as model:
        completed = subprocess.run(
            [str(host_python()), "-I", "-c", probe, str(Path(__file__).parents[1]), model.base_url, str(output)],
            env={**os.environ, "HERMES_HOME": str(tmp_path / "home")},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr
    summary, last = json.loads(output.read_text())
    assert summary == "Partial findings."
    assert len(model.requests) == 2
    assert last["role"] == "assistant"
    assert last["prompt_token_ids"] == [7, 8]
    assert last["generation_token_ids"] == [9]
