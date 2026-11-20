# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real SDK shim's budget hooks against a deterministic SDK double."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SDK_DOUBLE = """
import {existsSync} from "node:fs";
import {join} from "node:path";
export const AuthStorage={inMemory:()=>({})};
export const SettingsManager={inMemory:()=>({})};
export const SessionManager={create:()=>({})};
export class DefaultResourceLoader {
  async reload(){}
  getSkills(){return {skills:[{name:'agent-message',kind:'python'}]}}
}
export function loadSkillsFromDir({dir}){
  return {skills:existsSync(join(dir,'SKILL.md')) ? [{name:'agent-message',kind:'python',python:{importName:'agent_message',packagePath:dir,pyprojectPath:join(dir,'pyproject.toml')}}] : []};
}
export function getPythonSkillRuntimeInfo(skills){return skills.map(s=>({name:s.name,...s.python}))}
export class ModelRegistry {find(){return {contextWindow:131072}}}
export async function createAgentSession(options){
  let listener,aborted=false;
  const emit=e=>listener(e);
  const session={sessionId:'child',rlmDepth:1,systemPrompt:'sdk-system',
    agent:{onPayload:async payload=>({...payload,originalHook:true})},
    getActiveToolNames:()=>['ipython'],getAllTools:()=>[{name:'ipython'}],
    getContextUsage:()=>({tokens:100000}),
    subscribe:f=>{listener=f},abort:async()=>{aborted=true},disposeAsync:async()=>{},
    promptAndWait:async()=>{
      for(let turn=1;turn<=3;turn++){
        await session.agent.onPayload({max_completion_tokens:99}, {contextWindow:131072});
        const message={role:'assistant',content:[{type:'toolCall',name:'ipython'}],stopReason:'toolUse'};
        emit({type:'message_end',message});
        emit({type:'tool_execution_start',toolName:'ipython',toolCallId:String(turn)});
        if(turn===2) await options.agentMessageController.sendAgentMessage({target:'parent',message:'report on last allowed turn'});
        emit({type:'tool_execution_end',toolName:'ipython',toolCallId:String(turn),result:{details:{status:'ok'}}});
        emit({type:'turn_end',message});
        if(aborted) break;
      }
    }
  };
  return {session};
}
"""


def write_sdk_runtime(runtime: Path, source: str) -> Path:
    """Mirror the shared npm global prefix rather than the removed custom installer."""
    root = runtime / "lib/node_modules/prime-agent"
    for directory, code in (
        (root, source),
        (root / "node_modules/@earendil-works/pi-ai", "export function cleanupSessionResources(){}"),
    ):
        (directory / "dist").mkdir(parents=True)
        (directory / "package.json").write_text('{"type":"module","version":"0.7.0"}')
        (directory / "dist/index.js").write_text(code)
    return root


def write_sdk_skill(root: Path) -> Path:
    skill = root / "dist/skills/agent-message"
    (skill / "src/agent_message").mkdir(parents=True)
    (skill / "SKILL.md").write_text("Bundled agent-message fixture")
    (skill / "pyproject.toml").write_text('[project]\nname="agent-message"\nversion="0.1.0"\n')
    (skill / "src/agent_message/__init__.py").write_text(
        "async def send(*args, **kwargs): pass\nasync def list_agents(): pass\n"
    )
    (skill / "src/rlm.py").write_text("async def host_request(*args, **kwargs): pass\n")
    return skill


def sdk_error_capture(tmp_path, *, delivered: bool, dispose_error: bool = False, provider_error: str | None = None):
    """Have the actual checked-in shim catch an SDK error and write its real capture."""
    send = "if(turn===2) await options.agentMessageController.sendAgentMessage({target:'parent',message:'report on last allowed turn'});"
    failure_event = (
        "emit("
        + json.dumps(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [], "stopReason": "error", "errorMessage": provider_error},
            }
        )
        + ");"
        if provider_error is not None
        else ""
    )
    replacement = """
        if(turn===2){
          const receipt=await options.agentMessageController.sendAgentMessage({target:'parent',message:'Task-specific findings'});
          emit({type:'tool_execution_end',toolName:'ipython',toolCallId:String(turn),result:{details:{status:'ok',sentAgentMessages:[{id:receipt.id,receiverRole:'parent',deliveryStatus:'delivered'}]}}});
          THROW_AFTER_SEND
        }
    """.replace(
        "THROW_AFTER_SEND", failure_event + ("" if dispose_error else "throw new Error('SDK failed after delivery');")
    )
    if not delivered:
        replacement = "if(turn===2){" + failure_event + "throw new Error('SDK failed without delivery');}"
    source = SDK_DOUBLE.replace(send, replacement)
    if dispose_error:
        source = source.replace(
            "disposeAsync:async()=>{}", "disposeAsync:async()=>{throw new Error('SDK disposal failed')}"
        )
    runtime = tmp_path / "runtime"
    write_sdk_runtime(runtime, source)
    args = {
        "runtime": str(runtime),
        "state": str(tmp_path / "state"),
        "cwd": str(tmp_path),
        "models": {},
        "model": "test",
        "prompt": "Send findings",
        "maxTurns": 2,
        "maxTokens": 65536,
        "temperature": 0,
        "captureDirectory": str(tmp_path / "export"),
    }
    request = tmp_path / "request.json"
    request.write_text(json.dumps(args))
    result = subprocess.run(
        ["node", str(Path(__file__).parents[1] / "child_runtime.mjs"), str(request)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((tmp_path / "export/capture.json").read_text())


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for the SDK shim unit test")
def test_last_allowed_turn_executes_its_tools_before_stopping(tmp_path):
    runtime = tmp_path / "runtime"
    write_sdk_runtime(runtime, SDK_DOUBLE)
    state = tmp_path / "state"
    capture = tmp_path / "export"
    args = {
        "runtime": str(runtime),
        "state": str(state),
        "cwd": str(tmp_path),
        "models": {},
        "model": "test",
        "prompt": "Report to parent",
        "maxTurns": 2,
        "maxTokens": 65536,
        "temperature": 0,
        "captureDirectory": str(capture),
    }
    request = tmp_path / "request.json"
    request.write_text(json.dumps(args))
    shim = Path(__file__).parents[1] / "child_runtime.mjs"
    result = subprocess.run(["node", str(shim), str(request)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    captured = json.loads((capture / "capture.json").read_text())
    assert captured["termination"] == "turn_limit"
    assert captured["assistantTurns"] == 2
    assert [item["message"] for item in captured["deliveries"]] == ["report on last allowed turn"]
    requests = [json.loads(line) for line in (state / "model-requests.jsonl").read_text().splitlines()]
    assert len(requests) == 2  # No third model call.
    assert all(request["max_tokens"] == 131072 - 100000 - 8192 for request in requests)
    assert all(request["originalHook"] for request in requests)
    assert all("max_completion_tokens" not in request for request in requests)
    assert all(request["tool_choice"]["function"]["name"] == "ipython" for request in requests)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is required for the SDK capability probe")
@pytest.mark.parametrize("fault", [None, "sdk_api", "sdk_version", "ai_path", "skill", "python_import", "python_api"])
def test_runtime_probe_checks_imports_skill_and_kernel_without_model_calls(tmp_path, fault):
    runtime = tmp_path / "runtime"
    source = SDK_DOUBLE.replace(
        "export async function createAgentSession(options){",
        "export async function createAgentSession(options){ throw new Error('Probe must not create sessions');",
    )
    if fault == "sdk_api":
        source = source.replace("export async function createAgentSession", "async function createAgentSession")
    root = write_sdk_runtime(runtime, source)
    skill = write_sdk_skill(root)
    module = skill / "src/agent_message/__init__.py"
    if fault == "sdk_version":
        (root / "package.json").write_text('{"type":"module","version":"0.8.0"}')
    elif fault == "ai_path":
        (root / "node_modules/@earendil-works/pi-ai/dist/index.js").unlink()
    elif fault == "skill":
        (skill / "SKILL.md").unlink()
    elif fault == "python_import":
        module.unlink()
    elif fault == "python_api":
        module.write_text("send = None\nlist_agents = None\n")
    result = subprocess.run(
        ["node", str(Path(__file__).parents[1] / "child_runtime.mjs"), "--check-runtime", str(runtime)],
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PRIME_AGENT_KERNEL_PYTHON": sys.executable, "PYTHONPATH": str(skill / "src")},
    )
    if fault:
        assert result.returncode != 0
        assert "Probe must not create sessions" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "sdkVersion": "0.7.0",
            "aiVersion": "0.7.0",
            "skill": "agent-message",
            "pythonImport": "agent_message",
        }
    assert not (tmp_path / "state").exists()
