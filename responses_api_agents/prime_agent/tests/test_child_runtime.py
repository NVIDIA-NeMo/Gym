# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real SDK shim's budget hooks against a deterministic SDK double."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


SDK_DOUBLE = """
export const AuthStorage={inMemory:()=>({})};
export const SettingsManager={inMemory:()=>({})};
export const SessionManager={create:()=>({})};
export class DefaultResourceLoader {async reload(){}}
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


def sdk_error_capture(tmp_path, *, delivered: bool, dispose_error: bool = False):
    """Have the actual checked-in shim catch an SDK error and write its real capture."""
    send = "if(turn===2) await options.agentMessageController.sendAgentMessage({target:'parent',message:'report on last allowed turn'});"
    replacement = """
        if(turn===2){
          const receipt=await options.agentMessageController.sendAgentMessage({target:'parent',message:'Task-specific findings'});
          emit({type:'tool_execution_end',toolName:'ipython',toolCallId:String(turn),result:{details:{status:'ok',sentAgentMessages:[{id:receipt.id,receiverRole:'parent',deliveryStatus:'delivered'}]}}});
          THROW_AFTER_SEND
        }
    """.replace("THROW_AFTER_SEND", "" if dispose_error else "throw new Error('SDK failed after delivery');")
    if not delivered:
        replacement = "if(turn===2) throw new Error('SDK failed without delivery');"
    source = SDK_DOUBLE.replace(send, replacement)
    if dispose_error:
        source = source.replace(
            "disposeAsync:async()=>{}", "disposeAsync:async()=>{throw new Error('SDK disposal failed')}"
        )
    runtime = tmp_path / "runtime"
    for package, code in (
        ("prime-agent", source),
        ("@earendil-works/pi-ai", "export function cleanupSessionResources(){}"),
    ):
        directory = runtime / "node_modules" / package
        (directory / "dist").mkdir(parents=True)
        (directory / "package.json").write_text('{"type":"module"}')
        (directory / "dist/index.js").write_text(code)
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
    for package, source in (
        ("prime-agent", SDK_DOUBLE),
        ("@earendil-works/pi-ai", "export function cleanupSessionResources(){}"),
    ):
        root = runtime / "node_modules" / package
        (root / "dist").mkdir(parents=True)
        (root / "package.json").write_text('{"type":"module"}')
        (root / "dist/index.js").write_text(source)
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
