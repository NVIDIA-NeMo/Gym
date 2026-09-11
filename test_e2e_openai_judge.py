"""
End-to-end test of structured_outputs/app.py with a real OpenAI judge endpoint.

Calls the actual verify() method from the PR. The mock replaces only the
HTTP transport (server_client.post -> OpenAI /v1/chat/completions) since
we can't run the full Gym model server infra locally. Everything else --
config parsing, request construction, prompt templating, response parsing,
verdict extraction, score aggregation -- runs through the real PR code.

Usage:
    OPENAI_API_KEY=sk-... python test_e2e_openai_judge.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import openai
import orjson

sys.path.insert(0, str(Path(__file__).parent))

from resources_servers.structured_outputs.app import (
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)

TASKS_DIR = Path(os.environ.get("TASKS_DIR", "/Users/saumyachauhan/Projects/structured-outputs/tasks"))
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
MAX_TASKS = int(os.environ.get("MAX_TASKS", "30"))

oai_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class MockHTTPResponse:
    def __init__(self, data: dict):
        self._data = orjson.dumps(data)

    async def read(self):
        return self._data


async def openai_judge_call(server_name: str, url_path: str, json=None, **kwargs) -> MockHTTPResponse:
    json_params = json
    """Route server_client.post() to the real OpenAI API."""
    messages = []
    input_msgs = json_params.input if hasattr(json_params, "input") else json_params.get("input", [])
    for msg in input_msgs:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        messages.append({"role": role, "content": content})

    response = await oai_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )
    judge_text = response.choices[0].message.content.strip()

    gym_response = NeMoGymResponse.model_construct(
        id=response.id,
        created_at=int(time.time()),
        model=JUDGE_MODEL,
        object="response",
        output=[
            NeMoGymResponseOutputMessage.model_construct(
                id="msg-1",
                content=[NeMoGymResponseOutputText.model_construct(
                    text=judge_text, type="output_text", annotations=[]
                )],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        error=None,
    )
    return MockHTTPResponse(gym_response.model_dump())


def build_mock_response(gold_text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_construct(
        id="gold-response",
        created_at=0,
        model="gold",
        object="response",
        output=[
            NeMoGymResponseOutputMessage.model_construct(
                id="msg-1",
                content=[NeMoGymResponseOutputText.model_construct(
                    text=gold_text, type="output_text", annotations=[]
                )],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        error=None,
    )


def find_llmaaj_tasks(tasks_dir: Path, max_tasks: int) -> list[Path]:
    result = []
    for d in sorted(tasks_dir.iterdir()):
        tp = d / "task.json"
        if not tp.exists():
            continue
        t = json.loads(tp.read_text())
        svc = t.get("semantic_verifier_config")
        if not svc:
            vcp = d / "verifier_config.json"
            if vcp.exists():
                svc = json.loads(vcp.read_text())
        if not svc:
            continue
        if any(c.get("type") == "llmaaj" for c in svc.get("criteria", [])):
            result.append(d)
        if len(result) >= max_tasks:
            break
    return result


async def test_task(server, task_dir: Path) -> dict:
    task_data = json.loads((task_dir / "task.json").read_text())

    svc = task_data.get("semantic_verifier_config")
    if not svc:
        vcp = task_dir / "verifier_config.json"
        if vcp.exists():
            svc = json.loads(vcp.read_text())
            task_data["semantic_verifier_config"] = svc

    gold_files = [f for f in task_dir.iterdir() if f.stem == "gold_output"]
    if not gold_files:
        return {"task": task_dir.name, "error": "no gold_output"}
    gold_text = gold_files[0].read_text()

    rcp = task_data.get("responses_create_params", {})
    input_msgs = [NeMoGymEasyInputMessage(role=m["role"], content=m["content"]) for m in rcp.get("input", [])]
    params = NeMoGymResponseCreateParamsNonStreaming(input=input_msgs)

    request = StructuredOutputsVerifyRequest(
        responses_create_params=params,
        response=build_mock_response(gold_text),
        schema_str=task_data["schema_str"],
        schema_type=task_data["schema_type"],
        semantic_verifier_config=task_data.get("semantic_verifier_config"),
    )

    response = await server.verify(request)

    results = response.semantic_results or []
    return {
        "task": task_dir.name,
        "schema_type": task_data["schema_type"],
        "syntax_reward": response.reward,
        "syntax_error": response.error_type,
        "semantic_reward": response.semantic_reward,
        "llmaaj_passed": sum(1 for r in results if r.passed),
        "llmaaj_total": len(results),
        "criteria_detail": [{"name": r.name, "passed": r.passed, "weight": r.weight} for r in results],
    }


async def main():
    config = StructuredOutputsResourcesServerConfig(
        entrypoint="app.py",
        host="localhost",
        port=8000,
        name="structured_outputs_test",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[], temperature=0.0, max_output_tokens=512,
        ),
    )
    server = StructuredOutputsResourcesServer.model_construct(config=config, server_client=MagicMock())
    server.server_client.post = openai_judge_call

    task_dirs = find_llmaaj_tasks(TASKS_DIR, MAX_TASKS)
    print(f"End-to-end test: {len(task_dirs)} tasks, judge={JUDGE_MODEL} (real OpenAI API)")
    print(f"Code under test: resources_servers/structured_outputs/app.py verify()")
    print(f"Mock: server_client.post() -> OpenAI /v1/chat/completions (transport only)")
    print(f"{'=' * 80}")

    results = []
    for td in task_dirs:
        r = await test_task(server, td)
        results.append(r)
        if "error" in r:
            print(f"  {r['task']}: ERROR - {r['error']}")
            continue
        sem = f"semantic={r['semantic_reward']:.2f} ({r['llmaaj_passed']}/{r['llmaaj_total']})" if r["semantic_reward"] is not None else "no semantic"
        err = f" [{r['syntax_error']}]" if r["syntax_error"] else ""
        print(f"  {r['task']}: syntax={r['syntax_reward']:.0f}{err} | {sem}")

    print(f"\n{'=' * 80}")
    valid = [r for r in results if "error" not in r]
    syntax_pass = sum(1 for r in valid if r["syntax_reward"] == 1.0)
    print(f"Syntax: {syntax_pass}/{len(valid)} pass ({syntax_pass / len(valid) * 100:.1f}%)")

    semantic_tasks = [r for r in valid if r["semantic_reward"] is not None]
    if semantic_tasks:
        sem_pass = sum(1 for r in semantic_tasks if r["semantic_reward"] >= 0.8)
        avg_sem = sum(r["semantic_reward"] for r in semantic_tasks) / len(semantic_tasks)
        total_criteria = sum(r["llmaaj_total"] for r in semantic_tasks)
        total_passed = sum(r["llmaaj_passed"] for r in semantic_tasks)
        print(f"Semantic: {sem_pass}/{len(semantic_tasks)} pass >=0.8 ({sem_pass / len(semantic_tasks) * 100:.1f}%), avg={avg_sem:.3f}")
        print(f"LLMaaJ criteria: {total_passed}/{total_criteria} pass ({total_passed / total_criteria * 100:.1f}%)")

        failures = [r for r in semantic_tasks if r["semantic_reward"] < 0.8]
        if failures:
            print(f"\nSemantic failures (<0.8):")
            for r in failures:
                print(f"  {r['task']}: {r['semantic_reward']:.2f}")
                for c in r["criteria_detail"]:
                    status = "PASS" if c["passed"] else "FAIL"
                    print(f"    {status} [{c['weight']}] {c['name']}")

    print(f"\n{'=' * 80}")
    print("PR code exercised:")
    print("  [x] StructuredOutputsResourcesServerConfig (judge fields)")
    print("  [x] StructuredOutputsVerifyRequest.semantic_verifier_config")
    print("  [x] StructuredOutputsVerifyResponse.semantic_reward/results")
    print("  [x] verify() -> _evaluate_semantic() -> _evaluate_criterion()")
    print("  [x] Judge prompt templating from per-task rubrics")
    print("  [x] NeMoGymResponse parsing + _extract_judge_text()")
    print("  [x] [[PASS]]/[[FAIL]] verdict extraction via _extract_verdict()")
    print("  [x] _aggregate_semantic() weighted scoring")
    print("  [x] SemanticCriterionResult model")
    print(f"  Mock: server_client.post() -> OpenAI {JUDGE_MODEL} (transport layer only)")


if __name__ == "__main__":
    asyncio.run(main())
