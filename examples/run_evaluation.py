"""
End-to-end evaluation: send task prompts to a model, verify responses through
the structured_outputs resource server pipeline.

Usage:
    # Run on example tasks with OpenAI-compatible API
    python examples/run_evaluation.py \
        --gym-path /path/to/Gym \
        --api-base https://api.openai.com/v1 \
        --api-key sk-... \
        --model gpt-4o \
        --tasks examples/example_tasks.jsonl

    # Run on a tasks directory (each subfolder has task.json + gold_output.*)
    python examples/run_evaluation.py \
        --gym-path /path/to/Gym \
        --api-base https://api.openai.com/v1 \
        --api-key sk-... \
        --model gpt-4o \
        --tasks /path/to/tasks/ \
        --max-tasks 10

    # Use a separate judge model for semantic evaluation
    python examples/run_evaluation.py \
        --gym-path /path/to/Gym \
        --api-base https://api.openai.com/v1 \
        --api-key sk-... \
        --model gpt-5.4 \
        --judge-model gpt-4o \
        --tasks examples/example_tasks.jsonl

    # Verify gold outputs only (no model generation)
    python examples/run_evaluation.py \
        --gym-path /path/to/Gym \
        --tasks examples/example_tasks.jsonl \
        --gold-only

Requires: pip install openai orjson
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import openai
import orjson


def setup_gym_path(gym_path: str):
    sys.path.insert(0, gym_path)
    from nemo_gym.config_types import ModelServerRef
    from nemo_gym.openai_utils import (
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseOutputMessage,
        NeMoGymResponseOutputText,
    )
    from resources_servers.structured_outputs.app import (
        SchemaType,
        StructuredOutputsResourcesServer,
        StructuredOutputsResourcesServerConfig,
        StructuredOutputsVerifyRequest,
    )
    return {
        "ModelServerRef": ModelServerRef,
        "NeMoGymResponse": NeMoGymResponse,
        "NeMoGymResponseCreateParamsNonStreaming": NeMoGymResponseCreateParamsNonStreaming,
        "NeMoGymResponseOutputMessage": NeMoGymResponseOutputMessage,
        "NeMoGymResponseOutputText": NeMoGymResponseOutputText,
        "SchemaType": SchemaType,
        "StructuredOutputsResourcesServer": StructuredOutputsResourcesServer,
        "StructuredOutputsResourcesServerConfig": StructuredOutputsResourcesServerConfig,
        "StructuredOutputsVerifyRequest": StructuredOutputsVerifyRequest,
    }


def strip_code_fences(text: str) -> str:
    m = re.match(r"^```\w*\n(.*?)```\s*$", text, re.DOTALL)
    return m.group(1).strip() if m else text


def load_tasks_from_jsonl(path: Path, max_tasks: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:max_tasks] if max_tasks > 0 else rows


def load_tasks_from_dir(path: Path, max_tasks: int) -> list[dict]:
    rows = []
    for d in sorted(path.iterdir()):
        tp = d / "task.json"
        if not tp.exists():
            continue
        data = json.loads(tp.read_text())
        gold_files = [f for f in d.iterdir() if f.stem == "gold_output"]
        gold_text = gold_files[0].read_text() if gold_files else None
        rows.append({
            "task_id": d.name,
            "schema_str": data["schema_str"],
            "schema_type": data.get("schema_type", "json"),
            "responses_create_params": data.get("responses_create_params", {"input": []}),
            "semantic_verifier_config": data.get("semantic_verifier_config"),
            "gold_output": gold_text,
        })
        if max_tasks > 0 and len(rows) >= max_tasks:
            break
    return rows


async def main():
    parser = argparse.ArgumentParser(description="Run model evaluation through structured_outputs verify()")
    parser.add_argument("--gym-path", required=True, help="Path to NeMo-Gym repo root")
    parser.add_argument("--tasks", required=True, help="Path to JSONL file or tasks directory")
    parser.add_argument("--api-base", default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--judge-model", default=None, help="Judge model for semantic eval (default: same as --model)")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks to evaluate (0=all)")
    parser.add_argument("--gold-only", action="store_true", help="Verify gold outputs only (no model API calls)")
    parser.add_argument("--reward-mode", default="combined", choices=["combined", "independent"],
                        help="'combined': reward = syntax * semantic. 'independent': reward = syntax only")
    parser.add_argument("--strip-fences", action="store_true", default=True, help="Strip markdown code fences from model output")
    parser.add_argument("--output", default=None, help="Save results to JSONL file")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not args.gold_only and not api_key:
        print("Error: --api-key or OPENAI_API_KEY required (unless --gold-only)")
        sys.exit(1)

    gym = setup_gym_path(args.gym_path)
    judge_model = args.judge_model or args.model

    oai_client = openai.AsyncOpenAI(api_key=api_key, base_url=args.api_base) if api_key and not args.gold_only else None

    # --- Build judge mock for semantic evaluation ---
    async def judge_call(server_name, url_path, json_params=None, **kwargs):
        msgs = []
        input_msgs = json_params.input if hasattr(json_params, "input") else json_params.get("input", [])
        for msg in input_msgs:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "user")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            msgs.append({"role": role, "content": content})

        resp = await oai_client.chat.completions.create(
            model=judge_model, messages=msgs, max_tokens=512, temperature=0.0,
        )
        judge_text = resp.choices[0].message.content.strip()

        gym_resp = gym["NeMoGymResponse"].model_construct(
            id=resp.id, created_at=int(time.time()), model=judge_model, object="response",
            output=[gym["NeMoGymResponseOutputMessage"].model_construct(
                id="msg-judge",
                content=[gym["NeMoGymResponseOutputText"].model_construct(
                    text=judge_text, type="output_text", annotations=[],
                )],
                role="assistant", status="completed", type="message",
            )],
            parallel_tool_calls=False, tool_choice="auto", tools=[], error=None,
        )

        class MockResponse:
            def __init__(self, data):
                self._data = orjson.dumps(data)
            async def read(self):
                return self._data

        return MockResponse(gym_resp.model_dump())

    # --- Build server ---
    has_judge = oai_client is not None
    if has_judge:
        config = gym["StructuredOutputsResourcesServerConfig"](
            host="0.0.0.0", port=8080, entrypoint="", name="",
            reward_mode=args.reward_mode,
            judge_model_server=gym["ModelServerRef"](type="responses_api_models", name="judge"),
            judge_responses_create_params=gym["NeMoGymResponseCreateParamsNonStreaming"](
                input=[], temperature=0.0, max_output_tokens=512,
            ),
        )
        server = gym["StructuredOutputsResourcesServer"].model_construct(
            config=config, server_client=MagicMock(),
        )
        server.server_client.post = judge_call
    else:
        config = gym["StructuredOutputsResourcesServerConfig"](
            host="0.0.0.0", port=8080, entrypoint="", name="",
            reward_mode=args.reward_mode,
        )
        from nemo_gym.server_utils import ServerClient
        server = gym["StructuredOutputsResourcesServer"](
            config=config, server_client=MagicMock(spec=ServerClient),
        )

    # --- Load tasks ---
    tasks_path = Path(args.tasks)
    if tasks_path.is_file():
        tasks = load_tasks_from_jsonl(tasks_path, args.max_tasks)
    else:
        tasks = load_tasks_from_dir(tasks_path, args.max_tasks)

    print(f"{'='*70}")
    print(f"Structured Outputs Evaluation")
    print(f"  Model:       {args.model}")
    print(f"  Judge:       {judge_model if has_judge else 'none'}")
    print(f"  Tasks:       {len(tasks)}")
    print(f"  Gold only:   {args.gold_only}")
    print(f"  Reward mode: {config.reward_mode}")
    print(f"{'='*70}\n")

    results = []
    for task in tasks:
        tid = task.get("task_id", "?")
        st = task.get("schema_type", "json")
        rcp = task.get("responses_create_params", {"input": []})

        # Get model output
        if args.gold_only:
            model_output = task.get("gold_output") or task.get("model_output", "")
        else:
            try:
                oai_msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")}
                            for m in rcp.get("input", [])]
                try:
                    resp = await oai_client.chat.completions.create(
                        model=args.model, messages=oai_msgs,
                        max_completion_tokens=4096, temperature=0.0,
                    )
                except openai.BadRequestError:
                    resp = await oai_client.chat.completions.create(
                        model=args.model, messages=oai_msgs,
                        max_tokens=4096, temperature=0.0,
                    )
                model_output = resp.choices[0].message.content.strip()
                if args.strip_fences:
                    model_output = strip_code_fences(model_output)
            except Exception as e:
                print(f"  ERR   {tid} ({st}): API error - {e}")
                results.append({"task_id": tid, "error": str(e)})
                continue

        # Build verify request
        response_obj = gym["NeMoGymResponse"](
            id=f"resp_{tid}", created_at=0.0, model=args.model, object="response",
            output=[gym["NeMoGymResponseOutputMessage"](
                id=f"msg_{tid}",
                content=[gym["NeMoGymResponseOutputText"](
                    annotations=[], text=model_output, type="output_text",
                )],
                role="assistant", status="completed", type="message",
            )],
            parallel_tool_calls=False, tool_choice="none", tools=[],
        )
        request = gym["StructuredOutputsVerifyRequest"](
            responses_create_params=gym["NeMoGymResponseCreateParamsNonStreaming"](**rcp),
            response=response_obj,
            schema_str=task["schema_str"],
            schema_type=gym["SchemaType"](st),
            semantic_verifier_config=task.get("semantic_verifier_config"),
        )

        result = await server.verify(request)

        row = {
            "task_id": tid,
            "schema_type": st,
            "model": args.model,
            "model_output": model_output,
            "reward": result.reward,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "semantic_reward": result.semantic_reward,
            "semantic_results": [r.model_dump() for r in result.semantic_results]
            if result.semantic_results else None,
        }
        results.append(row)

        syntax = "FAIL" if result.error_type else "PASS"
        sem = f"semantic={result.semantic_reward:.2f}" if result.semantic_reward is not None else "no semantic"
        err = f" [{result.error_type}]" if result.error_type else ""
        print(f"  {syntax:<4}  {tid} ({st}): reward={result.reward:.2f}{err} | {sem}")

    # --- Summary ---
    valid = [r for r in results if "error" not in r]
    syntax_pass = sum(1 for r in valid if r["error_type"] is None)
    syntax_fail = len(valid) - syntax_pass

    print(f"\n{'='*70}")
    print(f"Results: {len(valid)} tasks evaluated")
    print(f"  Syntax:   {syntax_pass} pass, {syntax_fail} fail")

    sem_tasks = [r for r in valid if r.get("semantic_reward") is not None]
    if sem_tasks:
        avg_sem = sum(r["semantic_reward"] for r in sem_tasks) / len(sem_tasks)
        print(f"  Semantic: {len(sem_tasks)} tasks, avg={avg_sem:.3f}")

    combined = [r for r in valid if r["reward"] > 0]
    if valid:
        avg_reward = sum(r["reward"] for r in valid) / len(valid)
        print(f"  Reward:   avg={avg_reward:.3f} ({len(combined)}/{len(valid)} > 0)")

    if args.output:
        out = Path(args.output)
        with out.open("w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to {out}")

    sys.exit(0 if syntax_fail == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
