"""
Standalone test: run delivery-dataset gold outputs through the Gym structured_outputs verifier.

Usage:
    # Test the 30 example tasks bundled in this directory
    python examples/test_verify_tasks.py

    # Test a custom JSONL file
    python examples/test_verify_tasks.py --tasks path/to/tasks.jsonl

    # Limit number of tasks
    python examples/test_verify_tasks.py --max-tasks 10

Each JSONL row must have: responses_create_params, schema_str, schema_type, gold_output.
The script constructs a NeMoGymResponse wrapping gold_output as text, then calls verify().
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.structured_outputs.app import (
    SchemaType,
    StructuredOutputsResourcesServer,
    StructuredOutputsResourcesServerConfig,
    StructuredOutputsVerifyRequest,
)


def make_verify_request(row: dict) -> StructuredOutputsVerifyRequest:
    task_id = row.get("task_id", "unknown")
    gold_text = row.get("gold_output") or row["model_output"]
    response_obj = NeMoGymResponse(
        id=f"resp_{task_id}",
        created_at=0.0,
        model="gold",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{task_id}",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], text=gold_text, type="output_text"
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    params = row.get("responses_create_params", {"input": []})
    if isinstance(params, dict):
        params = NeMoGymResponseCreateParamsNonStreaming(**params)
    return StructuredOutputsVerifyRequest(
        responses_create_params=params,
        response=response_obj,
        schema_str=row["schema_str"],
        schema_type=SchemaType(row.get("schema_type", "json")),
    )


async def main():
    parser = argparse.ArgumentParser(description="Verify gold outputs against schemas")
    parser.add_argument(
        "--tasks",
        default=str(Path(__file__).parent / "example_tasks.jsonl"),
        help="Path to JSONL file with tasks",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks to test (0=all)")
    args = parser.parse_args()

    config = StructuredOutputsResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name=""
    )
    server = StructuredOutputsResourcesServer(
        config=config, server_client=MagicMock(spec=ServerClient)
    )

    rows = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.max_tasks > 0:
        rows = rows[: args.max_tasks]

    passes = 0
    fails = []
    for row in rows:
        tid = row.get("task_id", "?")
        schema_type = row.get("schema_type", "json")
        try:
            request = make_verify_request(row)
            result = await server.verify(request)
            if result.reward == 1.0:
                passes += 1
                print(f"  PASS  {tid} ({schema_type})")
            else:
                err = result.error_message or result.error_type or "unknown"
                fails.append((tid, err[:120]))
                print(f"  FAIL  {tid} ({schema_type}): {err[:120]}")
        except Exception as e:
            fails.append((tid, str(e)[:120]))
            print(f"  ERR   {tid} ({schema_type}): {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passes}/{passes + len(fails)} passed ({100*passes/(passes+len(fails)):.1f}%)")
    if fails:
        print(f"\nFailed tasks:")
        for tid, err in fails:
            print(f"  {tid}: {err}")

    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    asyncio.run(main())
