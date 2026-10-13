# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the DeepSWE oracle patch through every fresh verifier sandbox."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


def _empty_response() -> dict[str, Any]:
    return {
        "output": [],
        "id": "deepswe-golden",
        "created_at": 0,
        "model": "golden",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


async def main(config: dict[str, Any]) -> int:
    server_name = str(config.get("resources_server", "deepswe_resources_server"))
    input_path = Path(str(config.get("benchmark_jsonl", "benchmarks/deepswe/data/deepswe_benchmark.jsonl")))
    output_path = Path(str(config.get("output_jsonl", "resources_servers/deepswe/logs/golden_results.jsonl")))
    concurrency = int(config.get("concurrency", 16))
    limit = config.get("limit")
    selected_task_ids = {str(task_id) for task_id in config.get("task_ids", [])}

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if selected_task_ids:
        rows = [row for row in rows if str((row.get("verifier_metadata") or {}).get("task_id")) in selected_task_ids]
        found_task_ids = {str((row.get("verifier_metadata") or {}).get("task_id")) for row in rows}
        missing_task_ids = selected_task_ids - found_task_ids
        if missing_task_ids:
            raise ValueError(f"Unknown DeepSWE task IDs: {', '.join(sorted(missing_task_ids))}")
    if limit is not None:
        rows = rows[: int(limit)]
    client = ServerClient.load_from_global_config()
    semaphore = asyncio.Semaphore(concurrency)

    async def validate(row: dict[str, Any]) -> dict[str, Any]:
        task_id = str((row.get("verifier_metadata") or {}).get("task_id", "unknown"))
        request = row | {"response": _empty_response()}
        try:
            async with semaphore:
                response = await client.post(server_name=server_name, url_path="/verify", json=request)
                await raise_for_status(response)
                return await get_response_json(response)
        except Exception as error:
            return {
                "task_id": task_id,
                "reward": 0.0,
                "evaluation_completed": False,
                "verifier_error": f"{type(error).__name__}: {error}",
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    completed = 0
    results: list[dict[str, Any]] = []
    progress = tqdm(total=len(rows))
    for future in asyncio.as_completed([validate(row) for row in rows]):
        result = await future
        results.append(result)
        completed += int(bool(result.get("evaluation_completed")))
        passed += int(result.get("reward") == 1)
        progress.set_description(f"golden={passed}/{len(results)} completed={completed}/{len(results)}")
        progress.update(1)
    progress.close()

    results.sort(key=lambda item: str(item.get("task_id", "")))
    with output_path.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    failed = [result for result in results if not result.get("evaluation_completed") or result.get("reward") != 1]
    print(f"DeepSWE golden patches: {passed}/{len(results)} passed; {completed}/{len(results)} completed")
    if failed:
        print("Failed task IDs:", ", ".join(str(result.get("task_id", "unknown")) for result in failed))
    print(f"Wrote results to {output_path.resolve()}")
    return 0 if not failed and len(results) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(get_global_config_dict())))
