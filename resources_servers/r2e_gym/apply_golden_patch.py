# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Grade every task with the dataset's own golden patch.

This measures the dataset, not a model. A row whose golden patch does not make FAIL_TO_PASS
pass is unusable for evaluation or training, so this is the check to run before trusting any
score from R2E-Gym.

The negative control is the same sweep with ``+empty_patch=true``: every row is graded with no
patch at all and must come back unresolved. A row that resolves without a fix is not testing
anything and has to leave the supported set.

    gym env start --config resources_servers/r2e_gym/configs/r2e_gym.yaml \
        --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml

    python resources_servers/r2e_gym/apply_golden_patch.py \
        +training_jsonl=resources_servers/r2e_gym/data/r2e_gym_training.jsonl \
        +output_jsonl=results/r2e_gym_golden_patch.jsonl \
        +concurrency=32 +limit=100
"""

import asyncio
import json
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.server_utils import ServerClient


EMPTY_RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}

SERVER_NAME = "r2e_gym_golden_patch_resources_server"


def _stream_examples(path: Path, limit: int):
    """Yield rows lazily rather than materialising the whole training file."""
    with open(path, encoding="utf-8") as training:
        for index, line in enumerate(training):
            if limit and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def build_payload(example: dict, empty_patch: bool = False) -> dict:
    """Verify request for one row: the golden patch stands in for the model patch, or an empty
    patch for the negative control. Same request shape either way, so the two sweeps compare."""
    payload = example | {"responses_create_params": {"input": []}, "response": EMPTY_RESPONSE}
    if empty_patch:
        payload["patch"] = ""
    return payload


async def main() -> None:
    config = get_global_config_dict()
    limit = int(config.get("limit") or 0)
    concurrency = int(config.get("concurrency") or 16)
    training_fpath = Path(config["training_jsonl"])
    output_fpath = Path(config.get("output_jsonl") or "results/r2e_gym_golden_patch.jsonl")
    empty_patch = bool(config.get("empty_patch") or False)
    mode = "empty-patch control" if empty_patch else "golden-patch validation"

    total = sum(1 for _ in _stream_examples(training_fpath, limit))
    client = ServerClient.load_from_global_config()

    # Bounded: the queue holds at most one row per worker, so resident rows track concurrency
    # rather than dataset size.
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency)
    resolved = 0
    incomplete = 0
    by_repo: Counter[str] = Counter()
    resolved_by_repo: Counter[str] = Counter()
    finished = 0

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=total, desc="empty patch" if empty_patch else "golden patch")

    with output_fpath.open("w", encoding="utf-8") as output:

        async def worker() -> None:
            nonlocal resolved, incomplete, finished
            while True:
                example = await queue.get()
                if example is None:
                    queue.task_done()
                    return
                try:
                    payload = build_payload(example, empty_patch)
                    try:
                        response = await client.post(server_name=SERVER_NAME, url_path="/verify", json=payload)
                        result = await response.json()
                    except Exception as exc:
                        # One unreachable task must not abandon the sweep; record it and carry on.
                        result = {
                            "instance_id": example.get("instance_id", "?"),
                            "repo_name": example.get("repo_name", ""),
                            "resolved": False,
                            "evaluation_completed": False,
                            "error": f"request failed: {exc}",
                        }
                    if not isinstance(result, dict):
                        # Gym middleware returns repr(exc) as a bare JSON string when a handler
                        # raises, so a server-side fault arrives as a str.
                        result = {
                            "instance_id": example.get("instance_id", "?"),
                            "repo_name": example.get("repo_name", ""),
                            "resolved": False,
                            "evaluation_completed": False,
                            "error": f"server returned a non-object response: {str(result)[:300]}",
                        }

                    repo = str(result.get("repo_name") or example.get("repo_name") or "unknown")
                    by_repo[repo] += 1
                    if result.get("resolved"):
                        resolved += 1
                        resolved_by_repo[repo] += 1
                    if not result.get("evaluation_completed", False):
                        incomplete += 1
                    output.write(json.dumps(result) + "\n")
                    finished += 1
                    if finished % 50 == 0:
                        output.flush()  # a multi-hour run must leave usable rows if it is cut short
                    progress.update(1)
                    progress.set_description(
                        f"resolved {resolved}/{finished} ({100 * resolved / max(finished, 1):.1f}%) "
                        f"incomplete={incomplete}"
                    )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        for example in _stream_examples(training_fpath, limit):
            await queue.put(example)
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    progress.close()
    print(f"\n{mode} over {total} task(s)")
    print(f"  resolved:   {resolved}/{total} ({100 * resolved / max(total, 1):.1f}%)")
    print(f"  incomplete: {incomplete}/{total}  (no verdict: image pull, timeout, or setup failure)")
    if empty_patch:
        print("  every resolved row is a false positive: its tests pass without a fix")
    print("\n  by repo (resolved / total):")
    for repo, count in by_repo.most_common():
        got = resolved_by_repo[repo]
        print(f"    {repo:<26} {got:>5} / {count:<6} ({100 * got / count:5.1f}%)")
    print(f"\nper-task rows: {output_fpath}")


if __name__ == "__main__":
    asyncio.run(main())
