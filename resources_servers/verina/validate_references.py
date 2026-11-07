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
"""Score Verina's reference solutions, and three broken variants of each, through the real verifier.

Expected results, with the default `tests_only_reward` of 0.5:
- reference: 1.0 when the reference proof is complete, 0.5 when Verina ships it as `sorry`;
- sorry proof: 0.5 (the function still passes the tests);
- no proof: 0.5;
- wrong code (the body replaced by `default`): 0.0 unless `default` happens to pass every test.

Run `prepare_verina.py` first. Every row whose reward differs from the expectation is printed.
"""

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from resources_servers.verina.app import VerinaResourcesServerConfig, VerinaVerifyRequest, _fixture_verifier


def response_with(text: str) -> dict:
    return {
        "id": "reference",
        "created_at": 0,
        "model": "reference",
        "object": "response",
        "output": [
            {
                "id": "reference_message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def variants(answer: str) -> dict[str, str]:
    replace_proof = lambda proof: re.sub(r"<proof>.*?</proof>", f"<proof>\n{proof}\n</proof>", answer, flags=re.S)  # noqa: E731
    return {
        "reference": answer,
        "sorry proof": replace_proof("sorry"),
        "no proof": replace_proof(""),
        "wrong code": re.sub(r"<code>.*?</code>", "<code>\ndefault\n</code>", answer, flags=re.S),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", default="reference,sorry proof,no proof,wrong code")
    parser.add_argument("--output", type=Path, default=None, help="Write one JSON line per scored variant.")
    args = parser.parse_args()

    rows = {}
    for split in ("train", "validation"):
        for line in (args.data_dir / f"{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            rows[row["task_id"]] = row
    answers = [json.loads(line) for line in (args.data_dir / "reference_answers.jsonl").read_text().splitlines()]
    answers = answers[: args.limit]
    wanted = args.variants.split(",")

    verifier = _fixture_verifier()
    tests_only = VerinaResourcesServerConfig.model_fields["tests_only_reward"].default

    async def score(task_id: str, kind: str, text: str) -> dict:
        request = VerinaVerifyRequest(**rows[task_id], response=response_with(text))
        result = await verifier.verify(request)
        return {
            "task_id": task_id,
            "variant": kind,
            "reward": result.reward,
            "tests": f"{result.tests_passed}/{result.tests_total}",
            "proof_verified": result.proof_verified,
            "timed_out": result.timed_out,
            "errors": result.lean_errors[:2],
            "forbidden": result.forbidden,
        }

    jobs = []
    for answer in answers:
        for kind, text in variants(answer["answer"]).items():
            if kind in wanted:
                jobs.append(score(answer["task_id"], kind, text))
    results = await asyncio.gather(*jobs)

    reference_is_sorry = {
        answer["task_id"]: bool(re.search(r"<proof>\s*sorry\s*</proof>", answer["answer"])) for answer in answers
    }
    expected = {"sorry proof": tests_only, "no proof": tests_only, "wrong code": 0.0}
    summary: Counter = Counter()
    for result in results:
        want = expected.get(result["variant"])
        if result["variant"] == "reference":
            want = tests_only if reference_is_sorry[result["task_id"]] else 1.0
        ok = result["reward"] == want
        summary[(result["variant"], result["reward"], "ok" if ok else "UNEXPECTED")] += 1
        if not ok:
            print(json.dumps(result, ensure_ascii=False))
    for key, count in sorted(summary.items()):
        print(count, *key)
    if args.output:
        with args.output.open("w") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
