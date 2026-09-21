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

"""Model-free validation of the CombiBench harness against a live Lean server.

Upstream publishes no reference proofs, so the strongest model-free checks are:

* ``statement``: every prepared statement, ``sorry`` and all, compiles with no
  error (only ``sorry`` warnings). A statement that does not compile cannot be
  solved by any model, and the failure would be charged to the model.
* ``gold_answer``: for fill-in-the-blank rows, the published answer substituted
  into its ``abbrev`` compiles, and the verifier's own equality check
  ``example : <tag> = <gold> := by try rfl; try norm_num`` elaborates. A gold
  answer of the wrong type would make the row unsolvable.
* Negative controls through the real verifier, over every row: empty output,
  statement echoed back with ``sorry``, an ``axiom``-based proof, and a
  weakened statement. Each must score 0 with the intended status.

Run from the repository root with a Lean server up:

    python resources_servers/combibench/scripts/harness_validation.py \
        --input benchmarks/combibench/data/combibench_test.jsonl \
        --output resources_servers/combibench/data/harness_validation_test.json

The report is committed next to the example data so the numbers in the README
can be re-derived.
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig, set_global_aiohttp_client
from resources_servers.combibench.app import (
    CombibenchResourcesServerConfig,
    CombibenchVerifier,
    CombibenchVerifyRequest,
)
from resources_servers.combibench.fine_eval import (
    abbrev_types,
    answer_check,
    answer_tags,
    classify_lean_result,
    statement_chunks,
)
from resources_servers.combibench.lean_client import KiminaLeanClient


_ABBREV_RE = re.compile(
    r"^(?P<prefix>(?:noncomputable )?abbrev\s+(?P<name>\S+_solution)\b[^\n]*?):=\s*sorry", re.MULTILINE
)


def _fenced(code: str) -> str:
    return f"```lean4\n{code}\n```"


def _response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "validation",
            "created_at": 0.0,
            "model": "none",
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }
    )


def _request(row: dict, text: str) -> CombibenchVerifyRequest:
    return CombibenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "validation"}]},
        response=_response(text),
        **{k: row.get(k) for k in ("theorem_name", "formal_statement", "answers", "tag", "split")},
    )


def substitute_gold(statement: str, answers: list[str]) -> Optional[str]:
    """Replace each ``abbrev ..._solution ... := sorry`` with the published answer, in order."""
    matches = list(_ABBREV_RE.finditer(statement))
    if len(matches) != len(answers):
        return None
    out = statement
    for match, answer in reversed(list(zip(matches, answers))):
        prefix = match.group("prefix")
        # Real-valued answers need it, and marking a computable definition
        # noncomputable is harmless; the statement check is a substring test,
        # so the modifier does not break it for a model either.
        if not prefix.startswith("noncomputable"):
            prefix = "noncomputable " + prefix
        out = out[: match.start()] + f"{prefix}:= {answer}" + out[match.end() :]
    return out


def weaken(statement: str) -> Optional[str]:
    """Replace the goal of the main theorem with ``True`` — the classic tamper."""
    match = re.search(r"(theorem|lemma)\s+\S+[\s\S]*?:=\s*by\s*sorry\s*$", statement.rstrip() + "\n")
    if match is None:
        return None
    head = match.group(0)
    # Keep the theorem name so the tamper is only in the goal.
    name = re.match(r"(theorem|lemma)\s+(\S+)", head).group(0)
    return statement[: match.start()] + f"{name} : True := by trivial\n"


async def compile_only(client: KiminaLeanClient, code: str, timeout: int) -> dict[str, Any]:
    result = await client.verify(code, timeout)
    return {"status": classify_lean_result(result), "error": result.error, "time": result.time}


async def run(
    rows: list[dict], verifier: CombibenchVerifier, client: KiminaLeanClient, timeout: int, concurrency: int
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    per_row: dict[str, dict[str, Any]] = {}

    async def one(row: dict) -> None:
        name = row["theorem_name"]
        statement = row["formal_statement"]
        record: dict[str, Any] = {"tag": row.get("tag")}
        async with semaphore:
            record["statement"] = await compile_only(client, statement, timeout)

            answers = row.get("answers")
            tags = answer_tags(statement_chunks(statement))
            if answers and tags:
                substituted = substitute_gold(statement, answers)
                if substituted is None:
                    record["gold_answer"] = {"status": "substitution_failed"}
                else:
                    types = (
                        abbrev_types(statement_chunks(statement))
                        if verifier.config.answer_check_ascription
                        else [None] * len(tags)
                    )
                    checks = "".join(answer_check(tag, gold, ty) for tag, gold, ty in zip(tags, answers, types))
                    record["gold_answer"] = await compile_only(client, substituted + checks, timeout)

            controls = {
                "empty": "",
                "echo_with_sorry": _fenced(statement),
                "axiom": _fenced("import Mathlib\n\naxiom cheat : False\n\n" + statement.split("\n\n", 1)[1]),
            }
            weakened = weaken(statement)
            if weakened is not None:
                controls["weakened"] = _fenced(weakened)
            record["controls"] = {}
            for control, text in controls.items():
                verdict = await verifier.verify(_request(row, text))
                record["controls"][control] = {"reward": verdict.reward, "status": verdict.status}
        per_row[name] = record
        print(f"{name}: statement={record['statement']['status']}", file=sys.stderr)

    await asyncio.gather(*(one(row) for row in rows))

    def tally(getter) -> dict[str, int]:
        return dict(Counter(getter(r) for r in per_row.values() if getter(r) is not None))

    summary = {
        "rows": len(rows),
        "statement_status": tally(lambda r: r["statement"]["status"]),
        "statement_failures": sorted(n for n, r in per_row.items() if r["statement"]["status"] != "has_sorry"),
        "gold_answer_rows": sum(1 for r in per_row.values() if "gold_answer" in r),
        "gold_answer_status": tally(lambda r: r.get("gold_answer", {}).get("status")),
        "gold_answer_failures": sorted(
            n for n, r in per_row.items() if "gold_answer" in r and r["gold_answer"]["status"] != "has_sorry"
        ),
        "controls": {},
    }
    for control in ("empty", "echo_with_sorry", "axiom", "weakened"):
        applicable = [r["controls"][control] for r in per_row.values() if control in r["controls"]]
        summary["controls"][control] = {
            "denominator": len(applicable),
            "nonzero_reward": sum(1 for c in applicable if c["reward"] != 0.0),
            "status": dict(Counter(c["status"] for c in applicable)),
        }
    return {"summary": summary, "rows": per_row}


async def run_solutions(rows: list[dict], verifier: CombibenchVerifier, solutions: dict[str, str]) -> dict:
    """Gold-as-prediction: complete proofs (where known) must score 1.0 through the verifier."""
    by_name = {row["theorem_name"]: row for row in rows}
    results = {}
    for name, text in solutions.items():
        if name not in by_name:
            results[name] = {"status": "not_in_input"}
            continue
        verdict = await verifier.verify(_request(by_name[name], text))
        results[name] = {
            "reward": verdict.reward,
            "status": verdict.status,
            "lean_error": verdict.lean_error,
            "errors": [m.get("data") for m in verdict.lean_messages if m.get("severity") == "error"],
        }
    return {
        "passed": sum(1 for r in results.values() if r.get("reward") == 1.0),
        "total": len(results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-free CombiBench harness validation")
    parser.add_argument("--input", type=Path, required=True, help="Prepared JSONL (test or test_with_solution)")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report")
    parser.add_argument("--lean-server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=60, help="Lean timeout per compile, seconds")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-ascription",
        action="store_true",
        help="Use upstream's unascribed `example : tag = gold` answer check instead of the default ascribed form",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=None,
        help="JSON mapping theorem_name -> model-style output with a full proof; each must score 1.0",
    )
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    config = CombibenchResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="",
        name="combibench",
        lean_server_url=args.lean_server_url,
        lean_timeout_seconds=args.timeout,
        answer_check_ascription=not args.no_ascription,
    )
    client = KiminaLeanClient(args.lean_server_url)
    verifier = CombibenchVerifier(config, client)

    async def runner():
        # get_global_aiohttp_client() would fall through to a Hydra CLI parse that
        # rejects this script's own flags, so the shared client is set directly.
        set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
        report = await run(rows, verifier, client, args.timeout, args.concurrency)
        if args.solutions is not None:
            report["solutions"] = await run_solutions(
                rows, verifier, json.loads(args.solutions.read_text(encoding="utf-8"))
            )
        return report

    report = asyncio.run(runner())
    report["input"] = str(args.input)
    report["answer_check_ascription"] = not args.no_ascription
    report["lean_server_url"] = args.lean_server_url
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
