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

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.swebench.pro.prepare import OUTPUT_FPATH as PRO_OUTPUT_FPATH
from benchmarks.swebench.pro_multiturn.prepare import (
    DECOMPOSITIONS_FPATH,
    SPEC_FIELDS,
    build_row,
    render_turn,
    spec_text,
)


WRITER_INTRO = """You turn a software task specification into a short sequence of user requests that a developer sends to a coding assistant one at a time, in the same chat session, waiting for each to be finished before sending the next.

You get the PROBLEM STATEMENT plus the REQUIREMENTS and INTERFACE texts with their line numbers. Return one JSON object:
{"decomposable": true, "turns": [{"request": "...", "requirements": [[start, end], ...], "interface": [[start, end], ...]}, ...]}

Splitting:
- Use 2 to 4 turns. Each turn is one coherent piece of work that makes sense to ask for on its own. Order them the way a developer would build the feature: foundations first, then what uses them, then integration, edge cases, user-facing surfaces. No turn may depend on a later one.
- Assign every numbered REQUIREMENTS and INTERFACE line to exactly one turn with inclusive [start, end] line ranges. Keep each bullet together with its sub-bullets and continuation lines, and keep all lines of one interface entry together. Put each requirement and interface entry in the turn where that piece gets built. A line that only says the golden patch introduces interfaces can be left out.
- If the task is one small change that cannot be split into at least two natural requests, return {"decomposable": false, "reason": "...", "turns": []}.

"""
REQUEST_STYLE = """Writing each "request":
- First person, plain prose, 1 to 3 sentences, no headings, lists or code blocks. Write it in your own words, as the developer asking for the next piece of work.
- The assistant never sees the problem statement. The first request says briefly, in your own words, what is wrong or wanted overall. Later requests start with a neutral transition such as "Next," or "Now" and say what to work on.
- Each turn's requirement and interface lines are appended after its request verbatim. Do not restate them or the lines of earlier turns: no behaviors, conditions, values or signatures they already give. Name the piece of work in a few words and leave the details to those lines.
- Do not reproduce the problem statement: no sentences, phrases, examples, reproduction steps, error messages, code or config taken from it.

"""
REQUEST_RULES = """Hard rules for requests; violations are rejected automatically:
- Use only names, identifiers, file paths, commands and values that appear verbatim in the PROBLEM STATEMENT or in the requirement and interface lines assigned to that turn or an earlier one. Introduce nothing else: no new files, functions, libraries, tools or implementation ideas.
- Never mention tests, testing, test files, graders, hidden checks, benchmarks or rewards, and never say how the work will be checked.
- Never preview later requests ("later we will", "we'll also", "next we'll").
- Never give feedback on earlier turns ("that works", "looks good", "that's correct"); the developer has not checked the work.
- Never ask to change, skip or disable tests or checks, to look up the fix online, or to read git history.
- No run of eight or more words may match the PROBLEM STATEMENT or the requirement and interface lines of that turn or an earlier one.

"""
WRITER_SYSTEM = WRITER_INTRO + REQUEST_STYLE + REQUEST_RULES + "\nReturn only the JSON object."
REWRITE_SYSTEM = (
    """You write the user requests of a multi-turn coding session. A developer sends the turns to a coding assistant one at a time, in the same chat session, waiting for each to be finished before sending the next. Each turn is a short request followed by requirement and interface lines that are sent verbatim. The split into turns is fixed; you only write the requests.

You get the PROBLEM STATEMENT, the REQUIREMENTS and INTERFACE texts, and the lines each of the TURNS will carry. Return one JSON object {"requests": ["...", ...]} with exactly one request per turn, in order.

"""
    + REQUEST_STYLE
    + REQUEST_RULES
    + "\nReturn only the JSON object."
)


JUDGE_SYSTEM = """You audit a multi-turn rewrite of a software task. A developer will send the TURNS to a coding assistant one at a time. Each turn is a short request written for this rewrite, followed by requirement and interface lines copied verbatim from the ORIGINAL TASK.

Judge only the request prose at the start of each turn (before "Requirements:"); the copied lines are original and fine. Answer with one JSON object of booleans plus evidence:
{"adds_information": bool, "copies_original": bool, "restates_lines": bool, "mentions_checking": bool, "reveals_later_turns": bool, "gives_feedback": bool, "unnatural": bool, "missing_context": bool, "evidence": "..."}

- adds_information: a request states a fact, name, value, file, behavior, hint or implementation idea that the ORIGINAL TASK does not contain.
- copies_original: a request reproduces wording, examples, reproduction steps, messages, code or config from the PROBLEM STATEMENT instead of saying in its own words what is wanted.
- restates_lines: a request repeats behaviors, conditions, values, signatures or other details that the requirement or interface lines of its own or an earlier turn already state. Naming the piece of work in a few words is fine.
- mentions_checking: a request mentions tests, grading or how the work will be verified, or encourages changing tests or checks.
- reveals_later_turns: a request reveals or hints at work that a later turn asks for.
- gives_feedback: a request comments on, approves or praises earlier work.
- unnatural: a request is confusing or incoherent, a turn depends on a later turn, the order would not make sense to a developer, or a bullet or interface entry is split across turns.
- missing_context: the first turn does not give enough context to understand what is being built.

Quote the offending text in "evidence" for every true flag. Return only the JSON object."""

JUDGE_FLAGS = (
    "adds_information",
    "copies_original",
    "restates_lines",
    "mentions_checking",
    "reveals_later_turns",
    "gives_feedback",
    "unnatural",
    "missing_context",
)
EXEMPLAR_KEYS = ("teleport-eefac60a", "NodeBB-18c45b44")


def numbered(text: str) -> str:
    return "\n".join(f"L{number}: {line}" for number, line in enumerate(text.splitlines(), 1) if line.strip())


def task_prompt(row: Mapping[str, Any]) -> str:
    return (
        f"PROBLEM STATEMENT:\n{spec_text(row, 'problem_statement').strip()}\n\n"
        f"REQUIREMENTS:\n{numbered(spec_text(row, 'requirements'))}\n\n"
        f"INTERFACE:\n{numbered(spec_text(row, 'interface'))}"
    )


def turns_prompt(row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> str:
    return f"{task_prompt(row)}\n\nTURNS:\n" + "\n\n".join(
        f"=== TURN {index} ===\n{render_turn(row, turn | {'request': ''}).strip()}"
        for index, turn in enumerate(turns, 1)
    )


def split_issues(row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = []
    for field in SPEC_FIELDS:
        lines = spec_text(row, field).splitlines()
        indent = [len(line) - len(line.lstrip()) if line.strip() else None for line in lines]
        for index, turn in enumerate(turns, 1):
            for start, end in turn[field]:
                if indent[start - 1] is not None and indent[start - 1] >= 2:
                    issues.append(f"turn {index} {field} range {[start, end]} starts inside a bullet")
                following = next((value for value in indent[end:] if value is not None), None)
                if following is not None and following >= 2:
                    issues.append(f"turn {index} {field} range {[start, end]} ends inside a bullet")
    return issues


def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^.*?</think>", "", text, flags=re.S).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))


class Client:
    def __init__(self, base_url: str, model: str, concurrency: int, extra_body: dict[str, Any]) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.extra_body = extra_body
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600))
        self.usage: Counter[str] = Counter()

    async def chat(self, system: str, user: str, *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            **self.extra_body,
        }
        for attempt in range(5):
            try:
                async with self.semaphore, self.session.post(self.url, json=payload) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
                    data = await response.json()
                usage = data.get("usage") or {}
                self.usage["requests"] += 1
                self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
                return data["choices"][0]["message"].get("content") or ""
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, KeyError) as exc:
                self.usage["retries"] += 1
                if attempt == 4:
                    raise RuntimeError(f"chat failed: {exc!r}") from exc
                await asyncio.sleep(min(60, 5 * 2**attempt))
        raise AssertionError("unreachable")


def exemplar_messages(
    rows: Mapping[str, Mapping[str, Any]], decompositions: Mapping[str, Any], *, rewrite: bool
) -> str:
    blocks = []
    for key in EXEMPLAR_KEYS:
        instance_id = next(iid for iid in decompositions if key in iid)
        turns = decompositions[instance_id]
        if rewrite:
            reply = {"requests": [turn["request"] for turn in turns]}
            blocks.append(f"EXAMPLE INPUT:\n{turns_prompt(rows[instance_id], turns)}\n\nEXAMPLE OUTPUT:\n")
        else:
            reply = {"decomposable": True, "turns": turns}
            blocks.append(f"EXAMPLE INPUT:\n{task_prompt(rows[instance_id])}\n\nEXAMPLE OUTPUT:\n")
        blocks[-1] += json.dumps(reply, ensure_ascii=False)
    return "\n\n".join(blocks)


async def judge(client: Client, row: Mapping[str, Any], turns: Sequence[Mapping[str, Any]], samples: int) -> dict:
    rendered = "\n\n".join(f"=== TURN {index} ===\n{render_turn(row, turn)}" for index, turn in enumerate(turns, 1))
    user = f"ORIGINAL TASK:\n{task_prompt(row)}\n\nTURNS:\n{rendered}"
    replies = await asyncio.gather(
        *(
            client.chat(JUDGE_SYSTEM, user, temperature=0.0 if i == 0 else 0.6, max_tokens=16000)
            for i in range(samples)
        ),
        return_exceptions=True,
    )
    votes = []
    for reply in replies:
        try:
            votes.append(parse_json(reply) if isinstance(reply, str) else {"error": repr(reply)})
        except (ValueError, json.JSONDecodeError) as exc:
            votes.append({"error": repr(exc)})
    valid = [vote for vote in votes if "error" not in vote]
    flagged = [flag for flag in JUDGE_FLAGS if sum(bool(vote.get(flag)) for vote in valid) * 2 > len(valid)]
    return {"votes": votes, "flagged": flagged, "valid": len(valid)}


async def decompose(
    client: Client,
    row: Mapping[str, Any],
    exemplars: str,
    args: argparse.Namespace,
    current: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    feedback, history = "", []
    if current is None:
        system, task = WRITER_SYSTEM, f"{exemplars}\n\nNOW THE REAL TASK.\n\nINPUT:\n{task_prompt(row)}"
    else:
        system, task = REWRITE_SYSTEM, f"{exemplars}\n\nNOW THE REAL TASK.\n\nINPUT:\n{turns_prompt(row, current)}"
    for attempt in range(1, args.attempts + 1):
        try:
            reply = parse_json(await client.chat(system, task + feedback, temperature=0.7, max_tokens=24000))
        except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
            history.append({"attempt": attempt, "error": repr(exc)})
            continue
        if current is not None:
            requests = [str(request).strip() for request in reply.get("requests") or []]
            turns = [dict(turn) | {"request": request} for turn, request in zip(current, requests)]
        elif reply.get("decomposable") is False:
            return {"status": "declined", "reason": reply.get("reason"), "history": history}
        else:
            turns = [
                {
                    "request": str(t.get("request", "")),
                    "requirements": t.get("requirements") or [],
                    "interface": t.get("interface") or [],
                }
                for t in reply.get("turns") or []
            ]
        try:
            if current is not None and len(requests) != len(current):
                raise ValueError(f"return exactly {len(current)} requests, not {len(requests)}")
            build_row(row, turns)
            issues = split_issues(row, turns)
        except (ValueError, IndexError, TypeError) as exc:
            issues = [str(exc).removeprefix(f"{row['instance_id']}: ")]
        if issues:
            history.append({"attempt": attempt, "turns": turns, "issues": issues})
            feedback = "\n\nYOUR PREVIOUS ANSWER WAS REJECTED:\n- " + "\n- ".join(issues) + "\nFix every problem."
            continue
        verdict = await judge(client, row, turns, args.judge_samples)
        if not verdict["flagged"] and verdict["valid"] * 2 > args.judge_samples:
            return {"status": "accepted", "turns": turns, "judge": verdict, "history": history}
        evidence = [vote.get("evidence", "") for vote in verdict["votes"] if "error" not in vote]
        history.append({"attempt": attempt, "turns": turns, "judge": verdict})
        feedback = (
            "\n\nYOUR PREVIOUS ANSWER WAS REJECTED BY REVIEWERS for: "
            + ", ".join(verdict["flagged"] or ["unparseable reviews"])
            + "\nReviewer notes:\n- "
            + "\n- ".join(e for e in evidence if e)
            + "\nFix every problem."
        )
    return {"status": "rejected", "history": history}


async def run(args: argparse.Namespace) -> None:
    rows = {row["instance_id"]: row for row in map(json.loads, args.source.open(encoding="utf-8"))}
    hand_written = json.loads(DECOMPOSITIONS_FPATH.read_text(encoding="utf-8"))
    exemplars = exemplar_messages(rows, hand_written, rewrite=args.rewrite)
    wanted = set(args.instance_ids.read_text().split()) if args.instance_ids else set(rows)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["instance_id"] for line in args.output.open(encoding="utf-8")}
    if args.rewrite:
        todo = [
            iid
            for iid in rows
            if iid in wanted
            and iid in hand_written
            and iid not in done
            and not any(key in iid for key in EXEMPLAR_KEYS)
        ][: args.limit]
    else:
        todo = [iid for iid in rows if iid in wanted and iid not in hand_written and iid not in done][: args.limit]
    print(f"{len(todo)} tasks to decompose ({len(done)} already in {args.output})", file=sys.stderr)

    client = Client(args.base_url, args.model, args.concurrency, json.loads(args.extra_body))
    statuses: Counter[str] = Counter()
    lock = asyncio.Lock()

    task_semaphore = asyncio.Semaphore(max(1, args.concurrency // 2))

    async def one(instance_id: str) -> None:
        try:
            async with task_semaphore:
                current = hand_written[instance_id] if args.rewrite else None
                result = await decompose(client, rows[instance_id], exemplars, args, current)
        except RuntimeError as exc:
            result = {"status": "error", "error": repr(exc)}
        async with lock:
            statuses[result["status"]] += 1
            with args.output.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"instance_id": instance_id, **result}, ensure_ascii=False) + "\n")
            print(f"{sum(statuses.values())}/{len(todo)} {dict(statuses)} usage={dict(client.usage)}", file=sys.stderr)

    try:
        await asyncio.gather(*(one(instance_id) for instance_id in todo))
    finally:
        await client.session.close()


def merge(args: argparse.Namespace) -> None:
    decompositions = json.loads(DECOMPOSITIONS_FPATH.read_text(encoding="utf-8"))
    allowed = set(args.instance_ids.read_text().split()) if args.instance_ids else None
    added = 0
    for record in map(json.loads, args.output.open(encoding="utf-8")):
        if allowed is not None and record["instance_id"] not in allowed:
            continue
        if record["status"] == "accepted" and (args.replace or record["instance_id"] not in decompositions):
            decompositions[record["instance_id"]] = record["turns"]
            added += 1
    DECOMPOSITIONS_FPATH.write_text(json.dumps(decompositions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {added} decompositions; {len(decompositions)} total in {DECOMPOSITIONS_FPATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--source", type=Path, default=PRO_OUTPUT_FPATH)
    parser.add_argument("--instance-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--judge-samples", type=int, default=3)
    parser.add_argument("--extra-body", default="{}")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
