# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate the Gym VERA-MH adapter against the official upstream implementation.

All controls run the *upstream* code from a pinned checkout of https://github.com/SpringCare/VERA-MH (commit
``2c9d1fc``) in its own virtual environment, so they are subprocess calls:

``materialization``
    Persona prompts (``load_prompts_from_csv``), transcript rendering (``format_conversation_summary``) and message
    construction (``build_langchain_messages``) recomputed by upstream and compared with the Gym rows/agent.

``replay``
    Every Gym judge answer sequence is fed to upstream's ``LLMJudge`` through a mock LLM that returns the same
    answers, and the resulting dimension ratings are compared case by case (must agree exactly).

``score``
    Gym ratings are written as an upstream ``results.csv`` and scored with ``judge.score.score_results``; the pooled
    VERA-MH score and percentages must equal Gym's ``compute_metrics``.

``live``
    Upstream ``judge.py`` re-judges a stratified subset of Gym transcripts (written in upstream's ``.txt`` layout)
    with the same judge model through an OpenAI-compatible endpoint (``OPENAI_BASE_URL``/``OPENAI_API_KEY``), and
    the per-dimension ratings are compared. The judge samples, so this measures rating stability.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from resources_servers.vera_mh.app import DAMAGING, NOT_RELEVANT, pooled_scores


DIMENSIONS = [
    "Detects Potential Risk",
    "Confirms Risk",
    "Guides to Human Care",
    "Supportive Conversation",
    "Follows AI Boundaries",
]

UPSTREAM_REPLAY_SCRIPT = r"""
import asyncio, json, sys
sys.path.insert(0, ".")
from judge.llm_judge import LLMJudge
from judge.rubric_config import ConversationData, RubricConfig
from tests.mocks.mock_llm import MockLLM
from llm_clients.llm_interface import Role

payload = json.load(open(sys.argv[1]))
rubric = asyncio.run(RubricConfig.load(rubric_folder="data"))
out = []
for case in payload:
    answers = case["answers"]  # ordered {question_id: {"answer":..., "reasoning":...}}
    class _Mock(MockLLM):
        def __init__(self, *a, **k):
            super().__init__(name="mock", role=Role.JUDGE, system_prompt=k.get("system_prompt"))
            self._q = list(answers)
            self._i = 0
        async def generate_structured_response(self, message, response_model):
            item = answers[self._q[self._i]]; self._i += 1
            return response_model(answer=item["answer"], reasoning=item["reasoning"])
    import llm_clients.llm_factory as factory
    factory.LLMFactory.create_llm = staticmethod(lambda model_name, name, role, system_prompt=None, **kw: _Mock(system_prompt=system_prompt))
    import judge.llm_judge as lj
    lj.LLMFactory = factory.LLMFactory
    judge = LLMJudge(judge_model="mock", rubric_config=rubric, log_file=str(__import__("pathlib").Path(sys.argv[2]) / (case["id"] + ".log")))
    conversation = ConversationData(content=case["transcript"], metadata={"filename": case["id"] + ".txt", "run_id": "replay", "source_path": ""})
    results = asyncio.run(judge.evaluate_conversation_question_flow(conversation, output_folder=sys.argv[2], auto_save=False))
    out.append({"id": case["id"], "ratings": {d: r["score"] for d, r in results.items()}, "yes_question_ids": {d: r.get("yes_question_id", "") for d, r in results.items()}})
json.dump(out, open(sys.argv[3], "w"))
"""

UPSTREAM_SCORE_SCRIPT = r"""
import json, sys
sys.path.insert(0, ".")
from judge.score import score_results
results = score_results(sys.argv[1], output_path=sys.argv[2])
json.dump({"vera_score": results["aggregates"]["vera_score"], "bp": results["aggregates"]["overall_best_practice_pct"], "hph": results["aggregates"]["overall_damaging_pct"], "neutral": results["aggregates"]["overall_neutral_pct"], "total": results["summary"]["total_dimension_evaluations"], "dimensions": {d: v["vera_score"] for d, v in results["dimensions"].items()}}, open(sys.argv[3], "w"))
"""

UPSTREAM_MATERIALIZATION_SCRIPT = r"""
import json, sys
sys.path.insert(0, ".")
from generate_conversations.utils import load_prompts_from_csv
from utils.conversation_utils import format_conversation_summary, build_langchain_messages
from llm_clients.llm_interface import Role
payload = json.load(open(sys.argv[1]))
out = {"persona_prompts": {p["Name"]: p["prompt"] for p in load_prompts_from_csv()}, "transcripts": {}, "messages": {}}
for case in payload:
    out["transcripts"][case["id"]] = format_conversation_summary(case["conversation"])
    out["messages"][case["id"]] = {role: [("assistant" if m.type == "ai" else "user", m.text) for m in build_langchain_messages(getattr(Role, role.upper()), case["conversation"])] for role in ("persona", "provider")}
json.dump(out, open(sys.argv[2], "w"))
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_upstream(python: Path, repo: Path, script: str, *args: str) -> None:
    """Run an upstream-side script in the pinned checkout's environment; paths are made absolute."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    absolute = [str(Path(arg).resolve()) if Path(arg).parent.exists() else arg for arg in args]
    result = subprocess.run([str(python), script_path, *absolute], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"upstream control failed: {result.stderr[-2000:]}")


def replay(rollouts, python, repo, workdir):
    payload = []
    for row in rollouts:
        answers = {}
        for step in row.get("question_path", []):
            if step.get("special_case"):
                continue
            answers[step["question_id"]] = {
                "answer": step.get("raw_answer", step["answer"]),
                "reasoning": step.get("reasoning", ""),
            }
        payload.append({"id": row["id"], "transcript": row.get("transcript") or "", "answers": answers})
    inp = workdir / "replay-in.json"
    out = workdir / "replay-out.json"
    inp.write_text(json.dumps(payload), encoding="utf-8")
    (workdir / "replay-logs").mkdir(exist_ok=True)
    _run_upstream(python, repo, UPSTREAM_REPLAY_SCRIPT, str(inp), str(workdir / "replay-logs"), str(out))
    upstream = {item["id"]: item for item in json.loads(out.read_text(encoding="utf-8"))}
    cases, agreements = [], 0
    for row in rollouts:
        up = upstream[row["id"]]
        agree = up["ratings"] == row.get("ratings") and up["yes_question_ids"] == row.get("yes_question_ids")
        agreements += agree
        cases.append(
            {
                "case_id": f"{row['id']}/replay",
                "control": "replay",
                "task_id": row["id"],
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream": up,
                "gym": {"ratings": row.get("ratings"), "yes_question_ids": row.get("yes_question_ids")},
                "agree": agree,
            }
        )
    return cases, {
        "control": "replay",
        "cases": len(cases),
        "agreement": agreements,
        "disagreements": len(cases) - agreements,
    }


def materialization(rollouts, python, repo, workdir):
    payload = [{"id": row["id"], "conversation": row.get("conversation") or []} for row in rollouts]
    inp = workdir / "mat-in.json"
    out = workdir / "mat-out.json"
    inp.write_text(json.dumps(payload), encoding="utf-8")
    _run_upstream(python, repo, UPSTREAM_MATERIALIZATION_SCRIPT, str(inp), str(out))
    upstream = json.loads(out.read_text(encoding="utf-8"))
    from responses_api_agents.vera_mh_agent.app import REMINDER_PATH, build_messages, format_transcript

    reminder = REMINDER_PATH.read_text(encoding="utf-8")
    cases, agreements = [], 0
    for row in rollouts:
        conv = row.get("conversation") or []
        prompt_ok = upstream["persona_prompts"].get(row.get("persona_name")) == row.get("persona_system_prompt")
        transcript_ok = upstream["transcripts"][row["id"]] == format_transcript(conv) == (row.get("transcript") or "")
        messages_ok = all(
            [tuple(m) for m in upstream["messages"][row["id"]][role]]
            == [(m["role"], m["content"]) for m in build_messages(role, conv, reminder)]
            for role in ("persona", "provider")
        )
        agree = prompt_ok and transcript_ok and messages_ok
        agreements += agree
        cases.append(
            {
                "case_id": f"{row['id']}/materialization",
                "control": "materialization",
                "task_id": row["id"],
                "persona_prompt_agree": prompt_ok,
                "transcript_agree": transcript_ok,
                "messages_agree": messages_ok,
                "agree": agree,
            }
        )
    return cases, {
        "control": "materialization",
        "cases": len(cases),
        "agreement": agreements,
        "disagreements": len(cases) - agreements,
    }


def score(rollouts, python, repo, workdir):
    csv_path = workdir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "run_id", "judge_model", "judge_instance", "judge_id"] + DIMENSIONS
        )
        writer.writeheader()
        for row in rollouts:
            writer.writerow(
                {
                    "filename": row["id"] + ".txt",
                    "run_id": "gym",
                    "judge_model": row.get("judge_model"),
                    "judge_instance": 1,
                    "judge_id": 0,
                    **{d: row.get("ratings", {}).get(d, "") for d in DIMENSIONS},
                }
            )
    out = workdir / "score-out.json"
    _run_upstream(python, repo, UPSTREAM_SCORE_SCRIPT, str(csv_path), str(workdir / "scores.json"), str(out))
    upstream = json.loads(out.read_text(encoding="utf-8"))
    gym = pooled_scores(rollouts, DIMENSIONS)
    agree = (
        abs(upstream["vera_score"] - gym["vera_score"]) < 1e-6
        and abs(upstream["bp"] - gym["overall_best_practice_pct"]) < 1e-6
        and abs(upstream["hph"] - gym["overall_damaging_pct"]) < 1e-6
        and upstream["total"] == gym["total_dimension_evaluations"]
    )
    dims_agree = all(
        abs(upstream["dimensions"].get(d, 0.0) - gym["dimensions"][d]["vera_score"]) < 1e-6
        for d in DIMENSIONS
        if d in upstream["dimensions"]
    )
    case = {
        "case_id": "pooled/score",
        "control": "score",
        "upstream": upstream,
        "gym": {k: v for k, v in gym.items() if k != "dimensions"},
        "gym_dimensions": {d: gym["dimensions"][d]["vera_score"] for d in DIMENSIONS},
        "agree": agree and dims_agree,
    }
    return [case], {
        "control": "score",
        "cases": 1,
        "agreement": int(agree and dims_agree),
        "disagreements": int(not (agree and dims_agree)),
    }


def stratified_subset(rollouts, size, seed=20251211):
    def stratum(row):
        ratings = row.get("ratings", {})
        if row.get("all_not_relevant"):
            return "all_not_relevant"
        if any(v == DAMAGING for v in ratings.values()):
            return "high_harm"
        if all(v in ("Best Practice", NOT_RELEVANT) for v in ratings.values()):
            return "best_practice"
        return "suboptimal"

    groups = collections.defaultdict(list)
    for row in rollouts:
        groups[(stratum(row), row.get("user_simulator"))].append(row)
    rng = random.Random(seed)
    picked = []
    per_group = max(1, size // max(1, len(groups)))
    for key in sorted(groups, key=str):
        pool = sorted(groups[key], key=lambda row: row["id"])
        rng.shuffle(pool)
        picked.extend(pool[:per_group])
    remaining = [row for row in rollouts if row not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, size - len(picked))])
    return picked[:size]


def live(rollouts, python, repo, workdir, size, judge_model, judge_extra):
    subset = stratified_subset(rollouts, size)
    conv_dir = workdir / "live-transcripts"
    conv_dir.mkdir(exist_ok=True)
    id_by_file = {}
    for index, row in enumerate(subset):
        filename = f"{index:06x}_{row['persona_name']}_{str(row['user_simulator']).replace('-', '_')}_run1.txt"
        (conv_dir / filename).write_text(row.get("transcript") or "", encoding="utf-8")
        id_by_file[filename] = row["id"]
    out_dir = workdir / "live-evaluations"
    cmd = [
        str(python),
        "judge.py",
        "-f",
        str(conv_dir.resolve()),
        "-j",
        judge_model,
        "-o",
        str(out_dir.resolve()),
        "-m",
        "4",
    ]
    if judge_extra:
        cmd += ["-jep", judge_extra]
    completed = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"upstream judge.py failed: {completed.stderr[-2000:]}")
    # judge.py names its run folder after the judge model; gateway slugs such as "@openai/gpt-5.4"
    # contain a slash, so the folder can be nested one level deeper than upstream's plain names.
    results_files = sorted(out_dir.rglob("results.csv"))
    if not results_files:
        raise RuntimeError(f"upstream judge.py wrote no results.csv under {out_dir}")
    results = list(csv.DictReader(results_files[-1].open(encoding="utf-8")))
    # results.csv re-derives the transcript name from the per-judge TSV name, so match on the six-hex-digit
    # index prefix that this function assigned instead of on the full filename.
    id_by_prefix = {name[:6]: task_id for name, task_id in id_by_file.items()}
    by_id = {id_by_prefix[r["filename"][:6]]: r for r in results if r["filename"][:6] in id_by_prefix}
    cases, exact = [], 0
    dim_agree = dim_total = 0
    for row in subset:
        up = by_id.get(row["id"])
        if up is None:
            cases.append(
                {
                    "case_id": f"{row['id']}/live",
                    "control": "live",
                    "task_id": row["id"],
                    "upstream": None,
                    "gym": row.get("ratings"),
                    "agree": False,
                    "note": "upstream skipped this transcript",
                }
            )
            continue
        up_ratings = {d: up.get(d, "") for d in DIMENSIONS}
        agree = up_ratings == row.get("ratings")
        exact += agree
        for d in DIMENSIONS:
            dim_total += 1
            dim_agree += up_ratings[d] == row.get("ratings", {}).get(d)
        cases.append(
            {
                "case_id": f"{row['id']}/live",
                "control": "live",
                "task_id": row["id"],
                "rollout_id": f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                "upstream": up_ratings,
                "gym": row.get("ratings"),
                "agree": agree,
            }
        )
    return cases, {
        "control": "live",
        "cases": len(cases),
        "agreement": exact,
        "disagreements": len(cases) - exact,
        "dimension_agreement_rate": dim_agree / dim_total if dim_total else None,
        "judge_model": judge_model,
        "judge_extra_params": judge_extra,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--upstream-python", type=Path, required=True)
    parser.add_argument("--live-size", type=int, default=0)
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--judge-extra-params", default="reasoning_effort=low")
    args = parser.parse_args(argv)
    rollouts = _read_jsonl(args.rollouts)
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "work"
    workdir.mkdir(exist_ok=True)
    cases, summaries = [], []
    for control in (materialization, replay, score):
        control_cases, summary = control(rollouts, args.upstream_python, args.upstream_repo, workdir)
        cases += control_cases
        summaries.append(summary)
    if args.live_size:
        live_cases, live_summary = live(
            rollouts,
            args.upstream_python,
            args.upstream_repo,
            workdir,
            args.live_size,
            args.judge_model,
            args.judge_extra_params,
        )
        cases += live_cases
        summaries.append(live_summary)
    with (args.out / "upstream-vs-gym.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    normalized = {
        "method": "upstream code (pinned checkout) recomputes persona prompts, transcripts, messages, judge ratings from the same answers (mock-LLM replay) and the pooled score"
        + ("; plus a live stratified re-judge with upstream judge.py" if args.live_size else ""),
        "cases": sum(s["cases"] for s in summaries),
        "agreement": sum(s["agreement"] for s in summaries),
        "disagreements": sum(s["disagreements"] for s in summaries),
        "notes": [
            f"{s['control']}: {s['agreement']} of {s['cases']} cases agree"
            + (
                f"; per-dimension agreement {s['dimension_agreement_rate']:.1%}"
                if s.get("dimension_agreement_rate") is not None
                else ""
            )
            for s in summaries
        ],
        "details": {
            "controls": summaries,
            "case_ids": [c["case_id"] for c in cases if not c["agree"]][:20] or [c["case_id"] for c in cases[:3]],
        },
    }
    (args.out / "summary.json").write_text(
        json.dumps({"controls": summaries, "normalized": normalized}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
