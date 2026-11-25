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
"""Offline pair judging of labeled groups (development; same runner as the training endpoint will use).

    python -m resources_servers.turn_judge.offline --refs judge_refs.json --split train --tasks tasks.jsonl
        --labeling LABELING_annotator.md --endpoint-env qwen38x4_endpoint.env --out <dir>
        [--groups g1,g2] [--concurrency 48] [--seed 0] [--ill-frac 0.5] [--limit-pairs N]

Inputs per group: the turn_judge group workspace (TASK.md, answer_key/, PREPASS.md, group.json, R<k>/..., rollouts.jsonl)
read ONCE into memory; the task row (image). Outputs (append-only, few files, light on shared storage):
<out>/pairs.jsonl (one row per pair: status, timings, pair.json text, notes), <out>/exports.jsonl.gz (OpenCode session
exports), <out>/runs/<group>/out/labels.json + result.json (merged; tools/eval_judge.py reads these). Re-running skips
pairs already judged ok. Run inside the training image (Gym venv python; OpenSandbox env vars set)."""

import argparse
import asyncio
import dataclasses
import gzip
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from resources_servers.turn_judge.merge import merge_group, parse_json
from resources_servers.turn_judge.pairing import make_pairs
from resources_servers.turn_judge.prompt import (
    GROUP_INSTRUCTIONS,
    GROUP_INSTRUCTIONS_C,
    GROUP_INSTRUCTIONS_D,
    GROUP_INSTRUCTIONS_E,
    build_instructions,
)
from resources_servers.turn_judge.runner import PairRunSettings, run_pair

IMAGE_KEYS = ("image_name", "image_ref", "image_url", "docker_image")
HDR = re.compile(r"^===== #(\d+) \[call (\d+)\] (\S+) =====$")


def task_image(row: dict) -> str:
    for k in IMAGE_KEYS:
        if row.get(k):
            return row[k]
    raise KeyError(f"no image field in task row {row.get('instance_id')}")


def calls_index(rk: str, trajectory: str) -> str:
    lines = trajectory.split("\n")
    calls, cur = [], None
    for i, ln in enumerate(lines):
        m = HDR.match(ln)
        if not m:
            continue
        idx, c, typ = int(m.group(1)), int(m.group(2)), m.group(3)
        if cur is None or cur["call"] != c:
            cur = {"call": c, "first": idx, "last": idx, "line": i + 1, "reason": "", "tools": []}
            calls.append(cur)
        cur["last"] = idx
        body = next((lines[j] for j in range(i + 1, min(i + 6, len(lines))) if lines[j].strip()), "")
        if typ == "reasoning" and not cur["reason"]:
            cur["reason"] = body.strip()[:160]
        elif typ == "function_call":
            cur["tools"].append(body.strip()[:110])
        elif typ == "message":
            cur["tools"].append("msg: " + body.strip()[:80])
    out = [f"{rk}: {len(calls)} model calls; L<n> = line in trajectory.txt"]
    for c in calls:
        tools = " ; ".join(c["tools"][:4]) + (f" (+{len(c['tools']) - 4})" if len(c["tools"]) > 4 else "")
        out.append(f"call {c['call']}: #{c['first']}-#{c['last']} L{c['line']} | {c['reason']} | {tools}")
    return "\n".join(out) + "\n"


def load_group(ws: Path) -> dict:
    """Read the group workspace once: files for upload (bytes), rollout outputs for the checks."""
    files, rollouts = {}, {}
    for root, _, names in os.walk(ws):
        for n in names:
            p = Path(root) / n
            rel = p.relative_to(ws).as_posix()
            if rel == "rollouts.jsonl":
                continue
            files[rel] = p.read_bytes()
    with open(ws / "rollouts.jsonl") as f:
        for i, ln in enumerate(f):
            rollouts[i] = json.loads(ln)["response"]["output"]
    meta = json.loads(files["group.json"])
    for r in meta["rollouts"]:
        tr = files.get(f"R{r['R']}/trajectory.txt")
        if tr is not None:
            files[f"R{r['R']}/calls_index.txt"] = calls_index(f"R{r['R']}", tr.decode(errors="replace")).encode()
    return {"files": files, "outputs": rollouts, "meta": meta}


CHECKER = (Path(__file__).parent / "checks.py").read_bytes()
DECISION_LINES = (Path(__file__).parent / "decision_lines.py").read_bytes()
FIRST_MENTION = (Path(__file__).parent / "first_mention.py").read_bytes()
HEDGE_STRONG = re.compile(
    r"\b(?:50/50|fifty[- ]fifty|coin[- ]?flip|unknowable|no way (?:for me )?to (?:know|tell)|"
    r"can(?:not|'t|’t) (?:know|tell|determine) (?:which|what|whether|for sure|exactly)|"
    r"impossible to (?:know|tell|determine)|(?:best|educated|wild) guess|(?:have|need) to guess|"
    r"i(?:'ll| will) guess|guessing)\b",
    re.I,
)
HEDGE_WEAK = re.compile(
    r"\b(?:ambiguous|ambiguity|under-?specified|not specified|unspecified|does(?:n't| not) specify|"
    r"unclear (?:which|whether|what|how))\b",
    re.I,
)


def _agent_text(item: dict) -> str:
    parts = []
    for key in ("summary", "content"):
        v = item.get(key)
        if isinstance(v, list):
            parts += [str(x.get("text") or "") for x in v if isinstance(x, dict)]
        elif isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)


def group_hedges(group: dict) -> str:
    """hedges(group), computed once per loaded group (pair_files / group_files reuse it)."""
    if "hedges" not in group:
        group["hedges"] = hedges(group)
    return group["hedges"]


def hedges(group: dict, strong_max: int = 8, weak_max: int = 6) -> str:
    """Per rollout, the agent's own statements that it is guessing or that something is unknowable / ambiguous
    (reasoning and messages; script grep, LOG 065): evidence for the ill-posed check (eclipse: "50/50 guess")."""
    out = [
        "# Hedges: where an agent said it was guessing or that something is unknowable / ambiguous",
        "(script grep over reasoning and messages; ±150 characters; check the hits that concern a failing check)",
        "",
    ]
    for r in group["meta"]["rollouts"]:
        k, items = r["R"], group["outputs"][r["line"]]
        starts = [i for i, it in enumerate(items) if i == 0 or it.get("type") == "reasoning"]
        found = {"strong": [], "weak": []}
        for i, it in enumerate(items):
            if it.get("type") not in ("reasoning", "message"):
                continue
            text = _agent_text(it)
            for kind, rx in (("strong", HEDGE_STRONG), ("weak", HEDGE_WEAK)):
                for m in rx.finditer(text):
                    ctx = re.sub(r"\s+", " ", text[max(0, m.start() - 150) : m.end() + 150]).strip()
                    found[kind].append(f'- #{i} [call {sum(1 for s_ in starts if s_ <= i) - 1}] {kind}: "{ctx}"')
        n = len(found["strong"]) + len(found["weak"])
        out.append(
            f"## R{k} ({'PASS' if r['reward'] > 0.5 else 'FAIL'}): {n} hits"
            + (f" (showing {strong_max} strong + {weak_max} weak)" if n > strong_max + weak_max else "")
        )
        out += found["strong"][:strong_max] + found["weak"][:weak_max] + [""]
    return "\n".join(out)


GROUP_FACTS = ("tests.json", "failing_assertions.txt", "final.patch")


def test_matrix(group: dict) -> str:
    """Which rollout fails which hidden test (from each R<k>/tests.json)."""
    rows = ["| rollout | outcome | failing hidden tests |", "|---|---|---|"]
    for r in group["meta"]["rollouts"]:
        k = r["R"]
        try:
            st = json.loads(group["files"].get(f"R{k}/tests.json", b"{}"))
        except Exception:
            st = {}
        bad = sorted(t for t, v in st.items() if str(v).upper() not in ("PASSED", "PASS"))
        rows.append(f"| R{k} | {'PASS' if r['reward'] > 0.5 else 'FAIL'} | {', '.join(bad) or '-'} |")
    return "\n".join(rows) + "\n"


def pair_files(
    group: dict, fail: int, passing: int, labeling: bytes, variant: str = "A", decisions: str | None = None
) -> dict:
    keep = {"TASK.md", "PREPASS.md", "group.json"}
    out = {"LABELING.md": labeling, "bin/check_pair.py": CHECKER}
    if variant.upper() in ("C", "D", "E"):
        out["bin/decision_lines.py"] = DECISION_LINES
        if variant.upper() in ("D", "E"):
            out["bin/first_mention.py"] = FIRST_MENTION
        if variant.upper() == "E":
            out["BUT_FOR_REQUIRED"] = "every harmful label needs a non-empty but_for (variant E)\n"
        out["ws/HEDGES.md"] = group_hedges(group)
        out["decisions.json"] = decisions or ""
        out["DECISIONS_LIST"] = "the pair must address every decision of /judge/decisions.json (variant C)\n"
    for rel, data in group["files"].items():
        top = rel.split("/", 1)[0]
        if rel in keep or top == "answer_key" or top in (f"R{fail}", f"R{passing}"):
            out[f"ws/{rel}"] = data
        elif variant.upper() == "BP" and top.startswith("R") and rel.split("/", 1)[-1] in GROUP_FACTS:
            out[f"ws/group/{top}.{rel.split('/', 1)[1]}"] = data
    if variant.upper() == "BP":
        out["ws/group/MATRIX.md"] = test_matrix(group)
        out["DECISIONS_REQUIRED"] = "out/decisions.json is required (variant Bp)\n"
    if variant.upper() == "B" and decisions:
        out["decisions.json"] = decisions
    return out


def group_files(group: dict, labeling: bytes, variant: str = "B") -> dict:
    """Stage-1 workspace. B: everything except the trajectories. C: also the trajectories (for search), HEDGES.md and
    the checker (decisions.json format)."""
    c = variant.upper() in ("C", "D", "E")
    out = {"LABELING.md": labeling}
    for rel, data in group["files"].items():
        if c or not rel.endswith("/trajectory.txt"):
            out[f"ws/{rel}"] = data
    out["ws/MATRIX.md"] = test_matrix(group)
    if c:
        out["ws/HEDGES.md"] = group_hedges(group)
        out["bin/check_pair.py"] = CHECKER
        if variant.upper() == "E":
            out["DECISIONS_STRICT"] = "check_pair.py --decisions runs the membership checks (variant E)\n"
    return out


def load_endpoint(path: str) -> tuple[str, str]:
    env = dict(re.findall(r"export (\w+)=(\S+)", Path(path).read_text()))
    return env["JUDGE_BASE_URL"], env["JUDGE_MODEL_NAME"]


# Stage-1 (group analyst) instructions per variant.
STAGE1_INSTRUCTIONS = {
    "B": GROUP_INSTRUCTIONS,
    "C": GROUP_INSTRUCTIONS_C,
    "D": GROUP_INSTRUCTIONS_D,
    "E": GROUP_INSTRUCTIONS_E,
}


def sandbox_provider() -> tuple[dict, dict]:
    """(provider config, default sandbox metadata) from Gym's OpenSandbox config (connection from OPENSANDBOX_* env)."""
    gym_root = Path(__file__).resolve().parents[2]
    ocfg = OmegaConf.to_container(
        OmegaConf.load(gym_root / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"), resolve=True
    )["sandbox"]
    return {"opensandbox": ocfg["opensandbox"]}, dict(ocfg.get("default_metadata", {}))


async def main_async(a: argparse.Namespace) -> None:
    refs = json.loads(Path(a.refs).read_text())
    rows = {json.loads(ln)["instance_id"]: json.loads(ln) for ln in open(a.tasks)}
    groups = [
        g for g, v in sorted(refs.items()) if v["split"] == a.split and (not a.groups or g in a.groups.split(","))
    ]
    base_url, model = load_endpoint(a.endpoint_env)
    provider, default_metadata = sandbox_provider()
    settings = PairRunSettings(
        judge_base_url=base_url, judge_model=model, metadata={**default_metadata, "role": "turn-judge"}
    )
    settings.opencode.temperature = a.temperature
    settings.session_timeout_s = a.session_timeout
    stage1_settings = dataclasses.replace(settings, session_timeout_s=a.stage1_timeout)
    template = Path(a.prompt_file).read_text() if a.prompt_file else None
    labeling = Path(a.labeling).read_bytes()
    out = Path(a.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    done = {}
    if (out / "pairs.jsonl").exists():
        for ln in open(out / "pairs.jsonl"):
            r = json.loads(ln)
            if r.get("status") == "ok" or a.merge_only:
                done[(r["group"], r["fail"], r["passing"], r.get("sample", 0))] = r
    sem = asyncio.Semaphore(a.concurrency)
    plog, xlog = open(out / "pairs.jsonl", "a"), gzip.open(out / "exports.jsonl.gz", "at")
    lock = asyncio.Lock()
    print(f"{len(groups)} groups, endpoint {base_url} ({model}), concurrency {a.concurrency}", flush=True)

    async def one_pair(g, group, pr, image, sample=0, decisions=None, variant=None):
        key = (g, pr.fail, pr.passing, sample)
        if key in done:
            return done[key]
        if a.merge_only:  # never judged: merge without it
            return {"status": "missing"}
        variant = variant or a.variant
        instr = build_instructions(
            pr.fail,
            group["meta"]["rollouts"][pr.fail]["line"],
            pr.passing,
            group["meta"]["rollouts"][pr.passing]["line"],
            variant,
            template,
        )
        async with sem:
            res = await run_pair(
                provider, image, pair_files(group, pr.fail, pr.passing, labeling, variant, decisions), instr, settings
            )
        row = {
            "group": g,
            "fail": pr.fail,
            "passing": pr.passing,
            "sample": sample,
            "larger_side": pr.larger_side,
            **{k: v for k, v in res.items() if k != "export"},
            "time": time.strftime("%H:%M:%S"),
        }
        async with lock:
            plog.write(json.dumps(row) + "\n")
            plog.flush()
            xlog.write(
                json.dumps(
                    {"group": g, "fail": pr.fail, "passing": pr.passing, "sample": sample, "export": res.get("export")}
                )
                + "\n"
            )
            xlog.flush()
        print(
            f"{row['time']} {g} R{pr.fail}/R{pr.passing}: {res['status']} {res['timings']}"
            f"{' ' + res['error'][:160] if res.get('error') else ''}",
            flush=True,
        )
        return row

    async def one_group(g):
        ws = Path(a.ws_root) / Path(refs[g]["ws"]).name if a.ws_root else Path(refs[g]["ws"])
        group = load_group(ws)
        meta = group["meta"]
        row = rows[meta["task"]]
        pairs = make_pairs([r["reward"] for r in meta["rollouts"]], f"{a.seed}:{g}")
        if a.limit_pairs:
            pairs = pairs[: a.limit_pairs]
        t0 = time.time()
        decisions = None
        pair_variant = a.variant
        if a.variant.upper() in ("B", "C", "D", "E") and pairs:
            c = a.variant.upper() in ("C", "D", "E")
            group_instructions = STAGE1_INSTRUCTIONS[a.variant.upper()]

            async def stage1(j):
                gkey = (g, -1, -1, j)
                if gkey in done:
                    return done[gkey].get("pair_json")
                if a.merge_only:
                    return None
                text = None
                # C: longer cap and one retry (round 2: 3 stage-1 timeouts left their pairs without a list, LOG 065)
                for attempt in range(2 if c else 1):
                    async with sem:
                        gres = await run_pair(
                            provider,
                            task_image(row),
                            group_files(group, labeling, a.variant),
                            group_instructions,
                            stage1_settings if c else settings,
                            required="out/decisions.json",
                            extra_reads=(),
                        )
                    text = gres.get("pair_json")
                    grow = {
                        "group": g,
                        "fail": -1,
                        "passing": -1,
                        "sample": j,
                        "larger_side": "group-stage",
                        "attempt": attempt,
                        **{k: v for k, v in gres.items() if k != "export"},
                        "time": time.strftime("%H:%M:%S"),
                    }
                    async with lock:
                        plog.write(json.dumps(grow) + "\n")
                        plog.flush()
                        xlog.write(
                            json.dumps(
                                {
                                    "group": g,
                                    "fail": -1,
                                    "passing": -1,
                                    "sample": j,
                                    "attempt": attempt,
                                    "export": gres.get("export"),
                                }
                            )
                            + "\n"
                        )
                        xlog.flush()
                    print(
                        f"{grow['time']} {g} STAGE1 sample {j} (attempt {attempt}): {gres['status']} "
                        f"{gres['timings']}",
                        flush=True,
                    )
                    if gres["status"] == "ok" and (parse_json(text) or {}).get("decisions") is not None:
                        break
                return text

            # --stage1-samples k: the group's ill_posed verdict is the majority of k analysts (a tie is labelable);
            # the decision list comes from the first analyst on the majority side (LOG 066)
            texts = await asyncio.gather(*(stage1(j) for j in range(a.stage1_samples if c else 1)))
            parsed = [(t, parse_json(t)) for t in texts if parse_json(t)]
            ill_votes = sum(1 for _, d in parsed if d.get("group_ill_posed") is True)
            maj_ill = 2 * ill_votes > len(parsed)
            decisions = next((t for t, d in parsed if (d.get("group_ill_posed") is True) == maj_ill), None)
            if len(parsed) > 1:
                print(f"{g}: stage-1 ill_posed votes {ill_votes}/{len(parsed)}", flush=True)
            if (
                c
                and not (parse_json(decisions) or {}).get("decisions")
                and not (parse_json(decisions) or {}).get("group_ill_posed")
            ):
                pair_variant, decisions = "Bp", None  # no list: each pair writes its own from the group facts
                print(f"{g}: no stage-1 decision list, pairs fall back to Bp", flush=True)
            if c and (parse_json(decisions) or {}).get("group_ill_posed") is True:
                pairs = []  # the group analysts ruled the group ill_posed: no labels, so no pair sessions
                print(f"{g}: stage 1 says ill_posed, no pairs", flush=True)
        if a.stage1_only:
            pairs = []
        jobs = [(pr, k) for pr in pairs for k in range(a.samples)]
        results = await asyncio.gather(
            *(one_pair(g, group, pr, task_image(row), k, decisions, pair_variant) for pr, k in jobs)
        )
        merged = merge_group(
            meta,
            [
                {
                    "fail": pr.fail,
                    "passing": pr.passing,
                    "larger_side": pr.larger_side,
                    "verdict": parse_json(r.get("pair_json")),
                }
                for (pr, _), r in zip(jobs, results)
            ],
            group["outputs"],
            a.ill_frac,
            a.ill_min_votes,
            a.vote_frac,
            (parse_json(decisions) or {}).get("group_ill_posed") is True if decisions else None,
            decision_ids=bool((parse_json(decisions) or {}).get("decisions")),
            good_earliest=a.good_earliest,
        )
        gd = out / "runs" / g / "out"
        gd.mkdir(parents=True, exist_ok=True)
        (gd / "labels.json").write_text(json.dumps(merged, indent=1))
        res = {
            "status": "ok"
            if pairs and all(r.get("status") == "ok" for r in results)
            else (
                "ill_posed (stage 1)"
                if decisions and (parse_json(decisions) or {}).get("group_ill_posed") is True
                else "abstain"
                if not pairs
                else "partial"
            ),
            "pairs": len(pairs),
            "pairs_failed": sum(1 for r in results if r.get("status") != "ok"),
            "ill_votes": merged["ill_votes"],
            "wall_min": round((time.time() - t0) / 60, 2),
            "turns": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        (out / "runs" / g / "result.json").write_text(json.dumps(res, indent=1))
        print(f"GROUP {g}: {res}", flush=True)

    # SIGINT / SIGTERM: cancel the pair tasks inside the running loop so every runner's finally (sandbox stop) runs
    # before the loop closes (a plain KeyboardInterrupt leaked all in-flight sandboxes until their TTL, LOG 060).
    loop = asyncio.get_running_loop()
    tasks = [asyncio.ensure_future(one_group(g)) for g in groups]
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: [t.cancel() for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)
    plog.close()
    xlog.close()
    n_cancel = sum(isinstance(r, asyncio.CancelledError) for r in results)
    if n_cancel:
        print(f"interrupted: {n_cancel} groups cancelled; their sandboxes were stopped", flush=True)


def main() -> None:
    sys.argv, argv = sys.argv[:1], sys.argv[1:]  # Gym parses sys.argv as Hydra overrides on first use
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--labeling", required=True)
    ap.add_argument("--endpoint-env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default="")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--ill-frac", type=float, default=0.0)
    ap.add_argument("--ill-min-votes", type=int, default=2)
    ap.add_argument("--merge-only", action="store_true", help="re-merge saved pair results; start no sandbox")
    ap.add_argument("--prompt-file", default="", help="alternative instructions template (judge variant)")
    ap.add_argument("--samples", type=int, default=1, help="independent judgments per pair")
    ap.add_argument("--vote-frac", type=float, default=0.5, help="keep a label given by >= this share of judgments")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument(
        "--variant",
        default="A",
        help="A | B (group decision list first) | Bp (group facts in each pair) | C (B + LOG 065 fixes) | D (C + LOG 067 fixes) | E (D + LOG 072 fixes)",
    )
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--stage1-timeout", type=float, default=3600.0, help="variant C group-stage session cap (s)")
    ap.add_argument("--session-timeout", type=float, default=2400.0, help="pair session cap (s)")
    ap.add_argument("--stage1-samples", type=int, default=1, help="variant C: group analysts; ill_posed by majority")
    ap.add_argument("--stage1-only", action="store_true", help="run the group stage only (no pair sessions)")
    ap.add_argument("--good-earliest", action="store_true", help="merge: one good per decision id, earliest call")
    ap.add_argument("--ws-root", default="", help="node-local copy of the group workspaces (<root>/<ws basename>)")
    asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
