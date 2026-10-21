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
"""Model-free validation of the VeriSoftBench verifier against the running Lean container.

Runs every task through the *same* code path a model's output takes (``verify()`` of the
resources server, in process, with a fabricated response) and reports per-task verdicts.

Sweeps:

- ``gold``: the shipped ground-truth proof inside ``<lean4_proof>``; must verify.
- ``empty``: an empty assistant message.
- ``sorry``: ``<lean4_proof>by\\n  sorry</lean4_proof>``.
- ``wrong_theorem``: the ground-truth proof of a *different* task (the next id).
- ``unparseable``: text that is not Lean inside the proof tag.
- ``repair``: round-0 ``sorry`` then, as round 1, the gold proof inside
  ``<fixed_lean4_thm_proof>`` with the round-0 repair state; must verify on round 1.

Writes one JSON line per (sweep, task) with status, verified flag, Lean time and the
error message, then a summary with denominators. No model is involved.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import (  # noqa: E402
    VeriSoftBenchResourcesServer,
    VeriSoftBenchResourcesServerConfig,
    VeriSoftBenchVerifyRequest,
)
from upstream import UpstreamModules, no_rename_override, normalize_gold_body, strip_gold_contamination  # noqa: E402

from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


SWEEPS = ("gold", "empty", "sorry", "wrong_theorem", "unparseable", "repair")

# ``gold`` and ``repair`` can run in two forms:
# - ``contract``: the proof as a model would have to write it (``by ...``), through the
#   scoring path unchanged. Upstream's ``clean_leaked_identifiers`` renames references to
#   lemmas defined earlier in the same file (the prompt hides them; 433 of 500 reference
#   proofs use them), and prepends ``by`` to ``where``/``|`` proofs. So this form has a
#   ceiling set by upstream's protocol, and its result measures that ceiling.
# - ``oracle``: the same path with that one anti-leakage pass replaced by the identity, and
#   the proof body fed exactly as shipped minus its leading ``:=``. This is the check that
#   the assembly, compilation and verdict of the instrument accept every reference proof.
# - ``contract_norename``: the contract form with the anti-leakage rename replaced by the
#   identity. Separates what the protocol makes unsolvable by construction (a truncated
#   statement, a ``where``/``|`` proof) from what the rename costs.
GOLD_MODES = ("contract", "contract_norename", "oracle")

# Handle on the loaded upstream modules, set by ``main`` from the server's harness; all
# upstream interaction goes through the named adapters in ``upstream.py``.
UPSTREAM_MODULES: Optional[UpstreamModules] = None


def real_body(ground_truth_proof: str) -> tuple[str, bool]:
    """Upstream's Case 2a: drop a duplicated statement tail in front of the real proof.

    Returns (body starting at the real separator, was_missplit). Before ``main`` has
    loaded upstream, only the separator-free contamination strip applies.
    """
    if UPSTREAM_MODULES is not None:
        return normalize_gold_body(UPSTREAM_MODULES, ground_truth_proof)
    return strip_gold_contamination(ground_truth_proof), False


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_validation",
        created_at=0.0,
        model="model-free",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def gold_in_contract(ground_truth_proof: str) -> tuple[str, str]:
    """Express the shipped proof the way the output contract requires a model to write it.

    The dataset stores proofs as extracted from source: ``:= by ...`` (451 of 500),
    ``:= <term>`` (46), ``where ...`` (2) or ``| ...`` (1). The model path appends ``:=`` to
    the statement and asks for a body "starting with by", and upstream's
    ``clean_leaked_identifiers`` prepends ``by`` to a body that does not start with it.
    So: drop the leading ``:=``; wrap a bare term as ``by exact (<term>)``; leave
    ``where``/``|`` proofs alone (upstream keeps the separator on the proof side for
    them). Returns (text, class) so the sweep can report denominators per class.
    """
    g, missplit = real_body(ground_truth_proof)
    tag = "_missplit" if missplit else ""
    if g.startswith(":="):
        body = g[2:].lstrip()
        if body.startswith("by"):
            return body, "tactic" + tag
        # Column-0 continuation lines would end the tactic block; indent the whole term.
        indented = "\n".join(("    " + line) if line.strip() else line for line in body.splitlines())
        return "by\n  exact (\n" + indented + "\n  )", "term" + tag
    if g.startswith("where"):
        return g, "where" + tag
    if g.startswith("|"):
        return g, "match" + tag
    return g, "other" + tag


def gold_text(ground_truth_proof: str, mode: str) -> tuple[str, str]:
    if mode in ("contract", "contract_norename"):
        return gold_in_contract(ground_truth_proof)
    g = ground_truth_proof.strip()
    _, cls = gold_in_contract(ground_truth_proof)
    return (g[2:].lstrip() if g.startswith(":=") else g), cls


def candidate_text(
    sweep: str, row: dict[str, Any], rows_by_id: dict[int, dict[str, Any]], mode: str = "contract"
) -> str:
    meta = row["verifier_metadata"]
    if sweep == "gold":
        text, _ = gold_text(meta["ground_truth_proof"], mode)
        return f"<lean4_proof>\n{text}\n</lean4_proof>"
    if sweep == "empty":
        return ""
    if sweep in ("sorry", "repair"):
        return "<lean4_proof>\nby\n  sorry\n</lean4_proof>"
    if sweep == "wrong_theorem":
        other = rows_by_id.get(meta["id"] + 1) or rows_by_id[1]
        text, _ = gold_text(other["verifier_metadata"]["ground_truth_proof"], mode)
        return f"<lean4_proof>\n{text}\n</lean4_proof>"
    if sweep == "unparseable":
        return "<lean4_proof>\nby ⟨⟨⟨ this is not Lean %%% ]]]\n</lean4_proof>"
    raise ValueError(sweep)


async def run_sweep(
    server: VeriSoftBenchResourcesServer,
    sweep: str,
    rows: list[dict[str, Any]],
    concurrency: int,
    out,
    mode: str,
    all_rows: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    # The wrong-theorem control borrows the next task's proof, which may be outside --ids.
    rows_by_id = {r["verifier_metadata"]["id"]: r for r in (all_rows or rows)}
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def one(row: dict[str, Any]) -> None:
        meta = row["verifier_metadata"]
        async with sem:
            t0 = time.time()
            req = VeriSoftBenchVerifyRequest(
                responses_create_params=row["responses_create_params"],
                verifier_metadata=meta,
                response=make_response(candidate_text(sweep, row, rows_by_id, mode)),
            )
            resp = await server.verify(req)
            record = {
                "sweep": sweep,
                "gold_mode": mode,
                "id": meta["id"],
                "lean_root": meta["lean_root"],
                "thm_name": meta["thm_name"],
                "round": 0,
                "status": resp.status,
                "verified": resp.verified,
                "reward": resp.reward,
                "harness_failure": resp.harness_failure,
                "lean_time_s": round(resp.lean_time_s, 2),
                "err_msg": (resp.err_msg or "")[:2000],
                "wall_s": round(time.time() - t0, 2),
            }
            record["gold_class"] = gold_in_contract(meta["ground_truth_proof"])[1]
            if sweep == "repair" and resp.needs_repair:
                fixed_text, _ = gold_text(meta["ground_truth_proof"], mode)
                fixed = f"<fixed_lean4_thm_proof>\n{fixed_text}\n</fixed_lean4_thm_proof>"
                req2 = VeriSoftBenchVerifyRequest(
                    responses_create_params=row["responses_create_params"],
                    verifier_metadata=meta,
                    response=make_response(fixed),
                    repair_round=1,
                    repair_state=resp.repair_state,
                )
                resp2 = await server.verify(req2)
                record.update(
                    round=1,
                    round0_status=resp.status,
                    repair_prompt_chars=len(resp.repair_prompt or ""),
                    status=resp2.status,
                    verified=resp2.verified,
                    reward=resp2.reward,
                    harness_failure=resp2.harness_failure,
                    lean_time_s=round(resp.lean_time_s + resp2.lean_time_s, 2),
                    err_msg=(resp2.err_msg or "")[:2000],
                    wall_s=round(time.time() - t0, 2),
                )
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            flag = "PASS" if record["verified"] else "fail"
            print(
                f"[{sweep}] {len(results)}/{len(rows)} {flag} {record['status']} {meta['lean_root']}/{meta['thm_name']} ({record['lean_time_s']}s)",
                flush=True,
            )

    await asyncio.gather(*(one(r) for r in rows))
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for sweep in SWEEPS:
        rs = [r for r in results if r["sweep"] == sweep]
        if not rs:
            continue
        statuses: dict[str, int] = {}
        for r in rs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        by_class: dict[str, list[int]] = {}
        for r in rs:
            by_class.setdefault(r.get("gold_class", "?"), []).append(int(r["verified"]))
        summary[sweep] = {
            "n": len(rs),
            "verified": sum(r["verified"] for r in rs),
            "verified_by_gold_class": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_class.items())},
            "harness_failure": sum(r["harness_failure"] >= 1.0 for r in rs),
            "statuses": statuses,
            "lean_time_s_total": round(sum(r["lean_time_s"] for r in rs), 1),
            "lean_time_s_max": round(max(r["lean_time_s"] for r in rs), 1),
        }
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("resources_servers/verisoftbench/data/verisoftbench_full.jsonl"))
    p.add_argument("--upstream-dir", type=Path, default=Path("cache/verisoftbench/VeriSoftBench-d066523"))
    p.add_argument("--container", default="verisoftbench-lean")
    p.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=SWEEPS)
    p.add_argument("--ids", nargs="*", type=int, default=None, help="restrict to these task ids")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--gold-mode", choices=GOLD_MODES, default="contract", help="see GOLD_MODES")
    p.add_argument("--output", type=Path, required=True, help="JSONL of per-task records; summary written next to it")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    all_rows = rows
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r["verifier_metadata"]["id"] in wanted]
    from unittest.mock import MagicMock

    config = VeriSoftBenchResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="",
        name="verisoftbench",
        upstream_dir=str(args.upstream_dir),
        docker_container=args.container,
        max_concurrent_compilations=args.concurrency,
    )
    server = VeriSoftBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    global UPSTREAM_MODULES
    UPSTREAM_MODULES = server._harness.modules
    # The oracle/no-rename forms weaken shared upstream state, so they run inside the
    # scoped override: it is restored even when a sweep raises.
    no_rename = args.gold_mode in ("oracle", "contract_norename")
    override = no_rename_override(UPSTREAM_MODULES) if no_rename else nullcontext()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    with override, open(args.output, "a", encoding="utf-8") as out:
        for sweep in args.sweeps:
            all_results.extend(
                asyncio.run(run_sweep(server, sweep, rows, args.concurrency, out, args.gold_mode, all_rows=all_rows))
            )
    summary = summarize(all_results)
    summary["input_sha256"] = hashlib.sha256(open(args.input, "rb").read()).hexdigest()
    summary["gold_mode"] = args.gold_mode
    summary["n_tasks"] = len(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
