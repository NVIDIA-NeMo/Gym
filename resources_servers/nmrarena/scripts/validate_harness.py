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

"""Model-free validation of the NMRArena verifier over the whole prepared benchmark.

No model is called. Every check drives the server's ``verify`` method with a
constructed response, so the extraction, canonicalisation and scoring path is the
one a rollout takes.

1. Gold sweep: each row's gold SMILES as the only candidate must score Top-1 = 1.
2. Negative controls over all rows: empty output, an invalid SMILES, the user prompt
   echoed back, a constant trivial molecule, the gold preceded by garbage, and the
   gold followed by garbage, the last two in lenient and strict mode.
3. Do-nothing read: every metric the aggregator would publish, for each control.
4. Upstream parity: upstream's published per-item lists for one model
   (``results/combined_predictions_105_final.json`` at the pinned commit) are
   re-scored through ``verify`` and compared with upstream's own formulas and with
   the README table; agreement is reported per item, not just in aggregate.
5. Boundary: a SMILES over the length cap is a status, not a crash.

Usage (from the repository root, with the server's venv active)::

    python resources_servers/nmrarena/scripts/validate_harness.py \
        --input resources_servers/nmrarena/data/nmrarena_105.jsonl \
        --upstream-model gemini --output /path/to/validation.json
"""

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import NMRArenaResourcesServer, NMRArenaResourcesServerConfig  # noqa: E402
from prepare_nmrarena import GITHUB_COMMIT, GITHUB_REPO  # noqa: E402
from scoring import canonical, tanimoto  # noqa: E402

from nemo_gym.openai_utils import NeMoGymResponse  # noqa: E402
from nemo_gym.reward_profile import compute_aggregate_metrics  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


PREDICTIONS_URL = (
    f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_COMMIT}/results/combined_predictions_105_final.json"
)

# The README table at the pinned commit, for the LLM rows (mean ± SD over three runs).
README_TABLE = {
    "gemini": {"top1": (27.6, 1.9), "top10": (32.4, 1.9), "tanimoto": (0.59, 0.02)},
    "gpt": {"top1": (14.9, 1.1), "top10": (21.3, 0.5), "tanimoto": (0.46, 0.01)},
    "grok": {"top1": (11.7, 1.5), "top10": (13.3, 1.6), "tanimoto": (0.43, 0.00)},
    "claude": {"top1": (9.2, 3.6), "top10": (14.6, 2.4), "tanimoto": (0.43, 0.02)},
    "deepseek": {"top1": (2.9, 1.6), "top10": (3.8, 1.6), "tanimoto": (0.37, 0.06)},
    "qwen": {"top1": (2.2, 0.5), "top10": (2.9, 1.0), "tanimoto": (0.89, 0.10)},
}


def make_server(**overrides) -> NMRArenaResourcesServer:
    config = NMRArenaResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="nmrarena", **overrides)
    return NMRArenaResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_validate",
        created_at=0.0,
        model="none",
        object="response",
        output=[
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def candidates_json(smiles: list[str]) -> str:
    return json.dumps({"candidates": [{"rank": i + 1, "smiles": s} for i, s in enumerate(smiles)]})


def verify_text(server: NMRArenaResourcesServer, row: dict, text: str) -> dict:
    from app import NMRArenaVerifyRequest

    request = NMRArenaVerifyRequest(
        responses_create_params=row["responses_create_params"],
        response=make_response(text),
        verifier_metadata=row["verifier_metadata"],
    )
    return asyncio.run(server.verify(request)).model_dump()


def aggregate(results: list[dict]) -> dict:
    """What the aggregator would publish for these verify responses (one rollout per task)."""
    keyed = [dict(r, _ng_task_index=i, _ng_rollout_index=0) for i, r in enumerate(results)]
    server = make_server()
    metrics = compute_aggregate_metrics(
        keyed, compute_metrics_fn=server.compute_metrics, get_key_metrics_fn=server.get_key_metrics
    )
    agent = metrics.agent_metrics
    return {
        "key_metrics": metrics.key_metrics,
        "mean/tanimoto_top1": agent.get("mean/tanimoto_top1"),
        "tanimoto_top1/answered_only": agent.get("tanimoto_top1/answered_only"),
        "count/answered": agent.get("count/answered"),
    }


def user_prompt(row: dict) -> str:
    return next(m["content"] for m in row["responses_create_params"]["input"] if m["role"] == "user")


def run_control(server, rows, name: str, text_for_row, expect_top1: float) -> dict:
    results = [verify_text(server, row, text_for_row(row)) for row in rows]
    top1 = [r["top1"] for r in results]
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "control": name,
        "n": len(rows),
        "top1_hits": int(sum(top1)),
        "top10_hits": int(sum(r["top10"] for r in results)),
        "expected_top1_per_row": expect_top1,
        "as_expected": all(t == expect_top1 for t in top1),
        "statuses": statuses,
        "aggregate": aggregate(results),
    }


def paired_upstream(server, rows, model_key: str, predictions: dict) -> dict:
    """Re-score upstream's published lists for ``model_key`` through ``verify``."""
    by_id = {rec["compound_id"]: rec for group in predictions.values() for rec in group.values()}
    per_run: list[dict[str, Any]] = []
    agreement = {"same_rank": 0, "ours_only_hit": 0, "upstream_only_hit": 0, "tanimoto_defined_differs": 0}
    for run in range(3):
        results = []
        up_hits1 = up_hits10 = 0
        up_tani = []
        for row in rows:
            rec = by_id[row["verifier_metadata"]["compound_id"]]
            cands = rec[model_key][run]
            # Upstream's formulas (analysis notebook) on the same list.
            truth = canonical(rec["smiles"])
            up_rank = next((i + 1 for i, s in enumerate(cands) if canonical(s) == truth), None)
            up_hits1 += up_rank == 1
            up_hits10 += up_rank is not None and up_rank <= 10
            t = tanimoto(rec["smiles"], cands[0]) if cands else None
            if t is not None:
                up_tani.append(t)
            ours = verify_text(server, row, candidates_json(cands) if cands else "no answer")
            results.append(ours)
            if ours["hit_rank"] == up_rank:
                agreement["same_rank"] += 1
            elif up_rank is None:
                agreement["ours_only_hit"] += 1
            elif ours["hit_rank"] is None:
                agreement["upstream_only_hit"] += 1
            else:
                agreement["same_rank"] -= 0  # both hit at different ranks: counted below
                agreement.setdefault("both_hit_different_rank", 0)
                agreement["both_hit_different_rank"] += 1
            if (ours["tanimoto_top1"] is None) != (t is None):
                agreement["tanimoto_defined_differs"] += 1
        our_tani = [r["tanimoto_top1"] for r in results if r["tanimoto_top1"] is not None]
        per_run.append(
            {
                "run": run + 1,
                "ours": {
                    "top1_pct": 100 * sum(r["top1"] for r in results) / len(rows),
                    "top10_pct": 100 * sum(r["top10"] for r in results) / len(rows),
                    "tanimoto": statistics.mean(our_tani) if our_tani else None,
                    "answered": len(our_tani),
                },
                "upstream_formula": {
                    "top1_pct": 100 * up_hits1 / len(rows),
                    "top10_pct": 100 * up_hits10 / len(rows),
                    "tanimoto": statistics.mean(up_tani) if up_tani else None,
                    "answered": len(up_tani),
                },
            }
        )

    def summarise(key: str, side: str):
        vals = [r[side][key] for r in per_run]
        return {"mean": statistics.mean(vals), "sd": statistics.stdev(vals)}

    return {
        "model": model_key,
        "n_items": len(rows),
        "n_lists": 3 * len(rows),
        "per_run": per_run,
        "three_run_summary": {
            side: {k: summarise(k, side) for k in ("top1_pct", "top10_pct", "tanimoto")}
            for side in ("ours", "upstream_formula")
        },
        "readme_table": README_TABLE.get(model_key),
        "paired_agreement": agreement,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Prepared benchmark JSONL")
    parser.add_argument("--upstream-model", default="gemini", choices=sorted(README_TABLE))
    parser.add_argument("--predictions", default=None, help="Local copy of combined_predictions_105_final.json")
    parser.add_argument("--output", required=True, help="Where to write the JSON summary")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    lenient = make_server()
    strict = make_server(strict_candidates=True)
    report: dict[str, Any] = {"n_rows": len(rows), "rdkit": __import__("rdkit").__version__}

    # 1. gold sweep
    gold = [verify_text(lenient, row, candidates_json([row["verifier_metadata"]["smiles"]])) for row in rows]
    report["gold_sweep"] = {
        "top1_hits": int(sum(r["top1"] for r in gold)),
        "n": len(rows),
        "all_scored": all(r["status"] == "scored" for r in gold),
        "tanimoto_all_one": all(r["tanimoto_top1"] == 1.0 for r in gold),
        "failures": [r["compound_id"] for r in gold if r["top1"] != 1.0],
    }

    # 2 + 3. negative controls and the do-nothing read
    garbage = "C(C)(C)(C)(C)C"  # pentavalent carbon: never parses
    controls = [
        ("empty_output", lambda row: "", 0.0, lenient),
        ("invalid_smiles_only", lambda row: candidates_json([garbage]), 0.0, lenient),
        ("echo_user_prompt", user_prompt, 0.0, lenient),
        ("constant_methane", lambda row: candidates_json(["C"]), 0.0, lenient),
        ("constant_benzene", lambda row: candidates_json(["c1ccccc1"]), 0.0, lenient),
        (
            "garbage_then_gold_lenient",
            lambda row: candidates_json([garbage, row["verifier_metadata"]["smiles"]]),
            0.0,
            lenient,
        ),
        (
            "garbage_then_gold_strict",
            lambda row: candidates_json([garbage, row["verifier_metadata"]["smiles"]]),
            0.0,
            strict,
        ),
        (
            "gold_then_garbage_lenient",
            lambda row: candidates_json([row["verifier_metadata"]["smiles"], garbage]),
            1.0,
            lenient,
        ),
        (
            "gold_then_garbage_strict",
            lambda row: candidates_json([row["verifier_metadata"]["smiles"], garbage]),
            0.0,
            strict,
        ),
        ("gold_twice_strict", lambda row: candidates_json([row["verifier_metadata"]["smiles"]] * 2), 0.0, strict),
        (
            "gold_at_rank_10_lenient",
            lambda row: candidates_json(["C"] * 9 + [row["verifier_metadata"]["smiles"]]),
            0.0,
            lenient,
        ),
        (
            "gold_at_rank_11_lenient",
            lambda row: candidates_json(["C"] * 10 + [row["verifier_metadata"]["smiles"]]),
            0.0,
            lenient,
        ),
        ("oversize_smiles", lambda row: candidates_json(["C" * 20000]), 0.0, lenient),
        (
            "truncated_json_gold_first",
            lambda row: (
                '```json\n{"candidates": [{"rank": 1, "smiles": %s}, {"rank": 2, "smi'
                % json.dumps(row["verifier_metadata"]["smiles"])
            ),
            1.0,
            lenient,
        ),
        (
            "truncated_json_no_salvage",
            lambda row: (
                '```json\n{"candidates": [{"rank": 1, "smiles": %s}, {"rank": 2, "smi'
                % json.dumps(row["verifier_metadata"]["smiles"])
            ),
            0.0,
            make_server(salvage_truncated_json=False),
        ),
    ]
    report["controls"] = [run_control(server, rows, name, fn, expect) for name, fn, expect, server in controls]

    # 4. upstream parity
    if args.predictions:
        raw = Path(args.predictions).read_bytes()
    else:
        with urllib.request.urlopen(PREDICTIONS_URL, timeout=120) as response:
            raw = response.read()
    report["upstream_predictions_sha256"] = hashlib.sha256(raw).hexdigest()
    report["upstream_parity"] = paired_upstream(lenient, rows, args.upstream_model, json.loads(raw))

    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "controls"}, indent=2)[:4000])
    for c in report["controls"]:
        print(
            f"{c['control']:32s} top1 {c['top1_hits']:3d}/{c['n']} top10 {c['top10_hits']:3d} ok={c['as_expected']} {c['statuses']} tani_cond={c['aggregate']['tanimoto_top1/answered_only']} answered={c['aggregate']['count/answered']}"
        )


if __name__ == "__main__":
    main()
