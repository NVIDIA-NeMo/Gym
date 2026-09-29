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

"""Score Gym rollouts a second time with upstream's own harness and compare, row by row.

``fine_eval.py`` re-derives upstream's rules rather than copying them, and the
unit tests check that re-derivation against fixtures. This script checks it
against the real thing: it downloads CombiBench at the pinned revision, imports
``evaluation/verifier/one_stage_verify.py`` unmodified, and runs it on exactly
the model outputs a Gym run already scored, through the same Lean server.

Paired per-item agreement is the evidence that matters. A headline that matches
by coincidence -- the same count of successes on different problems -- is not a
reproduction, so the report records every disagreement with the status Gym
assigned and the error type upstream assigned.

Three sources of disagreement are expected by construction, all named in the
server README. Two run one way (Gym accepts, upstream rejects): the answer check
ascribes the abbreviation's declared type, and the statement check ignores
trailing whitespace. The third runs the other way (Gym rejects, upstream
accepts): a leftover ``sorry`` reported only in the REPL's ``sorries`` list is
failed here, and upstream's ``is_error`` never reads that field. Only the first
two are configurable; to measure agreement with them removed, rescore the same
rollouts through ``gym eval reverify`` with ``answer_check_ascription: false``
and ``normalize_trailing_whitespace: false``, and pass that file as
``--rescore-with``.

A ``sandbox_error`` row is not a verdict: this verifier reached no scoring
decision on it. Such rows still appear in the counters below, where upstream's
"not a success" happens to line up with Gym's 0.0 reward; read them out of
``gym_status_counts`` rather than as agreements.

Needs the dependencies upstream's modules import that Gym does not ship:

    uv pip install loguru strenum tenacity tqdm

    python resources_servers/combibench/scripts/upstream_agreement.py \
        --rollouts results/combibench/rollouts.jsonl \
        --output resources_servers/combibench/data/upstream_agreement.json \
        --lean-server-url http://127.0.0.1:12332
"""

import argparse
import json
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

# _text_of is deliberately the server's own reader: if it and this script
# disagreed about which part of the response is the model's answer, the
# comparison below would be measuring that instead of the scoring rules.
from resources_servers.combibench.app import CombibenchVerifyRequest, _text_of


GITHUB_REVISION = "c67e4213597b1477351d9ef5ca37fb622084cc78"  # pragma: allowlist secret
TARBALL_URL = f"https://codeload.github.com/MoonshotAI/CombiBench/tar.gz/{GITHUB_REVISION}"
DOWNLOAD_TIMEOUT_SECONDS = 120


def fetch_upstream(cache_dir: Path) -> Path:
    """Unpack the pinned CombiBench tree and return its root (the import root of ``evaluation``)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    root = cache_dir / f"CombiBench-{GITHUB_REVISION}"
    if root.is_dir():
        return root
    print(f"Downloading {TARBALL_URL}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with urllib.request.urlopen(TARBALL_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            tmp.write(response.read())
        tmp.flush()
        with tarfile.open(tmp.name) as tar:
            tar.extractall(cache_dir, filter="data")
    if not (root / "evaluation").is_dir():
        raise SystemExit(f"Expected {root}/evaluation in the downloaded archive")
    return root


def import_upstream(root: Path):
    """Import upstream's verifier from the unpacked tree, unmodified."""
    sys.path.insert(0, str(root))
    try:
        from evaluation.client.lean_client import Lean4Client
        from evaluation.verifier.one_stage_verify import one_stage_verify
    except ModuleNotFoundError as exc:
        if exc.name in {"loguru", "strenum", "tenacity", "tqdm"}:
            raise SystemExit(
                f"upstream's modules import {exc.name}; install it first: uv pip install loguru strenum tenacity tqdm"
            ) from exc
        raise
    return Lean4Client, one_stage_verify


def compat_client(lean4_client_cls, url: str, api_key: Optional[str]):
    """Upstream's client, with the one transport-level fix it needs to run at all.

    ``evaluation/client/lean_client.py::verify`` reads ``res["error"]`` by
    subscript. Kimina at the pinned commit omits that key entirely when there
    was no error, so the read raises ``KeyError``, upstream's blanket
    ``except Exception`` swallows it, and *every* submission that reaches the
    compile stage is reported as a failed proof -- including proofs that
    compile cleanly. Upstream was written against an older server that always
    sent ``"error": null``.

    Filling the absent key restores the reply shape upstream expects. This
    touches the transport only: no scoring rule, regex, threshold or verdict of
    upstream's is modified, and a genuine error still arrives as one. Without
    it the comparison would measure that incompatibility rather than whether
    the two harnesses agree. (This server's own client reads the field with
    ``.get``, which is why it is unaffected.)
    """

    class ErrorKeyCompatClient(lean4_client_cls):
        def verify(self, codes, timeout, infotree_type=None):
            body = super().verify(codes, timeout, infotree_type)
            for result in (body or {}).get("results", []):
                if isinstance(result, dict):
                    result.setdefault("error", None)
            return body

    return ErrorKeyCompatClient(url, api_key=api_key)


def model_text(row: dict) -> str:
    """The assistant text, read the way the server reads it, so both harnesses see one string."""
    return _text_of(CombibenchVerifyRequest.model_validate(row))


def row_key(row: dict, index: int) -> str:
    name = row.get("theorem_name") or f"row_{index}"
    return f"{name}#{row.get('_ng_rollout_index', 0)}"


def index_by_key(pairs: list[tuple[str, dict]], what: str) -> dict[str, dict]:
    """Key the records, failing closed when two of them collide.

    ``_ng_rollout_index`` defaults to 0, so a rollouts file written without that
    field gives all 16 repeats of a problem the same key. Keeping the last
    silently would turn 1600 paired comparisons into 100 while still reporting a
    complete run, and would mis-pair every ``--rescore-with`` verdict. Neither is
    a result anyone could tell from a real one, so this refuses instead.
    """
    keyed: dict[str, dict] = {}
    collisions: list[str] = []
    for key, record in pairs:
        if key in keyed:
            collisions.append(key)
        keyed[key] = record
    if collisions:
        first = sorted(set(collisions))[:5]
        raise SystemExit(
            f"{len(pairs)} {what} collapsed to {len(keyed)} keys: {len(set(collisions))} duplicated, "
            f"e.g. {', '.join(first)}. Rollouts must carry a distinct theorem_name/_ng_rollout_index pair; "
            "a file written without _ng_rollout_index cannot be compared row by row."
        )
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired agreement between this verifier and upstream's")
    parser.add_argument("--rollouts", type=Path, required=True, help="rollouts.jsonl from a Gym eval run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lean-server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--lean-server-api-key", default=None)
    parser.add_argument("--timeout", type=int, default=60, help="Lean timeout per compile, seconds")
    parser.add_argument("--concurrency", type=int, default=8, help="Keep at or below LEAN_SERVER_MAX_REPLS")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--upstream-cache", type=Path, default=Path(tempfile.gettempdir()) / "combibench_upstream")
    parser.add_argument(
        "--rescore-with",
        type=Path,
        default=None,
        help="Optional second rollouts.jsonl holding Gym verdicts to compare against, keyed by theorem_name and "
        "rollout index; by default the verdicts already in --rollouts are used",
    )
    parser.add_argument(
        "--full-rows",
        action="store_true",
        help="write every row into 'rows' rather than only the disagreements",
    )
    args = parser.parse_args()

    Lean4Client, one_stage_verify = import_upstream(fetch_upstream(args.upstream_cache))

    rows = [json.loads(line) for line in args.rollouts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    source = rows
    what = "rollouts"
    if args.rescore_with is not None:
        source = [
            json.loads(line) for line in args.rescore_with.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        what = "--rescore-with verdicts"
    gym_verdicts = index_by_key([(row_key(row, index), row) for index, row in enumerate(source)], what)

    client = compat_client(Lean4Client, args.lean_server_url, args.lean_server_api_key)

    def one(item: tuple[int, dict]) -> tuple[str, dict[str, Any]]:
        index, row = item
        key = row_key(row, index)
        error_type, _feedback, _code, _answers = one_stage_verify(
            text=model_text(row),
            formal_statement=row["formal_statement"],
            lean4_client=client,
            ground_truths=row.get("answers"),
        )
        upstream_name = getattr(error_type, "name", str(error_type))
        gym = gym_verdicts.get(key, {})
        record = {
            "upstream_error_type": upstream_name,
            "upstream_success": upstream_name == "SUCCESS",
            "gym_status": gym.get("status"),
            "gym_success": gym.get("reward") == 1.0,
        }
        record["agree"] = record["upstream_success"] == record["gym_success"]
        print(f"{key}: gym={record['gym_status']} upstream={upstream_name}", file=sys.stderr)
        return key, record

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        per_row = index_by_key(list(pool.map(one, enumerate(rows))), "rollouts")

    disagreements = {k: v for k, v in per_row.items() if not v["agree"]}
    summary = {
        "rows": len(per_row),
        "gym_successes": sum(1 for v in per_row.values() if v["gym_success"]),
        "upstream_successes": sum(1 for v in per_row.values() if v["upstream_success"]),
        "agreements": sum(1 for v in per_row.values() if v["agree"]),
        "disagreements": len(disagreements),
        # Which way each disagreement goes, and under which Gym status. The two
        # documented departures can only produce gym_only entries.
        "gym_only": sorted(k for k, v in disagreements.items() if v["gym_success"]),
        "upstream_only": sorted(k for k, v in disagreements.items() if v["upstream_success"]),
        "disagreement_by_gym_status": dict(Counter(v["gym_status"] for v in disagreements.values())),
        "gym_status_counts": dict(Counter(v["gym_status"] for v in per_row.values())),
        "upstream_error_type_counts": dict(Counter(v["upstream_error_type"] for v in per_row.values())),
    }
    # Only the disagreements are written: the agreeing rows are 1600 copies of the same
    # two fields, and a committed report is read for what did not match. ``--full-rows``
    # keeps the complete map for an ad-hoc comparison.
    report = {
        "summary": summary,
        "rollouts": str(args.rollouts),
        "upstream_revision": GITHUB_REVISION,
        "lean_server_url": args.lean_server_url,
        "rows": per_row if args.full_rows else disagreements,
    }
    if not args.full_rows:
        report["rows_note"] = "only disagreements are kept; pass --full-rows for every row"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
