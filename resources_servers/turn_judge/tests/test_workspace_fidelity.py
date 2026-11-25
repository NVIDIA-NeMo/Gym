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
"""Workspace fidelity: the workspace the service builds from a trainer-style payload equals the offline one.

For training groups with archived offline workspaces (turn_judge archive/ws/judge_ws_train.tar.gz = iterations/judge_ws,
built by tools/group_workspace.py train from the collection), build the payload the trainer would send (PHASE3_IMPL.md
"Rollout fields stashed by the trainer", taken from each rollout's verify.json; trajectory by reference: Gym's
opencode_results_fpath, i.e. export.json, whose sibling is generation.json) with shuffled generation indices, and compare
what the judge sessions would see: every workspace file (load_group), the trajectories, HEDGES.md, the stage-1 and pair
upload dicts. Service R<k> = generation_index order, offline R<k> = artifact-path order: files are compared after mapping
the numbering. Inherent difference: group.json "file" (the locator path of the node-local rollouts.jsonl).

Needs the cluster's shared storage (skipped elsewhere); reads ~16 verify.json + 16 generation.json per group and the
Terminal-Bench task folders, extracts the needed workspaces from the archive onto the test's tmp dir."""

import difflib
import json
import os
import random
import re
import tarfile
from pathlib import Path

import pytest

from resources_servers.turn_judge.merge import call_starts
from resources_servers.turn_judge.offline import group_files, group_hedges, load_group, pair_files
from resources_servers.turn_judge.pairing import make_pairs
from resources_servers.turn_judge.service import (
    JudgeRequest,
    JudgeService,
    ServiceConfig,
    _build_and_load,
    generation_json_path,
)

TJ = os.environ.get("TURN_JUDGE_FIXTURE_ROOT", "/nonexistent/turn_judge_fixtures")
WS_ARCHIVE = f"{TJ}/archive/ws/judge_ws_train.tar.gz"
TASKS = f"{TJ}/iterations/012-train-collection/tasks_96.jsonl"
LABELING = str(Path(__file__).resolve().parents[1] / "LABELING_annotator.md")
# 2 SWE (Python, JS/Jest) + 2 Terminal-Bench (atcb-2609 with SPEC.md; mercor deliveries)
GROUPS = [
    "biopragmatics_curies_pr1",
    "facebook-react-34069",
    "deduplicated-vault-credential-sweep",
    "strict-versioning-cascade",
]

pytestmark = pytest.mark.skipif(
    not (os.path.exists(WS_ARCHIVE) and os.path.exists(TASKS)),
    reason="needs the turn_judge archives on the cluster's shared storage",
)


@pytest.fixture(scope="module")
def offline_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline_ws")
    with tarfile.open(WS_ARCHIVE, "r:gz") as tar:  # one sequential read of the archive
        for m in tar:
            if m.name.split("/", 1)[0] in GROUPS:
                tar.extract(m, root, filter="data")
    return root


@pytest.fixture(scope="module")
def task_rows():
    with open(TASKS) as f:
        return {(r := json.loads(ln))["instance_id"]: r for ln in f}


def trainer_payload(group: str, offline_ws: Path, row: dict, gis: list[int]) -> dict:
    """What the trainer would POST: per rollout the stashed fields from verify.json, the trajectory by reference."""
    with open(offline_ws / "rollouts.jsonl") as f:
        recs = [json.loads(ln) for ln in f]  # offline order = artifact path order
    rollouts = []
    for i, rec in enumerate(recs):
        with open(os.path.join(rec["artifact"], "verify.json")) as f:
            v = json.load(f)
        rollouts.append(
            {
                "generation_index": gis[i],
                "reward": v["reward"],
                "mask_sample": bool(v.get("mask_sample")),
                "context_limited": bool(v.get("opencode_context_limited")),
                "cap_killed": "signal: killed" in (v.get("opencode_run_stderr") or ""),
                "evaluation_completed": v.get("evaluation_completed"),
                "resolved": v.get("resolved"),
                "test_results": v.get("test_results"),
                "test_output": v.get("test_output"),
                "test_return_code": v.get("test_return_code"),
                "model_patch": v.get("model_patch"),
                "patch_applied": v.get("patch_applied"),
                "opencode_results_fpath": v.get("opencode_results_fpath"),
                "n_calls": len(call_starts(rec["response"]["output"])),
                "response_output": None,
            }
        )
    return {
        "request_id": f"fid-{group}",
        "group_id": f"fid-{group}",
        "agent_name": "fidelity",
        "task": row,
        "rollouts": rollouts,
    }


# Where the builders write a rollout number (agent text may itself contain tokens like "R3": never touched):
# PREPASS / HEDGES section headings, MATRIX rows, the calls_index header (first line), PREPASS file references.
STRUCTURAL_R = re.compile(
    r"(?m)^## R(?P<a>\d+)(?= \()|^\| R(?P<b>\d+)(?= \|)|\AR(?P<c>\d+)(?=: )|"
    r"\bR(?P<d>\d+)(?=/(?:failing_assertions|test_output)\.txt)"
)


def renum(text: str, inv: dict[int, int]) -> str:
    def sub(m):
        g = next(k for k in "abcd" if m.group(k) is not None)
        s0, s1 = m.start(g) - m.start(0), m.end(g) - m.start(0)
        return m.group(0)[:s0] + str(inv.get(int(m.group(g)), int(m.group(g)))) + m.group(0)[s1:]

    return STRUCTURAL_R.sub(sub, text)


def sections(text: str, head: str) -> tuple[str, dict]:
    """Header + {R: section} of a per-rollout report (PREPASS.md / HEDGES.md: '## R<k> ...' sections)."""
    parts = re.split(rf"(?m)^(?={re.escape(head)}R\d+ )", text)
    return parts[0], {int(re.match(rf"{re.escape(head)}R(\d+)", p).group(1)): p for p in parts[1:]}


def normalize(rel: str, text: str) -> str:
    """Per-rollout blocks in R order (the service's are renumbered before this)."""
    if rel.endswith("PREPASS.md") or rel.endswith("HEDGES.md"):
        head, secs = sections(text, "## ")  # the file's last block lacks the blank line that separates the others
        return head + "".join(secs[k].rstrip("\n") + "\n\n" for k in sorted(secs))
    if rel.endswith("MATRIX.md"):
        return "\n".join(sorted(text.splitlines()))
    return text


def compare_text(rel: str, off: str, svc: str, inv: dict[int, int]) -> bool:
    """svc renumbered to offline numbering, per-rollout blocks reordered; prints a diff when they differ."""
    a, b = normalize(rel, off), normalize(rel, renum(svc, inv))
    if a != b:
        print(f"--- {rel} differs:")
        print(
            "".join(list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "offline", "service", n=1))[:40])
        )
    return a == b


def compare_group_json(off: bytes, svc: bytes, inv: dict[int, int]) -> list[str]:
    o, s = json.loads(off), json.loads(svc)
    diffs = []
    if (o["task"], o["terminal_bench"]) != (s["task"], s["terminal_bench"]):
        diffs.append("group.json task / terminal_bench")
    srs = []
    for r in s["rollouts"]:
        assert r["file"].endswith("/rollouts.jsonl") and r["line"] == r["R"]  # the inherent difference: the path
        srs.append({**r, "R": inv[r["R"]], "line": inv[r["line"]], "file": None})
    ors = [{**r, "file": None} for r in o["rollouts"]]
    if sorted(ors, key=lambda r: r["R"]) != sorted(srs, key=lambda r: r["R"]):
        diffs.append("group.json rollouts")
    return diffs


def compare_files(off: dict[str, bytes], svc: dict[str, bytes], m: dict[int, int], prefix: str = "") -> list[str]:
    """off / svc: {relative path: bytes} (a load_group files dict or an upload dict); m: offline R -> service R."""
    inv = {v: k for k, v in m.items()}
    rx = re.compile(rf"^{re.escape(prefix)}R(\d+)/")
    mapped = {rx.sub(lambda x: f"{prefix}R{m[int(x.group(1))]}/", rel): rel for rel in off}
    diffs = []
    if set(mapped) != set(svc):
        diffs.append(
            f"file sets differ: only offline {sorted(set(mapped) - set(svc))[:5]}, "
            f"only service {sorted(set(svc) - set(mapped))[:5]}"
        )
    for srel, orel in mapped.items():
        if srel not in svc:
            continue
        a, b = off[orel], svc[srel]
        a, b = (a.encode() if isinstance(a, str) else a), (b.encode() if isinstance(b, str) else b)
        if orel.endswith("group.json"):
            diffs += compare_group_json(a, b, inv)
        elif orel.endswith(("PREPASS.md", "HEDGES.md", "MATRIX.md", "calls_index.txt")):
            if not compare_text(orel, a.decode(), b.decode(), inv):
                diffs.append(orel)
        elif a != b:
            diffs.append(orel)
    return diffs


@pytest.mark.parametrize("group", GROUPS)
def test_service_workspace_equals_offline(group, offline_root, task_rows, tmp_path):
    offline_ws = offline_root / group
    group_o = load_group(offline_ws)
    group_hedges(group_o)
    meta_o = group_o["meta"]
    row = task_rows[meta_o["task"]]
    n = len(meta_o["rollouts"])
    gis = list(range(n))
    random.Random(f"fidelity:{group}").shuffle(gis)  # offline line i gets generation index gis[i]
    m = {i: gis[i] for i in range(n)}  # offline R i -> service R (rank of gis[i] = gis[i])
    assert sorted(m.values()) != list(m.values()), "the permutation must actually reorder"

    p = trainer_payload(group, offline_ws, row, gis)
    assert all(generation_json_path(r["opencode_results_fpath"]).name == "generation.json" for r in p["rollouts"])
    labeling = Path(LABELING).read_bytes()
    svc = JudgeService(
        ServiceConfig(
            labeling=labeling,
            endpoint_env=str(tmp_path / "none.env"),
            work_root=tmp_path / "work",
            run_id="tjsvc-fidelity",
        )
    )
    req = JudgeRequest(**p)
    built, group_s = _build_and_load(tmp_path / "ws1", req, svc._answer_key)
    built2, group_s2 = _build_and_load(tmp_path / "ws2", req, svc._answer_key)  # second build: answer-key cache hit
    group_s2["files"]["group.json"] = group_s2["files"]["group.json"].replace(
        b"/ws2/rollouts.jsonl", b"/ws1/rollouts.jsonl"
    )

    assert built["order"] == list(range(n)) and built["n_calls"] == built["n_calls_trainer"]
    assert built["task"] == meta_o["task"]
    # 1. every workspace file the judge sessions read
    diffs = compare_files(group_o["files"], group_s["files"], m)
    # 2. the cache-hit build is byte-identical to the first
    if group_s2["files"] != group_s["files"]:
        diffs.append("answer-key cache hit differs from the first build")
    # 3. trajectories (the merge's checks read these)
    for i in range(n):
        if group_o["outputs"][i] != group_s["outputs"][m[i]]:
            diffs.append(f"outputs R{i}")
    # 4. derived inputs: HEDGES.md, outcomes / pairing input
    if not compare_text("HEDGES.md", group_o["hedges"], group_s["hedges"], {v: k for k, v in m.items()}):
        diffs.append("HEDGES.md")
    rew_o = [r["reward"] for r in meta_o["rollouts"]]
    rew_s = [r["reward"] for r in group_s["meta"]["rollouts"]]
    if [rew_s[m[i]] for i in range(n)] != rew_o:
        diffs.append("rewards")
    # 5. the stage-1 upload and one pair upload (variant D)
    diffs += [
        f"stage1:{d}"
        for d in compare_files(
            group_files(group_o, labeling, "D"), group_files(group_s, labeling, "D"), m, prefix="ws/"
        )
    ]
    pr = make_pairs(rew_o, f"0:{group}")[0]
    diffs += [
        f"pair:{d}"
        for d in compare_files(
            pair_files(group_o, pr.fail, pr.passing, labeling, "D", "{}"),
            pair_files(group_s, m[pr.fail], m[pr.passing], labeling, "D", "{}"),
            m,
            prefix="ws/",
        )
    ]
    print(
        f"{group}: {len(group_o['files'])} workspace files, {n} rollouts, "
        f"{'TB' if meta_o['terminal_bench'] else 'SWE'}; differences: {diffs or 'none'}"
    )
    assert not diffs, diffs
