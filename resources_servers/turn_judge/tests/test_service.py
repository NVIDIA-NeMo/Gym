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
"""Turn-judge service (service.py) with run_pair replaced by a fake judge: no sandbox, no judge model, CPU only."""

import asyncio
import gzip
import json
import re
import tarfile
import time
from pathlib import Path

import pytest

from resources_servers.turn_judge import service as svc_mod
from resources_servers.turn_judge.service import (
    JudgeRequest,
    JudgeService,
    ServiceConfig,
    opensandbox_reaper,
    sandbox_run_label,
)

DECISIONS = {
    "group_ill_posed": False,
    "ill_posed_evidence": "",
    "decisions": [{"id": "D1", "what": "which file to edit", "correct": "x.py"}],
}


# ------------------------------------------------------------------------------------------------ synthetic groups


def reasoning(text):
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


def output_of(gi: int, n_calls: int = 4, lead_message: bool = False) -> list:
    """Call c = reasoning 'g<gi> call <c>: plan step <c>' + a tool call + its output (3 items). lead_message: the
    output starts with a message item, so call 0 is that item alone and every later call shifts by one item."""
    out = (
        [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "starting"}]}]
        if lead_message
        else []
    )
    for c in range(n_calls):
        out += [
            reasoning(f"g{gi} call {c}: plan step {c}"),
            {
                "type": "function_call",
                "name": "bash",
                "call_id": f"c{c}",
                "arguments": json.dumps({"command": f"echo g{gi}-{c}"}),
            },
            {"type": "function_call_output", "call_id": f"c{c}", "output": f"g{gi}-{c}"},
        ]
    return out


TASK = {
    "instance_id": "toy__task-1",
    "image_ref": "registry.example/toy:1",
    "repo": "toy/toy",
    "patch": "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n",
    "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
    "FAIL_TO_PASS": '["tests/test_x.py::test_a"]',
    "PASS_TO_PASS": "[]",
    "problem_statement": "make a == 2",
    "responses_create_params": {"input": [{"role": "user", "content": "Set a to 2 in x.py"}]},
}


def rollout(gi: int, reward: float, **kw) -> dict:
    out = kw.pop("output", None) or output_of(gi, lead_message=kw.pop("lead_message", False))
    ok = reward > 0.5
    return {
        "generation_index": gi,
        "reward": reward,
        "mask_sample": False,
        "context_limited": False,
        "cap_killed": False,
        "evaluation_completed": True,
        "resolved": ok,
        "test_results": {
            "FAIL_TO_PASS": {
                "success": ["tests/test_x.py::test_a"] if ok else [],
                "failure": [] if ok else ["tests/test_x.py::test_a"],
            },
            "PASS_TO_PASS": {"success": [], "failure": []},
        },
        "test_output": "PASSED tests/test_x.py::test_a" if ok else "FAILED tests/test_x.py::test_a",
        "test_return_code": 0 if ok else 1,
        "model_patch": f"diff --git a/x.py b/x.py\n+a = {2 if ok else 3}\n",
        "patch_applied": True,
        "opencode_results_fpath": None,
        "n_calls": len([i for i, it in enumerate(out) if i == 0 or it["type"] == "reasoning"]),
        "response_output": out,
        **kw,
    }


def payload(rid: str, rewards: dict, deadline_in: float | None = 60.0, **kw) -> dict:
    """rewards: generation_index -> reward (deliberately unsorted, non-contiguous indices)."""
    rolls = [rollout(gi, r, lead_message=(gi == 5)) for gi, r in rewards.items()]
    return {
        "request_id": rid,
        "group_id": f"grp-{rid}",
        "agent_name": "swe_toy",
        "task": TASK,
        "rollouts": rolls,
        "deadline_unix": None if deadline_in is None else time.time() + deadline_in,
        "stage1_timeout_s": 900,
        "pair_timeout_s": 1200,
        "overrides": {},
        **kw,
    }


MIXED = {7: 1.0, 2: 0.0, 5: 0.0, 0: 1.0}  # R0=g0 pass, R1=g2 fail, R2=g5 fail (lead message), R3=g7 pass
MIXED6 = {7: 1.0, 2: 0.0, 5: 0.0, 0: 1.0, 9: 0.0, 11: 0.0}  # 2 pass / 4 fail -> 4 pairs


# ------------------------------------------------------------------------------------------------ fake judge


class FakeJudge:
    """Stands in for runner.run_pair. Stage 1 returns ``stage1`` (a dict result or a callable); a pair labels the
    failing rollout's call ``harm_call`` harmful and the passing rollout's call ``good_call`` good (decision D1), with
    the label idx one item past the call start (the merge snaps it) and a verbatim quote of the call's reasoning."""

    def __init__(self, stage1=None, delay=None, pair_status=None, harm_call=2, good_call=1):
        self.stage1 = stage1 if stage1 is not None else ok(json.dumps(DECISIONS))
        self.delay = delay or (lambda kind, fail, passing: 0.0)
        self.pair_status = pair_status or (lambda fail, passing: "ok")
        self.harm_call, self.good_call = harm_call, good_call
        self.calls, self.cancelled, self.urls = [], [], []
        self.active = self.max_active = 0

    async def __call__(
        self,
        provider,
        image,
        files,
        instructions,
        settings,
        required="out/pair.json",
        extra_reads=("out/decisions.json",),
    ):
        kind = "stage1" if required == "out/decisions.json" else "pair"
        meta = json.loads(files["ws/group.json"])
        rs = sorted({int(m.group(1)) for k in files if (m := re.match(r"ws/R(\d+)/", k))})
        fail = passing = None
        if kind == "pair":
            fail = next(k for k in rs if meta["rollouts"][k]["reward"] <= 0.5)
            passing = next(k for k in rs if meta["rollouts"][k]["reward"] > 0.5)
            assert files["decisions.json"], "pairs get the stage-1 decision list"
        self.calls.append((kind, fail, passing))
        self.urls.append(settings.judge_base_url)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay(kind, fail, passing))
            if kind == "stage1":
                return self.stage1() if callable(self.stage1) else dict(self.stage1)
            st = self.pair_status(fail, passing)
            if st != "ok":
                return {
                    "status": st,
                    "pair_json": None,
                    "timings": {},
                    "error": "fake failure",
                    "opencode_finished": st == "no_output",
                }
            verdict = {
                "ill_posed": False,
                "fail": self._side(files, fail, "harmful", self.harm_call),
                "pass": self._side(files, passing, "good", self.good_call),
            }
            return ok(json.dumps(verdict))
        except asyncio.CancelledError:
            self.cancelled.append((kind, fail, passing))  # the real runner stops the sandbox in its finally
            raise
        finally:
            self.active -= 1

    @staticmethod
    def _side(files, k, label, call):
        traj = files[f"ws/R{k}/trajectory.txt"].decode()
        idx, text = next(
            (int(m.group(1)), m.group(3))
            for m in re.finditer(r"===== #(\d+) \[call (\d+)\] reasoning =====\n(.*)\n", traj)
            if int(m.group(2)) == call
        )
        return {
            "labels": [{"idx": idx + 1, "label": label, "decision": "D1", "evidence": f'#{idx}: "{text}"'}],
            "cause": "wrong file" if label == "harmful" else "",
            "summary": f"R{k}",
        }


def ok(text):
    return {
        "status": "ok",
        "pair_json": text,
        "timings": {"total_s": 0.0},
        "error": "",
        "opencode_finished": True,
        "export": '{"info": {}}',
    }


def make_service(tmp_path: Path, judge, max_sessions=240, endpoint=True, reaper=None, **kw) -> JudgeService:
    ep = tmp_path / "judge_endpoint.env"
    if endpoint:
        ep.write_text("export JUDGE_BASE_URL=http://judge-a:18080/v1\nexport JUDGE_MODEL_NAME=qwen-test\n")
    cfg = ServiceConfig(
        labeling=b"# LABELING\n",
        endpoint_env=str(ep),
        work_root=tmp_path / "work",
        run_id="tjsvc-test-1",
        max_sessions=max_sessions,
        archive_dir=str(tmp_path / "archive"),
        gym_sha="abc1234",
        endpoint_poll_s=0.2,
        endpoint_max_age_s=0.0,
        shutdown_wait_s=5.0,
        **kw,
    )
    return JudgeService(cfg, provider={"fake": {}}, default_metadata={}, run_pair=judge, reaper=reaper)


async def wait_done(svc: JudgeService, rid: str, timeout: float = 20.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = svc.get(rid)
        if body and body["state"] == "done":
            return body["result"]
        await asyncio.sleep(0.02)
    raise AssertionError(f"{rid} not done after {timeout}s: {svc.get(rid)}")


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------------------------------------ tests


def test_ok_labels_idx_to_call_and_n_calls(tmp_path):
    judge = FakeJudge()

    async def main():
        svc = make_service(tmp_path, judge)
        body = svc.submit(JudgeRequest(**payload("r1", MIXED)))
        assert body == {"request_id": "r1", "state": "queued", "result": None}
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "ok", res
    info = res["info"]
    assert info["pairs_total"] == 2 and info["pairs_ok"] == 2 and info["stage1_status"] == "ok"
    assert info["sessions"] == 3 and info["rollout_order"] == [0, 2, 5, 7] and info["n_calls_mismatch"] == {}
    assert info["judge_model"] == "qwen-test" and info["gym_sha"] == "abc1234" and info["endpoint_wait_s"] == 0
    assert judge.calls[0] == ("stage1", None, None) and len(judge.calls) == 3
    assert set(res["labels"]) == {"0", "2", "5", "7"}
    # plain outputs: call c starts at item 3c; the fake put idx 3c+1, the merge snapped it to 3c
    # inline outputs, no OpenCode export: the call alignment is not proven (the trainer applies no label then)
    NA = {"aligned": False, "trailing_empty_calls": 0, "alignment_reason": "no_export", "step_output_tokens": None}
    assert res["labels"]["0"] == {
        "n_calls": 4,
        "labels": [{"idx": 3, "call": 1, "label": "good", "decision_id": "D1"}],
        **NA,
    }
    assert res["labels"]["2"] == {
        "n_calls": 4,
        "labels": [{"idx": 6, "call": 2, "label": "harmful", "decision_id": "D1"}],
        **NA,
    }
    # g5 starts with a message item: call 0 = item 0 alone, call c (c >= 1) starts at item 3c - 2
    assert res["labels"]["5"]["n_calls"] == 5
    assert res["labels"]["5"]["labels"] == [{"idx": 4, "call": 2, "label": "harmful", "decision_id": "D1"}]


def test_idempotent_post(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 0.2)

    async def main():
        svc = make_service(tmp_path, judge)
        b1 = svc.submit(JudgeRequest(**payload("r1", MIXED)))
        await asyncio.sleep(0.05)
        b2 = svc.submit(JudgeRequest(**payload("r1", {1: 1.0, 3: 0.0})))  # same id, other payload: starts nothing
        res = await wait_done(svc, "r1")
        b3 = svc.submit(JudgeRequest(**payload("r1", MIXED)))
        await asyncio.sleep(0.1)
        await svc.shutdown()
        return b1, b2, b3, res

    b1, b2, b3, res = run(main())
    assert b1["state"] == "queued" and b2["state"] in ("queued", "running") and b2["result"] is None
    assert b3 == {"request_id": "r1", "state": "done", "result": res}
    assert len(judge.calls) == 3  # one stage 1 + two pairs, never repeated
    assert sorted(res["labels"]) == ["0", "2", "5", "7"]


def test_stage1_ill_posed_returns_at_once(tmp_path):
    judge = FakeJudge(
        stage1=ok(json.dumps({"group_ill_posed": True, "ill_posed_evidence": "spec says X, tests Y", "decisions": []}))
    )

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED)))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "ill_posed" and res["ill_posed_evidence"] == "spec says X, tests Y"
    assert res["info"]["ill_posed_by"] == "stage1" and res["info"]["pairs_total"] == 0
    assert judge.calls == [("stage1", None, None)]
    assert all(v["labels"] == [] for v in res["labels"].values()) and res["labels"]["5"]["n_calls"] == 5


@pytest.mark.parametrize(
    "stage1,err",
    [
        (
            {
                "status": "timeout",
                "pair_json": None,
                "timings": {},
                "error": "session: TimeoutError",
                "opencode_finished": False,
            },
            "stage 1 timeout",
        ),
        (ok(json.dumps({"group_ill_posed": False, "decisions": []})), "no decision list"),
        (ok("not json at all"), "not parseable"),
    ],
)
def test_stage1_failure_no_retry(tmp_path, stage1, err):
    judge = FakeJudge(stage1=stage1)

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED)))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "stage1_failed" and err in res["error"], res
    assert judge.calls == [("stage1", None, None)]  # one session: no retry, no pairs
    assert res["info"]["sessions"] == 1 and res["info"]["stage1_status"] == stage1["status"]


def test_no_pairs_for_uniform_group(tmp_path):
    judge = FakeJudge()

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", {3: 1.0, 1: 1.0, 8: 1.0})))
        svc.submit(JudgeRequest(**payload("r2", {3: 0.0, 1: 0.0})))
        return await wait_done(svc, "r1"), await wait_done(svc, "r2"), await svc.shutdown()

    r1, r2, _ = run(main())
    assert r1["status"] == "no_pairs" and r2["status"] == "no_pairs" and judge.calls == []
    NA = {"aligned": False, "trailing_empty_calls": 0, "alignment_reason": "no_export", "step_output_tokens": None}
    assert r1["labels"] == {
        "1": {"n_calls": 4, "labels": [], **NA},
        "3": {"n_calls": 4, "labels": [], **NA},
        "8": {"n_calls": 4, "labels": [], **NA},
    }


def test_deadline_partial_cancels_unfinished(tmp_path):
    # 4 pairs (each fail with a pass); the pairs of R4 (g9) and R5 (g11) never finish before the deadline
    slow = {4, 5}
    judge = FakeJudge(delay=lambda kind, f, p: 30.0 if kind == "pair" and f in slow else 0.05)

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED6, deadline_in=1.5)))
        t0 = time.time()
        res = await wait_done(svc, "r1")
        wall = time.time() - t0
        await svc.shutdown()
        return res, wall

    res, wall = run(main())
    assert wall < 5, wall
    assert res["status"] == "partial", res
    i = res["info"]
    assert (i["pairs_total"], i["pairs_ok"], i["pairs_cancelled"], i["pairs_failed"]) == (4, 2, 2, 0)
    assert sorted(f for k, f, p in judge.cancelled) == [4, 5] and judge.active == 0
    labeled = {g for g, v in res["labels"].items() if v["labels"]}
    assert {"2", "5"} <= labeled and not ({"9", "11"} & labeled)  # cut pairs' fails have no labels
    rows = [json.loads(ln) for ln in open(tmp_path / "work" / "logs" / "pairs.jsonl")]
    assert sum(r["status"] == "cancelled" for r in rows) == 2 and sum(r["status"] == "ok" for r in rows) == 3


def test_deadline_during_stage1(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 30.0)

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED, deadline_in=0.5)))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "stage1_failed" and "deadline" in res["error"]
    assert judge.cancelled == [("stage1", None, None)]


def test_failed_and_capped_pairs_are_missing_judgments(tmp_path):
    # MIXED6: R0, R3 pass; R1, R2, R4, R5 fail -> 4 pairs. R4's session errors, R5's is capped (no pair.json)
    judge = FakeJudge(pair_status=lambda f, p: {4: "error", 5: "timeout"}.get(f, "ok"))

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED6)))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "partial"
    i = res["info"]
    assert (i["pairs_ok"], i["pairs_failed"], i["pairs_capped"], i["pairs_cancelled"]) == (2, 1, 1, 0)
    assert res["labels"]["9"]["labels"] == [] and res["labels"]["11"]["labels"] == []
    assert res["labels"]["2"]["labels"] and res["labels"]["5"]["labels"]
    # each pass is in two pairs; its labels come only from the judgments that exist (one per pass here)
    assert [x["label"] for x in res["labels"]["0"]["labels"] + res["labels"]["7"]["labels"]] == ["good", "good"]


def test_delete_cancels_and_forgets(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 0.05 if kind == "stage1" else 30.0)

    async def main():
        svc = make_service(tmp_path, judge)
        svc.submit(JudgeRequest(**payload("r1", MIXED)))
        while sum(k == "pair" for k, _, _ in judge.calls) < 2:
            await asyncio.sleep(0.02)
        assert svc.get("r1")["state"] == "running"
        body = await svc.delete("r1")
        gone = svc.get("r1")
        again = svc.submit(JudgeRequest(**payload("r1", MIXED)))  # a new judgment under the same id
        missing = await svc.delete("nope")
        await svc.shutdown()
        return body, gone, again, missing

    body, gone, again, missing = run(main())
    assert body["state"] == "done" and body["result"]["status"] == "error" and "DELETE" in body["result"]["error"]
    first_pairs = {c for c in judge.calls[:3] if c[0] == "pair"}
    assert len(first_pairs) == 2 and set(judge.cancelled[:2]) == first_pairs  # DELETE cancelled both running pairs
    assert gone is None and again["state"] == "queued" and missing is None


def test_global_session_cap(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 0.15)

    async def main():
        svc = make_service(tmp_path, judge, max_sessions=2)
        for n in range(3):
            svc.submit(JudgeRequest(**payload(f"r{n}", MIXED6)))
        res = [await wait_done(svc, f"r{n}") for n in range(3)]
        h = svc.health()
        await svc.shutdown()
        return res, h

    res, h = run(main())
    assert all(r["status"] == "ok" for r in res)
    assert judge.max_active == 2 and len(judge.calls) == 3 * 5
    assert h["sessions"] == 0 and h["sessions_total"] == 15 and h["running"] == 0


def test_n_calls_mismatch_and_trajectory_by_reference(tmp_path):
    judge = FakeJudge()
    art = tmp_path / "artifacts" / "uuid-7"
    art.mkdir(parents=True)
    out7 = output_of(7)
    (art / "generation.json").write_text(json.dumps({"response": {"output": out7}, "execution": {}}))
    (art / "export.json").write_text("{}")
    p = payload("r1", MIXED)
    for r in p["rollouts"]:
        if r["generation_index"] == 7:  # by reference: Gym's opencode_results_fpath names export.json
            r["response_output"], r["opencode_results_fpath"] = None, str(art / "export.json")
        if r["generation_index"] == 2:
            r["n_calls"] = 99

    async def main():
        svc = make_service(tmp_path, judge, keep_workspaces=True)
        svc.submit(JudgeRequest(**p))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "ok" and res["info"]["n_calls_mismatch"] == {"2": [99, 4]}
    assert res["labels"]["7"]["labels"] and res["labels"]["7"]["n_calls"] == 4
    ws = next((tmp_path / "work" / "jobs").iterdir()) / "ws"
    g = json.loads((ws / "group.json").read_text())
    assert [r["R"] for r in g["rollouts"]] == [0, 1, 2, 3] and g["rollouts"][3]["artifact"] == str(art)
    assert "g7 call 3: plan step 3" in (ws / "R3" / "trajectory.txt").read_text()
    assert (ws / "TASK.md").read_text().startswith("# toy__task-1\n\n## Prompt the agent received\nSet a to 2 in x.py")


def test_bad_requests(tmp_path):
    async def main():
        svc = make_service(tmp_path, FakeJudge())
        errs = []
        for p in (
            payload("a", MIXED, overrides={"sedd": 1}),
            payload("b", MIXED, overrides={"variant": "A"}),
            {**payload("c", MIXED), "task": {"instance_id": "x"}},
        ):
            try:
                svc.submit(JudgeRequest(**p))
            except svc_mod.BadRequest as e:
                errs.append(str(e))
        p = payload("d", MIXED)
        p["rollouts"][1]["generation_index"] = p["rollouts"][0]["generation_index"]
        try:
            svc.submit(JudgeRequest(**p))
        except svc_mod.BadRequest as e:
            errs.append(str(e))
        await svc.shutdown()
        return errs, svc.jobs

    errs, jobs = run(main())
    assert len(errs) == 4 and "unknown overrides" in errs[0] and "variant" in errs[1] and "image" in errs[2]
    assert "duplicate generation_index" in errs[3] and not jobs


def test_endpoint_missing_then_appears_and_changes(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 0.3 if kind == "stage1" else 0.05)

    async def main():
        svc = make_service(tmp_path, judge, endpoint=False)
        ep = Path(svc.cfg.endpoint_env)
        assert svc.health()["judge_base_url"] is None
        svc.submit(JudgeRequest(**payload("r1", MIXED, deadline_in=20)))
        await asyncio.sleep(0.7)
        assert judge.calls == [] and svc.get("r1")["state"] == "queued"
        tmp = ep.with_name("x.tmp")
        tmp.write_text("export JUDGE_BASE_URL=http://judge-a:18080/v1\nexport JUDGE_MODEL_NAME=qwen-test\n# job 1\n")
        tmp.replace(ep)
        while not judge.calls:
            await asyncio.sleep(0.02)
        tmp.write_text("export JUDGE_BASE_URL=http://judge-b:18080/v1\nexport JUDGE_MODEL_NAME=qwen-test\n# job 2\n")
        tmp.replace(ep)  # the next serving job took over while stage 1 runs: pairs must use it
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "ok"
    assert judge.urls == ["http://judge-a:18080/v1", "http://judge-b:18080/v1", "http://judge-b:18080/v1"]
    assert res["info"]["endpoint_wait_s"] >= 0.5 and res["info"]["judge_base_url"] == "http://judge-a:18080/v1"


def test_endpoint_missing_until_deadline(tmp_path):
    judge = FakeJudge()

    async def main():
        svc = make_service(tmp_path, judge, endpoint=False)
        svc.submit(JudgeRequest(**payload("r1", MIXED, deadline_in=1.0)))
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "stage1_failed" and "waiting for the judge endpoint file" in res["error"]
    assert judge.calls == [] and res["info"]["endpoint_wait_s"] >= 0.8 and res["info"]["sessions"] == 0


def test_endpoint_vanishes_between_stage1_and_pairs(tmp_path):
    judge = FakeJudge(delay=lambda kind, f, p: 0.3 if kind == "stage1" else 0.05)

    async def main():
        svc = make_service(tmp_path, judge)
        ep = Path(svc.cfg.endpoint_env)
        svc.submit(JudgeRequest(**payload("r1", MIXED, deadline_in=1.5)))
        while not judge.calls:
            await asyncio.sleep(0.02)
        ep.unlink()  # the serving job exited with no successor
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return res

    res = run(main())
    assert res["status"] == "partial" and res["info"]["pairs_ok"] == 0 and res["info"]["pairs_cancelled"] == 2
    assert res["info"]["pairs_cut_waiting_endpoint"] == 2 and len(judge.calls) == 1
    assert all(v["labels"] == [] for v in res["labels"].values())


def test_startup_reap_gates_sessions(tmp_path):
    judge = FakeJudge()
    reaps = []

    async def reaper(label):
        reaps.append((label, time.time()))
        await asyncio.sleep(0.5)  # listing / deleting the previous job's orphans
        return {"matched": 1}

    async def main():
        svc = make_service(tmp_path, judge, reaper=reaper)
        t0 = time.time()
        await svc.start()
        svc.submit(JudgeRequest(**payload("r1", MIXED)))  # accepted at once (the service listens during the reap)
        await asyncio.sleep(0.3)
        early = list(judge.calls)
        res = await wait_done(svc, "r1")
        await svc.shutdown()
        return early, res, t0

    early, res, t0 = run(main())
    assert early == [] and res["status"] == "ok"  # no session (sandbox) before the reap of this run label finished
    assert res["info"]["startup_reap_wait_s"] >= 0.4
    assert [r[0] for r in reaps] == ["tjsvc-test-1", "tjsvc-test-1"]  # startup, then exit


@pytest.mark.parametrize("keep_exports", [False, True])
def test_shutdown_cancels_reaps_and_archives_once(tmp_path, keep_exports):
    judge = FakeJudge(delay=lambda kind, f, p: 0.05 if kind == "stage1" else 30.0)
    reaps = []

    async def reaper(label):
        reaps.append(label)
        return {"matched": 0}

    async def main():
        svc = make_service(tmp_path, judge, reaper=reaper, keep_exports=keep_exports)
        await svc.start()
        svc.submit(JudgeRequest(**payload("r1", MIXED)))
        while sum(k == "pair" for k, _, _ in judge.calls) < 2:
            await asyncio.sleep(0.02)
        await svc.shutdown()
        await svc.shutdown()  # idempotent: one archive
        with pytest.raises(RuntimeError):
            svc.submit(JudgeRequest(**payload("r2", MIXED)))
        return svc.get("r1")

    body = run(main())
    assert body["result"]["status"] == "error" and "service shutdown" in body["result"]["error"]
    assert len([c for c in judge.cancelled if c[0] == "pair"]) == 2
    assert reaps == ["tjsvc-test-1", "tjsvc-test-1"]
    assert [p.name for p in (tmp_path / "archive").iterdir()] == ["tjsvc-test-1.tar.gz"]
    with tarfile.open(tmp_path / "archive" / "tjsvc-test-1.tar.gz") as tar:
        names = set(tar.getnames())
        rows = [json.loads(ln) for ln in tar.extractfile("tjsvc-test-1/results.jsonl")]
        prow = [json.loads(ln) for ln in tar.extractfile("tjsvc-test-1/pairs.jsonl")]
        ex = tar.extractfile("tjsvc-test-1/exports.jsonl.gz").read() if keep_exports else None
    assert {"tjsvc-test-1/pairs.jsonl", "tjsvc-test-1/results.jsonl"} <= names
    assert ("tjsvc-test-1/exports.jsonl.gz" in names) == keep_exports
    assert rows[-1]["request_id"] == "r1" and rows[-1]["result"]["status"] == "error"
    summary = rows[-1]["payload"]
    assert summary["task"] == "toy__task-1" and len(summary["rollouts"]) == 4
    assert "test_output" not in summary["rollouts"][0] and "response_output" not in summary["rollouts"][0]
    assert prow[0]["kind"] == "stage1" and json.loads(prow[0]["pair_json"])["decisions"]  # decisions.json text
    if keep_exports:
        assert json.loads(gzip.decompress(ex).splitlines()[0])["fail"] == -1  # stage-1 export logged
    else:  # node-local only, removed with the other node-local bulk once the archive is written
        assert not (tmp_path / "work" / "exports.jsonl.gz").exists()
    assert not (tmp_path / "work" / "answer_keys").exists() and (tmp_path / "work" / "logs" / "pairs.jsonl").exists()
    assert not (tmp_path / "work" / "jobs").exists()  # workspaces removed


def test_sandbox_run_label():
    assert sandbox_run_label("tj-train-short-run") == "tj-train-short-run"
    base = "tj-train-nt4nano-3env-sa-sft6000-grpo-swetb-superband-sugamrepro-nosub-32x16-rf3-cispo"
    a, b = (
        sandbox_run_label(base + "-turncredit-20261116-010101"),
        sandbox_run_label(base + "-shuffled-20261116-010101"),
    )
    assert len(a) <= 63 and len(b) <= 63 and a != b and a.startswith("tj-train-nt4nano")
    assert sandbox_run_label(base + "-turncredit-20261116-010101") == a  # stable across the run's jobs


def test_reaper_selection(monkeypatch):
    """opensandbox_reaper's filter: role turn-judge, this user, run label == ours (exactly)."""
    items = [
        {
            "id": "a",
            "metadata": {"role": "turn-judge", "nemo-gym.nvidia.com/user": "me", "nemo-gym.nvidia.com/run": "tjsvc-1"},
        },
        {
            "id": "b",
            "metadata": {"role": "turn-judge", "nemo-gym.nvidia.com/user": "me", "nemo-gym.nvidia.com/run": "tjsvc-0"},
        },
        {
            "id": "c",
            "metadata": {
                "role": "turn-judge",
                "nemo-gym.nvidia.com/user": "me",
                "nemo-gym.nvidia.com/run": "tj-20261115-1",
            },
        },  # offline round: never touched
        {
            "id": "d",
            "metadata": {
                "role": "turn-judge",
                "nemo-gym.nvidia.com/user": "other",
                "nemo-gym.nvidia.com/run": "tjsvc-0",
            },
        },
        {
            "id": "e",
            "metadata": {"role": "grader", "nemo-gym.nvidia.com/user": "me", "nemo-gym.nvidia.com/run": "tjsvc-0"},
        },
    ]
    deleted = []

    class Resp:
        def __init__(self, data=None, status=200):
            self.data, self.status = data, status

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def json(self, content_type=None):
            return self.data

        async def read(self):
            return b""

    class Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, url, params=None):
            return Resp({"items": items, "pagination": {"hasNextPage": False}})

        def delete(self, url):
            deleted.append(url.rsplit("/", 1)[-1])
            return Resp(status=204)

    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", Session)
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://cell.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "k")
    reap = opensandbox_reaper("me")
    out = run(reap("tjsvc-1"))
    assert deleted == ["a"] and out["matched"] == 1 and out["listed"] == 5
    deleted.clear()
    run(reap("tjsvc-0"))
    assert deleted == ["b"]  # never another user's, never another role's


def test_http_routes(tmp_path):
    from fastapi.testclient import TestClient

    judge = FakeJudge(delay=lambda kind, f, p: 0.05)
    svc = make_service(tmp_path, judge)
    with TestClient(svc_mod.create_app(svc)) as client:
        assert client.get("/health").json()["ok"] is True
        r = client.post("/judge_group", json=payload("grp/1:a", MIXED))
        assert r.status_code == 200 and r.json()["state"] == "queued"
        assert client.post("/judge_group", json=payload("bad", MIXED, overrides={"x": 1})).status_code == 400
        assert client.post("/judge_group", json={"request_id": "x"}).status_code == 422
        t0 = time.time()
        while client.get("/judge_group/grp/1:a").json()["state"] != "done" and time.time() - t0 < 20:
            time.sleep(0.05)
        body = client.get("/judge_group/grp/1:a").json()
        assert body["result"]["status"] == "ok"
        assert client.get("/judge_group/unknown").status_code == 404
        assert client.delete("/judge_group/grp/1:a").json()["state"] == "done"
        assert client.get("/judge_group/grp/1:a").status_code == 404
        h = client.get("/health").json()
        assert h["judge_base_url"] == "http://judge-a:18080/v1" and h["run_id"] == "tjsvc-test-1"
        r = client.post("/judge_group", json=payload("slow", MIXED))
        assert client.post("/shutdown").json()["shutting_down"] is True
        assert (tmp_path / "archive" / "tjsvc-test-1.tar.gz").exists()  # archived before /shutdown returned
        assert client.post("/judge_group", json=payload("late", MIXED)).status_code == 503
        assert client.get("/judge_group/slow").json()["state"] == "done"
        assert client.get("/health").json()["ok"] is False


def test_alignment_from_the_export_beside_generation_json(tmp_path):
    """A rollout given by reference: the service reads export.json next to generation.json and reports the proof."""
    import json as _json

    from resources_servers.turn_judge.service import RolloutIn, rollout_alignment

    d = tmp_path / "artifact"
    d.mkdir()
    # OpenCode writes one assistant message per step, with that step's usage
    export = {
        "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "task"}]},
            {
                "info": {"role": "assistant", "tokens": {"output": 57}},
                "parts": [
                    {"type": "step-start"},
                    {"type": "text", "text": "<think>a</think>"},
                    {"type": "tool", "callID": "c1", "tool": "bash", "state": {"input": {}, "output": ""}},
                    {"type": "step-finish"},
                ],
            },
            {
                "info": {"role": "assistant", "tokens": {"output": 1}},
                "parts": [{"type": "step-start"}, {"type": "step-finish"}],
            },
        ]
    }
    output = [
        {"type": "reasoning", "summary": [{"text": "a"}]},
        {"type": "function_call", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1"},
    ]
    (d / "export.json").write_text(_json.dumps(export))
    r = RolloutIn(generation_index=0, reward=0.0, opencode_results_fpath=str(d / "export.json"))
    assert rollout_alignment(r, output) == {
        "aligned": True,
        "trailing_empty_calls": 1,
        "alignment_reason": "ok",
        "step_output_tokens": [57, 1],
    }
    r_missing = RolloutIn(
        generation_index=0, reward=0.0, opencode_results_fpath=str(tmp_path / "nope" / "export.json")
    )
    assert rollout_alignment(r_missing, output)["aligned"] is False


def test_merge_tolerates_list_valued_free_text_fields():
    """A pair judge wrote `summary` (and `cause`) as lists: job 1966567 lost a group to `" || ".join` (18:50)."""
    from resources_servers.turn_judge.merge import merge_group

    out = [
        {"type": "reasoning", "summary": [{"text": "a"}]},
        {"type": "function_call", "call_id": "c"},
        {"type": "function_call_output", "call_id": "c", "output": "x"},
    ]
    meta = {
        "rollouts": [{"R": 0, "file": "f", "line": 0, "reward": 0.0}, {"R": 1, "file": "f", "line": 1, "reward": 1.0}]
    }
    verdict = {
        "ill_posed": False,
        "fail": {"rollout": "R0", "line": 0, "summary": ["first", "second"], "cause": ["wrong_action"], "labels": []},
        "pass": {"rollout": "R1", "line": 1, "summary": "ok", "labels": []},
    }
    merged = merge_group(
        meta,
        [{"fail": 0, "passing": 1, "larger_side": "fail", "verdict": verdict}],
        {0: out, 1: out},
        0.0,
        2,
        0.5,
        False,
    )
    assert merged["entries"][0]["summary"] == "first second" and merged["entries"][0]["cause"] == "wrong_action"


def test_sandbox_client_warm_up_runs_before_the_loop():
    """The sandbox client's one-time imports + Gym global config run at startup, not on the event loop (LOG 096).

    A fresh interpreter, as the service starts (sys.argv trimmed: Gym parses it as Hydra overrides)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys; sys.argv = sys.argv[:1]\n"
        "from resources_servers.turn_judge.service import warm_up_sandbox_client\n"
        "import nemo_gym.global_config as g\n"
        "s = warm_up_sandbox_client()\n"
        "assert g.maybe_get_global_config_dict() is not None\n"
        "print('WARM', s)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "WARM" in out.stdout
