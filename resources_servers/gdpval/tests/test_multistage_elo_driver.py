# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import random
from pathlib import Path

import pytest

import resources_servers.gdpval.multistage_elo_driver as driver
from resources_servers.gdpval.multistage_elo import MultiStageEloConfig, StageResult, StageSpec
from resources_servers.gdpval.multistage_elo_driver import (
    _load_reference_elos,
    _parse_stage,
    build_judge_stage,
    build_verify_request_body,
    cached_task_ids,
    check_coverage,
    ensure_distribution,
    load_distribution,
    load_task_prompts,
    main,
    pool_per_reference,
    run_multistage_elo,
    stage_results_to_dict,
    task_repeat_dirs,
)


def _make_cache(root: Path, task_ids, repeats=("repeat_0",)):
    for tid in task_ids:
        for rep in repeats:
            d = root / f"task_{tid}" / rep
            d.mkdir(parents=True)
            (d / "finish_params.json").write_text("{}")


def _dist(groups):
    total = sum(len(v) for v in groups.values()) or 1
    return {k: {"percentage": len(v) / total, "task_ids": list(v)} for k, v in groups.items()}


class TestCacheDiscovery:
    def test_task_repeat_dirs_lists_attempted_repeats(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a"], repeats=("repeat_0", "repeat_1"))
        dirs = task_repeat_dirs(tmp_path, "a")
        assert [d.name for d in dirs] == ["repeat_0", "repeat_1"]

    def test_task_repeat_dirs_skips_unattempted(self, tmp_path: Path) -> None:
        (tmp_path / "task_a" / "repeat_0").mkdir(parents=True)  # no finish_params.json
        assert task_repeat_dirs(tmp_path, "a") == []

    def test_task_repeat_dirs_flat_layout(self, tmp_path: Path) -> None:
        d = tmp_path / "task_a"
        d.mkdir(parents=True)
        (d / "finish_params.json").write_text("{}")
        assert [p.name for p in task_repeat_dirs(tmp_path, "a")] == ["task_a"]

    def test_missing_task_returns_empty(self, tmp_path: Path) -> None:
        assert task_repeat_dirs(tmp_path, "ghost") == []

    def test_cached_task_ids(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a", "b"])
        assert cached_task_ids(tmp_path) == {"a", "b"}

    def test_cached_task_ids_missing_dir(self, tmp_path: Path) -> None:
        assert cached_task_ids(tmp_path / "nope") == set()

    def test_check_coverage(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a", "c"])
        present, missing = check_coverage(tmp_path, ["a", "b", "c"])
        assert present == ["a", "c"]
        assert missing == ["b"]


class TestPoolPerReference:
    def test_sums_counts_and_keeps_elo(self) -> None:
        responses = [
            {"per_reference": {"a": {"wins": 2, "losses": 1, "ties": 0, "reference_elo": 1000.0}}},
            {"per_reference": {"a": {"wins": 1, "losses": 0, "ties": 1, "reference_elo": 1000.0}}},
        ]
        pooled = pool_per_reference(responses)
        assert pooled["a"]["wins"] == 3
        assert pooled["a"]["losses"] == 1
        assert pooled["a"]["ties"] == 1
        assert pooled["a"]["reference_elo"] == 1000.0

    def test_handles_missing_per_reference(self) -> None:
        assert pool_per_reference([{}, {"per_reference": None}]) == {}


class TestLoaders:
    def test_load_distribution(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text(json.dumps(_dist({"x": ["a"]})))
        assert load_distribution(p)["x"]["task_ids"] == ["a"]

    def test_load_distribution_rejects_non_object(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text("[1,2,3]")
        with pytest.raises(ValueError):
            load_distribution(p)

    def test_load_task_prompts_top_level(self, tmp_path: Path) -> None:
        p = tmp_path / "b.jsonl"
        p.write_text(json.dumps({"task_id": "a", "prompt": "do x"}) + "\n")
        assert load_task_prompts(p) == {"a": "do x"}

    def test_load_task_prompts_metadata_nested(self, tmp_path: Path) -> None:
        p = tmp_path / "b.jsonl"
        p.write_text(json.dumps({"responses_create_params": {"metadata": {"task_id": "a", "prompt": "y"}}}) + "\n")
        assert load_task_prompts(p) == {"a": "y"}


class TestEnsureDistribution:
    def test_loads_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "d.json"
        p.write_text(json.dumps(_dist({"x": ["a"]})))
        dist, path = ensure_distribution(str(p))
        assert path == p
        assert dist["x"]["task_ids"] == ["a"]

    def test_builds_from_dataset_when_missing(self, tmp_path: Path) -> None:
        dataset = tmp_path / "tasks.jsonl"
        rows = [
            {"task_id": "t1", "occupation": "Lawyer"},
            {"task_id": "t2", "occupation": "Lawyer"},
            {"task_id": "t3", "occupation": "Nurse"},
        ]
        dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        cache = tmp_path / "cache"

        dist, path = ensure_distribution(None, dataset_path=str(dataset), cache_dir=str(cache))

        assert path == cache / "occupation_distribution.json"
        assert path.is_file()
        assert dist["Lawyer"]["task_ids"] == ["t1", "t2"]
        assert dist["Nurse"]["task_ids"] == ["t3"]

    def test_writes_to_distribution_path_when_given(self, tmp_path: Path) -> None:
        dataset = tmp_path / "tasks.jsonl"
        dataset.write_text(json.dumps({"task_id": "t1", "occupation": "Lawyer"}) + "\n")
        out = tmp_path / "sub" / "mydist.json"

        _dist_, path = ensure_distribution(str(out), dataset_path=str(dataset))

        assert path == out
        assert out.is_file()

    def test_custom_columns_in_filename(self, tmp_path: Path) -> None:
        dataset = tmp_path / "tasks.jsonl"
        dataset.write_text(json.dumps({"task_id": "t1", "sector": "Legal", "occupation": "Lawyer"}) + "\n")
        cache = tmp_path / "cache"
        _dist_, path = ensure_distribution(
            None, dataset_path=str(dataset), columns=["sector", "occupation"], cache_dir=str(cache)
        )
        assert path == cache / "sector_occupation_distribution.json"

    def test_raises_when_no_dataset_available(self, tmp_path: Path, monkeypatch) -> None:
        import responses_api_agents.stirrup_agent.task_distribution as td

        monkeypatch.setattr(td, "DEFAULT_DATASET_CANDIDATES", (tmp_path / "missing.jsonl",))
        with pytest.raises(FileNotFoundError):
            ensure_distribution(None, cache_dir=str(tmp_path / "cache"))


class TestBuildVerifyRequestBody:
    def test_includes_reference_ids_and_deliverables(self) -> None:
        body = build_verify_request_body("t1", "/cache/task_t1/repeat_0", "prompt", ["a", "b"])
        assert body["task_id"] == "t1"
        assert body["deliverables_dir"] == "/cache/task_t1/repeat_0"
        assert body["reference_ids"] == ["a", "b"]
        assert body["prompt"] == "prompt"


class TestBuildJudgeStage:
    def test_judges_present_tasks_and_pools(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a", "b"], repeats=("repeat_0", "repeat_1"))
        calls = []

        def fake_verify_one(task_id, deliverables_dir, prompt, reference_ids):
            calls.append((task_id, Path(deliverables_dir).name, tuple(reference_ids)))
            return {"per_reference": {reference_ids[0]: {"wins": 1, "losses": 0, "ties": 0, "reference_elo": 1000.0}}}

        judge = build_judge_stage(fake_verify_one, tmp_path, {"a": "pa", "b": "pb"})
        pooled = judge(["a", "b"], ["ref1"])
        # 2 tasks x 2 repeats = 4 verify calls.
        assert len(calls) == 4
        assert pooled["ref1"]["wins"] == 4

    def test_missing_raises_when_no_producer(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a"])
        judge = build_judge_stage(lambda *a: {}, tmp_path, {})
        with pytest.raises(FileNotFoundError):
            judge(["a", "missing"], ["ref1"])

    def test_missing_skipped_when_produce_missing_false(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a"])

        def fake_verify_one(task_id, deliverables_dir, prompt, reference_ids):
            return {"per_reference": {"ref1": {"wins": 1, "losses": 0, "ties": 0, "reference_elo": 1000.0}}}

        judge = build_judge_stage(fake_verify_one, tmp_path, {"a": ""}, produce_missing=False)
        pooled = judge(["a", "missing"], ["ref1"])
        assert pooled["ref1"]["wins"] == 1

    def test_producer_materializes_then_judges(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a"])

        def producer(task_ids):
            _make_cache(tmp_path, list(task_ids))

        def fake_verify_one(task_id, deliverables_dir, prompt, reference_ids):
            return {"per_reference": {"ref1": {"wins": 1, "losses": 0, "ties": 0, "reference_elo": 1000.0}}}

        judge = build_judge_stage(fake_verify_one, tmp_path, {}, producer=producer)
        pooled = judge(["a", "b"], ["ref1"])
        assert pooled["ref1"]["wins"] == 2  # both tasks judged after production

    def test_progress_callback_reports_each_unit(self, tmp_path: Path) -> None:
        _make_cache(tmp_path, ["a", "b"], repeats=("repeat_0", "repeat_1"))

        def fake_verify_one(task_id, deliverables_dir, prompt, reference_ids):
            return {"per_reference": {"ref1": {"wins": 1, "losses": 0, "ties": 0, "reference_elo": 1000.0}}}

        seen = []
        judge = build_judge_stage(
            fake_verify_one, tmp_path, {}, progress=lambda done, total, tid: seen.append((done, total, tid))
        )
        judge(["a", "b"], ["ref1"])
        # 2 tasks x 2 repeats = 4 units; progress reports running done/total.
        assert [s[0] for s in seen] == [1, 2, 3, 4]
        assert all(s[1] == 4 for s in seen)


class TestRunMultistageElo:
    def test_requires_eval_dir(self, tmp_path: Path) -> None:
        cfg = MultiStageEloConfig(distribution_path="x.json", stages=[StageSpec(1)], reference_elos={"a": 1000.0})
        with pytest.raises(ValueError):
            run_multistage_elo(cfg, lambda *a: {}, {})

    def test_end_to_end_with_fakes(self, tmp_path: Path) -> None:
        # 30 cached tasks, 2-stage adaptive run with a fake judge.
        task_ids = [f"t{i}" for i in range(30)]
        _make_cache(tmp_path, task_ids)
        dist_path = tmp_path / "dist.json"
        dist_path.write_text(json.dumps(_dist({"x": task_ids})))

        def fake_verify_one(task_id, deliverables_dir, prompt, reference_ids):
            return {
                "per_reference": {
                    rid: {"wins": 7, "losses": 3, "ties": 0, "reference_elo": elo}
                    for rid, elo in {"a": 1000.0, "b": 1200.0, "c": 1300.0, "d": 1500.0}.items()
                    if rid in reference_ids
                }
            }

        cfg = MultiStageEloConfig(
            distribution_path=str(dist_path),
            stages=[StageSpec(num_tasks=5, num_models=None), StageSpec(num_tasks=12, num_models=2)],
            reference_elos={"a": 1000.0, "b": 1200.0, "c": 1300.0, "d": 1500.0},
            eval_deliverables_dir=str(tmp_path),
        )
        results = run_multistage_elo(cfg, fake_verify_one, {t: "" for t in task_ids}, rng=random.Random(0))

        assert len(results) == 2
        assert results[0].reference_ids == ["a", "b", "c", "d"]
        assert len(results[1].reference_ids) == 2
        assert results[1].eval_elo is not None

        summary = stage_results_to_dict(results)
        assert summary["num_stages"] == 2
        assert summary["final_eval_elo"] == results[1].eval_elo

    def test_stage_results_to_dict_empty(self) -> None:
        assert stage_results_to_dict([])["final_eval_elo"] is None


class TestParseStage:
    def test_tasks_only(self) -> None:
        s = _parse_stage("5")
        assert (s.num_tasks, s.num_models, s.seed) == (5, None, None)

    def test_tasks_and_models(self) -> None:
        s = _parse_stage("88:4")
        assert (s.num_tasks, s.num_models, s.seed) == (88, 4, None)

    def test_all_models_keyword_and_seed(self) -> None:
        s = _parse_stage("5:all:7")
        assert (s.num_tasks, s.num_models, s.seed) == (5, None, 7)

    @pytest.mark.parametrize("bad", ["", "x", "5:y", "5:4:z"])
    def test_invalid(self, bad: str) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_stage(bad)


class TestLoadReferenceElos:
    def test_inline_json(self) -> None:
        assert _load_reference_elos('{"a": 1500, "b": 1200}') == {"a": 1500.0, "b": 1200.0}

    def test_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "refs.json"
        f.write_text(json.dumps({"a": 1000}))
        assert _load_reference_elos(f"@{f}") == {"a": 1000.0}

    @pytest.mark.parametrize("bad", ["[]", "{}", '"x"'])
    def test_invalid(self, bad: str) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _load_reference_elos(bad)


class TestCliMain:
    def _setup(self, tmp_path: Path):
        _make_cache(tmp_path, ["a", "b"])
        prompts = tmp_path / "bench.jsonl"
        prompts.write_text(json.dumps({"task_id": "a", "prompt": "p"}) + "\n")
        refs = tmp_path / "refs.json"
        refs.write_text(json.dumps({"a": 1000.0, "b": 1200.0}))
        return prompts, refs

    def test_main_writes_summary(self, tmp_path: Path, monkeypatch, capsys) -> None:
        prompts, refs = self._setup(tmp_path)
        captured = {}

        def fake_run(config, verify_one, task_prompts, *, rng=None, producer=None, on_event=None, progress=None):
            captured["config"] = config
            captured["rng"] = rng
            return [
                StageResult(
                    stage_index=0,
                    task_ids=["a"],
                    reference_ids=["a", "b"],
                    per_reference={},
                    eval_elo=1234.0,
                    normalized_elo=0.5,
                    num_references=2,
                )
            ]

        monkeypatch.setattr(driver, "run_multistage_elo", fake_run)
        out = tmp_path / "summary.json"
        rc = main(
            [
                "--server-url",
                "http://localhost:9999",
                "--eval-deliverables-dir",
                str(tmp_path),
                "--reference-elos",
                f"@{refs}",
                "--stage",
                "5",
                "--stage",
                "12:1",
                "--task-prompts",
                str(prompts),
                "--nested-tasks",
                "--skip-missing",
                "--seed",
                "3",
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        summary = json.loads(out.read_text())
        assert summary["final_eval_elo"] == 1234.0
        cfg = captured["config"]
        assert [s.num_tasks for s in cfg.stages] == [5, 12]
        assert cfg.stages[1].num_models == 1
        assert cfg.nested_tasks is True
        assert cfg.produce_missing is False
        assert cfg.reference_elos == {"a": 1000.0, "b": 1200.0}
        assert isinstance(captured["rng"], random.Random)

    def test_main_to_stdout(self, tmp_path: Path, monkeypatch, capsys) -> None:
        prompts, refs = self._setup(tmp_path)
        monkeypatch.setattr(driver, "run_multistage_elo", lambda *a, **k: [])
        rc = main(
            [
                "--server-url",
                "http://localhost:9999",
                "--eval-deliverables-dir",
                str(tmp_path),
                "--reference-elos",
                f"@{refs}",
                "--stage",
                "5",
                "--task-prompts",
                str(prompts),
            ]
        )
        assert rc == 0
        assert json.loads(capsys.readouterr().out)["num_stages"] == 0

    def test_main_missing_eval_dir(self, tmp_path: Path, capsys) -> None:
        _, refs = self._setup(tmp_path)
        rc = main(
            [
                "--server-url",
                "http://x",
                "--eval-deliverables-dir",
                str(tmp_path / "nope"),
                "--reference-elos",
                f"@{refs}",
                "--stage",
                "5",
            ]
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err.lower()

    def test_main_missing_prompts(self, tmp_path: Path, capsys) -> None:
        _, refs = self._setup(tmp_path)
        rc = main(
            [
                "--server-url",
                "http://x",
                "--eval-deliverables-dir",
                str(tmp_path),
                "--reference-elos",
                f"@{refs}",
                "--stage",
                "5",
                "--task-prompts",
                str(tmp_path / "nope.jsonl"),
            ]
        )
        assert rc == 2
        assert "not found" in capsys.readouterr().err.lower()
