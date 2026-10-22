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
"""Tests for the RNA design agent.

``run()`` is exercised for real: it starts the worker, which runs the campaign and scores the pool with
EternaFold. Tests that need EternaFold skip when ``arnie`` is not configured."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.rna_design_agent import prepare
from responses_api_agents.rna_design_agent.app import (
    RnaDesignAgent,
    RnaDesignAgentConfig,
    RnaDesignRunRequest,
)


AGENT_DIR = Path(__file__).resolve().parent.parent


def _have_eternafold() -> bool:
    import os

    try:
        import arnie  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ARNIEFILE")) and Path(os.environ["ARNIEFILE"]).is_file()


needs_eternafold = pytest.mark.skipif(not _have_eternafold(), reason="EternaFold and arnie are not configured")


def _config(tmp_path: Path, **overrides) -> RnaDesignAgentConfig:
    values = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="rna_design_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        arm="fixed",
        generator="generate.inverse_fold",
        work_dir=str(tmp_path),
        rollout_timeout_s=300,
    )
    values.update(overrides)
    return RnaDesignAgentConfig(**values)


@pytest.fixture(autouse=True)
def _no_model_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Gym model server exists in these tests. The model-free arm never calls it, and tests that need a model
    set `RnaDesignAgent.stub_url` to a local server."""
    monkeypatch.setattr(
        RnaDesignAgent,
        "resolve_model_base_url",
        lambda self, name, rollout_id=None: getattr(RnaDesignAgent, "stub_url", "http://model.invalid/v1"),
        raising=False,
    )


def _agent(tmp_path: Path, **overrides) -> RnaDesignAgent:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return RnaDesignAgent(config=_config(tmp_path, **overrides), server_client=client)


def _request(suite: str, target_name: str) -> RnaDesignRunRequest:
    row = next(r for r in prepare.build_rows(suite) if r["target"]["name"] == target_name)
    return RnaDesignRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(**row["responses_create_params"]),
        suite=row["suite"],
        target=row["target"],
    )


class _ScriptedModelServer:
    """A local OpenAI-style endpoint whose model plays a fixed list of (tool, arguments) calls, then stops."""

    def __init__(self, script: list[tuple[str, dict]] = ()) -> None:
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        calls = iter(script)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                name, arguments = next(calls, ("stop", {"reason": "done"}))
                call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                body = {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                }
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()


class TestConfig:
    def test_builds(self, tmp_path: Path) -> None:
        _agent(tmp_path)

    def test_unknown_arm_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="arm must be one of"):
            _agent(tmp_path, arm="nonsense")

    def test_unknown_suite_cannot_become_a_path(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        with pytest.raises(ValueError, match="unknown suite"):
            agent._suite_path("../../etc/passwd")
        assert agent._suite_path("rnainvbench_eterna100_v1").is_file()


class TestWorkerPython:
    def test_unset_worker_python_without_torch_warns_at_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without it every gRNAde call fails with no torch, so the cause is named before any rollout runs."""
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with caplog.at_level("WARNING"):
            _agent(tmp_path)
        assert "RNA_WORKER_PYTHON" in caplog.text

    def test_a_configured_worker_python_does_not_warn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with caplog.at_level("WARNING"):
            _agent(tmp_path, worker_python=sys.executable)
        assert "RNA_WORKER_PYTHON" not in caplog.text

    def test_the_shipped_config_leaves_it_to_the_environment(self) -> None:
        text = (AGENT_DIR / "configs" / "rna_design_agent.yaml").read_text()
        assert "worker_python: ${oc.env:RNA_WORKER_PYTHON,null}" in text


class TestPrepare:
    def test_every_suite_expands(self) -> None:
        for path in sorted((AGENT_DIR / "suites").glob("*.yaml")):
            rows = prepare.build_rows(path.stem)
            assert rows, path.stem
            for row in rows:
                assert row["suite"] == path.stem
                assert len(row["target"]["target_structure"]) == row["target"]["length"]

    def test_limit_is_seeded(self) -> None:
        a = prepare.build_rows("rnainvbench_eterna100_v1", limit=5, seed=1)
        b = prepare.build_rows("rnainvbench_eterna100_v1", limit=5, seed=1)
        c = prepare.build_rows("rnainvbench_eterna100_v1", limit=5, seed=2)
        assert [r["target"]["name"] for r in a] == [r["target"]["name"] for r in b]
        assert [r["target"]["name"] for r in a] != [r["target"]["name"] for r in c]


class TestRun:
    @needs_eternafold
    @pytest.mark.asyncio
    async def test_a_solvable_target_earns_the_reward(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path)
        out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_57"))

        assert out.mask_sample is False
        assert out.reward == 1.0 and out.solved is True
        assert out.best == 1.0
        # The reward comes from re-scoring the whole pool, so the pool is what the run built.
        assert out.pool_size == 400 and out.n_passing_gates == 400
        assert out.protocol and out.metric == "score.fold_match.match.f1"
        assert out.tool_failures == 0
        assert out.code

    @needs_eternafold
    @pytest.mark.asyncio
    async def test_a_target_the_run_cannot_solve_scores_zero_but_is_kept(self, tmp_path: Path) -> None:
        """A real failure is a measured 0, not a masked sample: the design was bad, the system worked."""
        agent = _agent(tmp_path)
        out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_121"))

        assert out.mask_sample is False
        assert out.reward == 0.0 and out.solved is False
        assert out.best is not None and 0.0 < out.best < out.pass_threshold

    @needs_eternafold
    @pytest.mark.asyncio
    async def test_a_model_that_designs_nothing_scores_zero_without_being_masked(self, tmp_path: Path) -> None:
        """An empty pool has no `top`, and must score 0, not crash."""
        stub = _ScriptedModelServer()
        try:
            agent = _agent(tmp_path, arm="agent")
            out = await agent._run_worker(
                {
                    "suite_path": str(agent._suite_path("rnainvbench_eterna100_v1")),
                    "target_name": "eterna100_v1_57",
                    "run_dir": str(tmp_path / "empty"),
                    "arm": "agent",
                    "model": "stub",
                    "base_url": stub.base_url,
                    "seed": 1,
                }
            )
        finally:
            stub.close()

        assert out["ok"] is True, out
        result = out["result"]
        assert result["pool_size"] == 0 and result["best"] is None
        assert result["solved"] is False and result["top"] == []
        assert result["stop_reason"] == "controller_stop"
        assert (result["tokens_in"], result["tokens_out"]) == (100, 10)

    @needs_eternafold
    @pytest.mark.asyncio
    async def test_the_response_is_a_transcript_of_the_campaign(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gym replaces response.output with the captured calls in training, but plain eval keeps it."""
        stub = _ScriptedModelServer([("fold_eternafold", {"sequences": ["GGGGAAAACCCC"]})])
        agent = _agent(tmp_path, arm="agent", model_name="stub")
        monkeypatch.setattr(RnaDesignAgent, "stub_url", stub.base_url, raising=False)
        try:
            out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_57"))
        finally:
            stub.close()

        assert out.mask_sample is False
        text = out.response.output[0].content[0].text
        assert "round 1: fold_eternafold" in text and "GGGGAAAACCCC" in text
        assert "reasoning" not in text and "<think>" not in text

    @needs_eternafold
    @pytest.mark.asyncio
    async def test_a_run_where_every_tool_call_failed_is_masked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool that cannot run (a missing install) says nothing about the model, so the sample is masked."""
        stub = _ScriptedModelServer([("generate_grnade", {"target": "eterna100_v1_57", "n": 5})] * 2)
        agent = _agent(tmp_path, arm="agent", model_name="stub")
        monkeypatch.setattr(RnaDesignAgent, "stub_url", stub.base_url, raising=False)
        try:
            out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_57"))
        finally:
            stub.close()

        assert out.tool_calls == 2 and out.tool_failures == 2 and out.pool_size == 0
        assert out.mask_sample is True and out.failure_kind == "agent_run_error"
        assert "failed: FileNotFoundError" in out.response.output[0].content[0].text

    @pytest.mark.asyncio
    async def test_the_worker_runs_under_the_configured_python(self, tmp_path: Path) -> None:
        """gRNAde and the pseudoknot scorer need torch, which the agent server's own Python does not have."""
        marker = tmp_path / "ran_under"
        launcher = tmp_path / "python_wrapper"
        launcher.write_text(f'#!/bin/sh\necho "$0" > {marker}\nexec {sys.executable} "$@"\n')
        launcher.chmod(0o755)
        agent = _agent(tmp_path, worker_python=str(launcher))
        out = await agent._run_worker(
            {
                "suite_path": str(agent._suite_path("rnainvbench_eterna100_v1")),
                "target_name": "no_such_target",
                "run_dir": str(tmp_path / "run"),
                "arm": "fixed",
                "seed": 1,
            }
        )
        # The target does not exist, so the worker reports an error as data. What matters is who ran it.
        assert marker.read_text().strip() == str(launcher)
        assert out["ok"] is False and "no_such_target" in out["error"]

    @pytest.mark.asyncio
    async def test_an_infrastructure_fault_is_masked_not_scored_as_a_failure(self, tmp_path: Path) -> None:
        """Pseudoknot suites need torch and RibonanzaNet weights. Without them the sample is invalid."""
        try:
            import torch  # noqa: F401

            pytest.skip("torch is installed, so this fault cannot be reproduced here")
        except ImportError:
            pass
        agent = _agent(tmp_path)
        out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_anta_pseudo_test", "anta_pseudo-test_56"))

        assert out.mask_sample is True
        assert out.failure_kind == "agent_run_error"
        assert out.reward == 0.0
        assert out.failure_reason

    @pytest.mark.asyncio
    async def test_a_temp_run_dir_is_removed_after_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no work_dir the run directory is scratch, removed even when the run is cut short."""
        import tempfile

        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        agent = _agent(tmp_path, work_dir="", rollout_timeout_s=0.05)
        await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_57"))
        assert list((tmp_path / "rna_design_agent").iterdir()) == []

    @pytest.mark.asyncio
    async def test_a_timeout_is_masked_with_its_own_kind(self, tmp_path: Path) -> None:
        agent = _agent(tmp_path, rollout_timeout_s=0.05)
        out = await agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", "eterna100_v1_57"))

        assert out.mask_sample is True
        assert out.failure_kind == "agent_timeout"

    @pytest.mark.asyncio
    async def test_concurrent_rollouts_do_not_share_state(self, tmp_path: Path) -> None:
        """rds sets process-wide environment variables, so concurrency relies on one process per rollout."""
        import asyncio

        if not _have_eternafold():
            pytest.skip("EternaFold and arnie are not configured")
        agent = _agent(tmp_path, max_concurrent_rollouts=4)
        names = ["eterna100_v1_57", "eterna100_v1_50", "eterna100_v1_57", "eterna100_v1_50"]
        outs = await asyncio.gather(
            *[agent.run(MagicMock(cookies={}), _request("rnainvbench_eterna100_v1", n)) for n in names]
        )

        assert [o.target_name for o in outs] == names
        assert all(o.solved and not o.mask_sample for o in outs)
        run_dirs = {p.name for p in tmp_path.iterdir()}
        assert len(run_dirs) == 4, "each rollout must get its own run directory"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
class TestPygShims:
    """The pure PyTorch stand-ins for the one function gRNAde uses from torch_scatter and torch_cluster."""

    def _load(self, name: str):
        path = AGENT_DIR / "rds" / "tools" / "pyg_shims" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"shim_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_scatter_add_counts_destinations(self) -> None:
        import torch

        scatter_add = self._load("torch_scatter").scatter_add
        dst = torch.tensor([0, 2, 2, 1, 2])
        assert scatter_add(torch.ones_like(dst), dst, dim_size=4).tolist() == [1, 1, 3, 0]

    def test_scatter_add_sums_rows(self) -> None:
        import torch

        scatter_add = self._load("torch_scatter").scatter_add
        src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = scatter_add(src, torch.tensor([1, 0, 1]), dim=0, dim_size=2)
        assert out.tolist() == [[3.0, 4.0], [6.0, 8.0]]

    @pytest.mark.parametrize("n,k", [(20, 5), (12, 32), (40, 8)])
    def test_knn_graph_matches_brute_force(self, n: int, k: int) -> None:
        import torch

        knn_graph = self._load("torch_cluster").knn_graph
        x = torch.randn(n, 3, generator=torch.Generator().manual_seed(n))
        distances = torch.cdist(x, x)
        expected = set()
        for query in range(n):
            others = sorted((j for j in range(n) if j != query), key=lambda j: distances[query, j].item())
            expected |= {(j, query) for j in others[:k]}
        edges = knn_graph(x, k)
        assert set(map(tuple, edges.t().tolist())) == expected
        assert edges.shape[1] == n * min(k, n - 1)


class TestGrnadeInterpreter:
    def test_design_runs_under_the_workers_python_not_the_first_python3_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under the Gym server the first python3 on PATH has no torch, but the worker's own interpreter does."""
        import json
        import subprocess

        fake = tmp_path / "bin"
        fake.mkdir()
        (fake / "python3").write_text("#!/bin/sh\necho 'wrong python3 was used' >&2\nexit 99\n")
        (fake / "python3").chmod(0o755)
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "design.py").write_text("import sys; print('design.py ran under', sys.executable)\n")
        monkeypatch.setenv("PATH", f"{fake}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("GRNADE_REPO", str(repo))
        script = (
            "import json, sys; sys.path.insert(0, %r)\n"
            "from rds.registry import dispatch\n"
            "from rds.targets import from_dict, register\n"
            "register(from_dict({'name': 't', 'length': 4, 'target_structure': '(..)', 'provides': ['secondary_structure']}))\n"
            "try:\n"
            "    dispatch('generate.grnade', {'target': 't', 'n': 2, 'batch_size': 2, 'mode': '2d', 'timeout_s': 60})\n"
            "except Exception as e:\n"
            "    print(json.dumps(str(e)))\n"
        ) % str(AGENT_DIR)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True).stdout
        message = json.loads(out.strip().splitlines()[-1])
        assert "wrong python3 was used" not in message


@needs_eternafold
class TestFoldEngine:
    """ViennaRNA is not installed, so a model that asks for it must get a clear answer, not an import error."""

    def _register(self) -> None:
        from responses_api_agents.rna_design_agent.rds.targets import from_dict, register

        register(from_dict({"name": "t", "length": 12, "target_structure": "((((....))))"}))

    def test_vienna_is_refused_with_the_reason(self) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import dispatch

        self._register()
        with pytest.raises(ValueError, match="EternaFold only") as error:
            dispatch("score.fold_match", {"sequences": ["GGGGAAAACCCC"], "target": "t", "engine": "vienna"})
        assert "No module named" not in str(error.value)

    def test_eternafold_still_scores(self) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import dispatch

        self._register()
        result = dispatch("score.fold_match", {"sequences": ["GGGGAAAACCCC"], "target": "t"})
        assert result["results"][0]["match"]["f1"] == 1.0

    def test_the_removed_tool_is_gone(self) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import all_tools

        assert "fold.vienna" not in {tool.name for tool in all_tools()}


class TestCrossedPairScore:
    """crossed_pair_f1 ranks the ArchiveII and anta pseudo suites, so a wrong crossing pair must cost."""

    def _compare(self, predicted: str, target: str) -> dict:
        from responses_api_agents.rna_design_agent.rds.structure import compare

        return compare(predicted, target)

    def test_an_exact_pseudoknot_scores_one(self) -> None:
        assert self._compare("([..)]..{}", "([..)]..{}")["crossed_pair_f1"] == 1.0

    def test_a_crossing_pair_the_target_lacks_costs_precision(self) -> None:
        # The prediction has both of the target's crossing pairs and adds a third that crosses them.
        result = self._compare("([{.)]}.{}", "([..)]..{}")
        assert result["crossed_pair_f1"] == pytest.approx(0.8)
        assert result["f1"] < 1.0

    def test_a_nested_target_has_no_crossing_score(self) -> None:
        assert self._compare("((..))", "((..))")["crossed_pair_f1"] is None


@needs_eternafold
class TestFreeEnergyGate:
    """A target can set max_free_energy, so score.constraints must fail designs more stable than that."""

    def _constraints(self, sequence: str) -> dict:
        from responses_api_agents.rna_design_agent.rds.registry import dispatch
        from responses_api_agents.rna_design_agent.rds.targets import from_dict, register

        hairpin = {
            "name": "dg_gate_hairpin",
            "length": 16,
            "target_structure": "((((((....))))))",
            "max_free_energy": -5.0,
        }
        target = register(from_dict(hairpin))
        return dispatch("score.constraints", {"sequences": [sequence], "target": target.name})["results"][0]

    def test_a_design_below_the_limit_fails_with_the_reason(self) -> None:
        result = self._constraints("GGGGGGAAAACCCCCC")
        assert result["passed"] is False
        assert any("below the limit -5.0" in violation for violation in result["violations"])

    def test_a_design_that_meets_the_limit_still_passes(self) -> None:
        assert self._constraints("AAAAAAAAAAAAAAAA")["passed"] is True


class TestUpstreamFixes:
    """Regression tests for fixes taken from the upstream harness."""

    def test_a_minimised_metric_lists_the_lowest_value_first(self, tmp_path: Path) -> None:
        from responses_api_agents.rna_design_agent.rds.state import CampaignState

        state = CampaignState(tmp_path / "min", "min")
        state.add_candidates(["AAAA", "CCCC", "GGGG"], "test")
        state.score({"AAAA": {"m": 3.0}, "CCCC": {"m": 1.0}, "GGGG": {"m": 2.0}})
        assert state.select("@top:2", rank_by="m") == ["CCCC", "GGGG"]
        state.close()

    def test_a_maximised_metric_still_lists_the_highest_value_first(self, tmp_path: Path) -> None:
        from responses_api_agents.rna_design_agent.rds.state import CampaignState

        state = CampaignState(tmp_path / "max")
        state.add_candidates(["AAAA", "CCCC", "GGGG"], "test")
        state.score({"AAAA": {"m": 3.0}, "CCCC": {"m": 1.0}, "GGGG": {"m": 2.0}})
        assert state.select("@top:2", rank_by="m") == ["AAAA", "GGGG"]
        state.close()

    def test_reflections_stop_once_the_allowance_is_spent(self, tmp_path: Path) -> None:
        from responses_api_agents.rna_design_agent.rds.budget import Budget, Limits
        from responses_api_agents.rna_design_agent.rds.runner import Action, run_campaign
        from responses_api_agents.rna_design_agent.rds.state import CampaignState

        class Recorder:
            name = "recorder"

            def __init__(self) -> None:
                self.reflecting = []

            def next_action(self, view):
                self.reflecting.append("Reflection." in str(view.objective.get("this_round", "")))
                return Action("stop", {}) if len(self.reflecting) >= 5 else Action("report.rules", {"rules": []})

        controller = Recorder()
        state = CampaignState(tmp_path / "reflect")
        run_campaign(controller, state, Budget(Limits(reflections=2)), rank_by="m", max_rounds=10, reflect_every=1)
        state.close()
        assert controller.reflecting[:5] == [True, True, False, False, False]

    def test_a_gate_on_a_scorer_with_no_target_does_not_raise(self) -> None:
        from responses_api_agents.rna_design_agent.rds import evaluate
        from responses_api_agents.rna_design_agent.rds.campaign import Campaign, Scoring

        seen = {}

        def fake_dispatch(tool, arguments):
            seen[tool] = arguments
            return {"results": [{"sequence": "GGGG", "passed": True, "match": {"f1": 1.0}}]}

        campaign = Campaign(
            name="c",
            target="inline",
            scoring=Scoring(primary="score.fold_match.match.f1", gates={"fold.eternafold.passed": True}),
            inline_target={"name": "t", "length": 4, "target_structure": "(..)"},
        )
        original = evaluate.dispatch
        evaluate.dispatch = fake_dispatch
        try:
            evaluate.ground_truth(["GGGG"], campaign)
        finally:
            evaluate.dispatch = original
        assert "target" in seen["score.fold_match"]
        assert "target" not in seen["fold.eternafold"]

    def test_agent_code_does_not_run_unisolated_by_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import shutil

        from responses_api_agents.rna_design_agent.rds.tools.code import _sandbox_command

        monkeypatch.setattr(shutil, "which", lambda name: None)
        monkeypatch.delenv("RDS_SANDBOX", raising=False)
        script = tmp_path / "s.py"
        script.write_text("print(1)\n")
        with pytest.raises(RuntimeError, match="does not run unisolated by default"):
            _sandbox_command(script, "", tmp_path)

    def test_the_unisolated_subprocess_is_available_when_asked_for_by_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from responses_api_agents.rna_design_agent.rds.tools.code import _sandbox_command

        monkeypatch.setenv("RDS_SANDBOX", "subprocess")
        script = tmp_path / "s.py"
        script.write_text("print(1)\n")
        command, mode = _sandbox_command(script, "", tmp_path)
        assert mode == "subprocess" and str(script) in command


class TestToolArguments:
    """Arguments a model gets slightly wrong cost it a round, so the ones that cannot mean anything else
    are honoured rather than refused."""

    def _register(self) -> None:
        from responses_api_agents.rna_design_agent.rds.targets import from_dict, register

        register(from_dict({"name": "running", "length": 12, "target_structure": "((((....))))"}))

    def test_an_omitted_target_is_the_campaign_in_progress(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import get

        self._register()
        monkeypatch.setenv("RDS_TARGET", "running")
        assert get("score.constraints").call(sequences=["GGGGAAAACCCC"])["target"] == "running"

    def test_a_target_passed_to_a_tool_without_one_is_ignored(self, tmp_path: Path) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import get

        rules = [{"statement": "stems want GC", "context": "stem", "quantity": "gc_fraction", "minimum": 0.6}]
        out = get("report.rules").call(rules=rules, target="running", out_dir=str(tmp_path))
        assert out["n_rules"] == 1

    def test_a_missing_required_argument_names_the_tool(self) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import get

        with pytest.raises(TypeError, match=r"code\.run: missing required argument\(s\) code"):
            get("code.run").call(purpose="no code at all")

    def test_the_engine_is_named_not_spelled(self) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import dispatch

        self._register()
        for spelling in ("eternafold", "EternaFold", " EternaFold "):
            result = dispatch(
                "score.fold_match", {"sequences": ["GGGGAAAACCCC"], "target": "running", "engine": spelling}
            )
            assert result["results"][0]["match"]["f1"] == 1.0


class TestSavedTools:
    """A tool the model saved has to be callable from wherever the run lives."""

    def test_a_saved_tool_runs_from_a_copy_not_from_the_toolbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from responses_api_agents.rna_design_agent.rds.registry import get

        monkeypatch.setenv("RDS_TOOLBOX", str(tmp_path / "box"))
        monkeypatch.setenv("RDS_SANDBOX", "subprocess")
        box = tmp_path / "box"
        box.mkdir()
        (box / "make_one.py").write_text('# makes one sequence\nprint(\'{"sequences": ["GGGGAAAACCCC"]}\')\n')

        out = get("code.call").call(name="make_one")
        assert [row["sequence"] for row in out["results"]] == ["GGGGAAAACCCC"]
        assert out["purpose"] == "makes one sequence"


class TestGatewayRetries:
    """A shared gateway answers its own database trouble with 401, and a run is a long sequence of calls."""

    def _raising(self, body: bytes, tries: list[int]):
        import io
        import urllib.error

        def open_url(request, timeout=None):
            tries.append(1)
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(body))

        return open_url

    def test_the_gateways_own_401_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from responses_api_agents.rna_design_agent.rds.llm import GatewayProvider

        body = b'{"error":{"message":"Error in connector: FATAL: sorry, too many clients already"}}'
        tries: list[int] = []
        monkeypatch.setattr("responses_api_agents.rna_design_agent.rds.llm.time.sleep", lambda s: None)
        monkeypatch.setattr(
            "responses_api_agents.rna_design_agent.rds.llm.urllib.request.urlopen", self._raising(body, tries)
        )
        provider = GatewayProvider(model="m", base_url="http://gateway/v1", api_key="k", max_retries=4)
        with pytest.raises(RuntimeError, match="too many clients"):
            provider.chat([{"role": "user", "content": "hi"}])
        assert len(tries) == 4

    def test_a_wrong_key_still_fails_at_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from responses_api_agents.rna_design_agent.rds.llm import GatewayProvider

        tries: list[int] = []
        monkeypatch.setattr("responses_api_agents.rna_design_agent.rds.llm.time.sleep", lambda s: None)
        monkeypatch.setattr(
            "responses_api_agents.rna_design_agent.rds.llm.urllib.request.urlopen",
            self._raising(b'{"error":{"message":"Invalid API key"}}', tries),
        )
        provider = GatewayProvider(model="m", base_url="http://gateway/v1", api_key="k", max_retries=4)
        with pytest.raises(RuntimeError, match="Invalid API key"):
            provider.chat([{"role": "user", "content": "hi"}])
        assert len(tries) == 1
