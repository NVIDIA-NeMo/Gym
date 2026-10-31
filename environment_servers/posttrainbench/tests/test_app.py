# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from environment_servers.harbor.app import HarborEnvironmentServer
from environment_servers.harbor.tests.conftest import make_client
from environment_servers.posttrainbench.app import (
    JUDGING_INCONCLUSIVE_KIND,
    MISSING_KEY,
    SCORE_KEY,
    PostTrainBenchEnvironmentServer,
    PostTrainBenchEnvironmentServerConfig,
    weighted_score,
)
from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_collection import NG_TASK_ID_KEY


# Upstream's factors.json at the pinned revision.
FACTORS = {
    "aime2025": 0.24480059495473774,
    "arenahardwriting": 0.09763625841479416,
    "gpqamain": 0.242731857532585,
    "gsm8k": 0.10113359431088526,
    "healthbench": 0.1989911302939207,
    "humaneval": 0.11470656449307715,
}
MODELS = ("qwen3-1.7b", "qwen3-4b", "smollm3-3b", "gemma3-4b")
TASKS = [f"posttrainbench-{benchmark}-{model}" for benchmark in FACTORS for model in MODELS]
BASELINES = {task: 0.01 * index for index, task in enumerate(TASKS)}


def rollout(task: str, reward: float, repeat: int = 0, **fields: float) -> dict:
    """A record as the collector sends it to /aggregate_metrics: the TaskId only as _ng_task_id."""
    return {
        NG_TASK_ID_KEY: {"taskset": "posttrainbench", "task_id": task},
        "reward": reward,
        "verifier_rewards": {"reward": reward, **fields},
        ROLLOUT_INDEX_KEY_NAME: repeat,
    }


def make_server(tmp_path: Path) -> PostTrainBenchEnvironmentServer:
    taskset = tmp_path / "taskset.json"
    taskset.write_text(
        json.dumps(
            {
                "taskset": "posttrainbench",
                "folder": str(tmp_path / "tasks"),
                "tasks": {task: f"sha256:{index}" for index, task in enumerate(TASKS)},
                "baselines": BASELINES,
                "factors": FACTORS,
            }
        )
    )
    config = PostTrainBenchEnvironmentServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="posttrainbench_environment_server",
        agent_server={"type": "responses_api_agents", "name": "agent"},
        taskset_file=taskset,
        artifacts_dir=tmp_path / "artifacts",
    )
    return PostTrainBenchEnvironmentServer(config=config, server_client=make_client())


def test_weighted_score_is_upstreams_aggregate() -> None:
    rewards = {task: (index % 7) / 10 for index, task in enumerate(TASKS)}
    score = weighted_score([rollout(task, value) for task, value in rewards.items()], 0, BASELINES, FACTORS)

    # scripts/aggregate.py: for each benchmark, the mean over the models times its factor, summed.
    expected = sum(
        factor * sum(rewards[f"posttrainbench-{benchmark}-{model}"] for model in MODELS) / len(MODELS)
        for benchmark, factor in FACTORS.items()
    )
    assert score[SCORE_KEY] == pytest.approx(expected)
    assert score["ptb/gsm8k"] == pytest.approx(sum(rewards[t] for t in TASKS if "-gsm8k-" in t) / 4)
    assert score["ptb/tasks_without_score"] == 0


def test_the_score_is_withheld_while_a_task_has_no_scored_rollout() -> None:
    rollouts = [rollout(task, 1.0) for task in TASKS[1:]]
    score = weighted_score(rollouts, 0, BASELINES, FACTORS)

    assert SCORE_KEY not in score
    assert score["ptb/tasks_without_score"] == 1
    assert score["ptb/aime2025"] == 1.0  # the three scored aime2025 tasks


def test_a_benchmark_without_a_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="bfcl"):
        weighted_score([], 0, {"posttrainbench-bfcl-qwen3-4b": 0.1}, FACTORS)


def test_server_loads_the_prepared_taskset_and_scores_each_repeat(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    assert set(server.config.tasksets["posttrainbench"].tasks) == set(TASKS)

    first = [[rollout(task, 1.0, 0, judge_flagged=0.0)] for task in TASKS]
    second = [[rollout(task, 0.0, 1, judge_flagged=1.0)] for task in TASKS]
    metrics = server.compute_metrics([a + b for a, b in zip(first, second)])

    assert metrics[SCORE_KEY] == pytest.approx(sum(FACTORS.values()) / 2)
    assert metrics["ptb/judge_flagged"] == len(TASKS)
    assert server.compute_repeat_metrics(second) == {SCORE_KEY: 0.0}
    assert server.get_key_metrics({SCORE_KEY: 0.5, "mean/reward": 0.4, "ptb/gsm8k": 0.3, MISSING_KEY: 0}) == {
        SCORE_KEY: 0.5,
        "mean/reward": 0.4,
        MISSING_KEY: 0,
    }


def test_a_repeat_with_an_unscored_task_withholds_the_score(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    complete = [[rollout(task, 1.0, 0), rollout(task, 1.0, 1)] for task in TASKS]
    complete[0] = complete[0][:1]  # repeat 1 of the first task was masked

    metrics = server.compute_metrics(complete)

    assert SCORE_KEY not in metrics
    assert metrics[MISSING_KEY] == 1
    assert server.compute_repeat_metrics([[rollout(task, 1.0, 1)] for task in TASKS[1:]]) == {}


@pytest.mark.asyncio
async def test_a_wholly_masked_repeat_withholds_the_score_on_the_aggregate_path(tmp_path: Path) -> None:
    server = make_server(tmp_path)

    def row(index: int, task: str, repeat: int, masked: bool) -> dict:
        return rollout(task, 0.0 if masked else 0.8, repeat) | {TASK_INDEX_KEY_NAME: index, "mask_sample": masked}

    responses = [row(i, task, 0, False) for i, task in enumerate(TASKS)]
    responses += [row(i, task, 1, True) for i, task in enumerate(TASKS)]

    metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))

    assert SCORE_KEY not in metrics.agent_metrics and SCORE_KEY not in metrics.key_metrics
    assert metrics.key_metrics[MISSING_KEY] == len(TASKS)


def test_every_rollout_masked_withholds_the_score(tmp_path: Path) -> None:
    metrics = make_server(tmp_path).compute_metrics([])

    assert SCORE_KEY not in metrics
    assert metrics[MISSING_KEY] == len(TASKS)


@pytest.mark.parametrize("fields", [{"judgement_missing": 1.0}, {"judgement_unreadable": 1.0}])
def test_unfinished_judging_is_inconclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict) -> None:
    server = make_server(tmp_path)
    rewards = {"reward": 0.1, "baseline_accuracy": 0.1, "judgement_missing": 0.0, **fields}
    measured = {"reward": 0.1, "verifier_rewards": rewards}
    monkeypatch.setattr(HarborEnvironmentServer, "_score", lambda *args, **kwargs: dict(measured))

    scored = server._score(SimpleNamespace(task_id=TASKS[0]), None, None, tmp_path, 0.0, rewards=None)

    assert scored["mask_sample"] is True
    assert scored["failure_kind"] == JUDGING_INCONCLUSIVE_KIND
    assert scored["verifier_rewards"] == measured["verifier_rewards"]


def test_a_judged_episode_keeps_its_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = make_server(tmp_path)
    measured = {"reward": 0.6, "verifier_rewards": {"reward": 0.6, "baseline_accuracy": 0.1, "judgement_missing": 0.0}}
    monkeypatch.setattr(HarborEnvironmentServer, "_score", lambda *args, **kwargs: dict(measured))

    assert server._score(SimpleNamespace(task_id=TASKS[0]), None, None, tmp_path, 0.0, rewards=None) == measured


# Upstream's verdict readers, reduced to the files gym_score.py consults (scripts/utils.py names and fields).
UPSTREAM_UTILS = """
import json, os

def _read(logs, name):
    path = os.path.join(logs, name)
    return json.load(open(path)) if os.path.exists(path) else None

def load_judgement(logs):
    verdict = _read(logs, "judgement_gpt5_4.json")
    if verdict is None:
        raise FileNotFoundError(logs)
    return verdict

def load_api_judgement(logs):
    verdict = _read(logs, "judgement_api.json")
    return None if verdict is None else verdict["disallowed_api_usage"]

def load_ptb_lookup_judgement(logs):
    verdict = _read(logs, "judgement_ptb_lookup.json")
    return None if verdict is None else verdict["disallowed_ptb_lookup"]

def judgement_to_cell(judgement, api_usage):
    return "M" * judgement["disallowed_model"] + "C" * judgement["contamination"] + "A" * bool(api_usage)
"""


def task_folder(tmp_path: Path) -> Path:
    tests = tmp_path / "task" / "tests"
    (tests / "ptb" / "scripts").mkdir(parents=True)
    (tests / "ptb" / "scripts" / "utils.py").write_text(UPSTREAM_UTILS)
    gym_score = Path(__file__).resolve().parents[3] / "benchmarks" / "posttrainbench" / "gym_score.py"
    (tests / "gym_score.py").write_text(gym_score.read_text())
    return tmp_path / "task"


@pytest.mark.parametrize("judged, masked", [(True, False), (False, True)])
def test_a_verifier_that_stopped_before_its_reward_is_scored_from_its_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, judged: bool, masked: bool
) -> None:
    server = make_server(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    if judged:  # the judges finished; the final evaluation ran out of time and left no metrics.json
        (logs / "judgement_gpt5_4.json").write_text('{"contamination": false, "disallowed_model": false}')
        (logs / "judgement_api.json").write_text('{"disallowed_api_usage": false}')
        (logs / "judgement_ptb_lookup.json").write_text('{"disallowed_ptb_lookup": false}')
    timed_out = {"reward": 0.0, "mask_sample": False, "failure_kind": "harbor:verifier_timeout", "failure_reason": "x"}
    monkeypatch.setattr(HarborEnvironmentServer, "_score", lambda *args, **kwargs: dict(timed_out))

    scored = server._score(SimpleNamespace(task_id=TASKS[0], path=task_folder(tmp_path)), None, None, logs, 0.0)

    assert scored["mask_sample"] is masked
    if not masked:
        assert scored["reward"] == BASELINES[TASKS[0]]
        assert scored["verifier_rewards"]["eval_failed"] == 1.0
    assert "utils" not in sys.modules or sys.modules["utils"].__name__ != "posttrainbench_upstream_utils"


@pytest.mark.parametrize("cap", [None, 8])
def test_the_benchmark_config_validates_with_and_without_a_concurrency_cap(tmp_path: Path, cap: int | None) -> None:
    gym = Path(__file__).resolve().parents[3]
    merged = OmegaConf.merge(
        OmegaConf.load(gym / "environment_servers/posttrainbench/configs/posttrainbench.yaml"),
        OmegaConf.load(gym / "benchmarks/posttrainbench/config.yaml"),
        {"ptb_max_concurrent_episodes": cap},
    )
    block = OmegaConf.to_container(
        merged.posttrainbench_environment_server.environment_servers.posttrainbench, resolve=True
    )

    config = PostTrainBenchEnvironmentServerConfig(host="0.0.0.0", port=8080, name="ptb", **block)

    assert config.max_concurrent_episodes == cap
    assert config.queue_timeout_seconds == 259200


def test_a_missing_taskset_file_names_the_prepare_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gym eval prepare --benchmark posttrainbench"):
        PostTrainBenchEnvironmentServer(
            config=PostTrainBenchEnvironmentServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="posttrainbench_environment_server",
                agent_server={"type": "responses_api_agents", "name": "agent"},
                taskset_file=tmp_path / "missing.json",
                artifacts_dir=tmp_path / "artifacts",
            ),
            server_client=make_client(),
        )
