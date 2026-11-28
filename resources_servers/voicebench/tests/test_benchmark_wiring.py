# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``benchmarks/voicebench`` config drives this server: it validates, finds its prompts and scores its rows."""

import asyncio
from pathlib import Path

import pytest
import yaml

from resources_servers.voicebench.app import VoiceBenchConfig, VoiceBenchServer
from resources_servers.voicebench.tests.test_app import candidate, request
from resources_servers.voicebench.tests.test_judge import client


ROOT = Path(__file__).resolve().parents[3]
SERVER_DEFAULTS = yaml.safe_load((ROOT / "resources_servers/voicebench/configs/voicebench.yaml").read_text())
BENCHMARK = yaml.safe_load((ROOT / "benchmarks/voicebench/config.yaml").read_text())


def benchmark_settings() -> dict:
    """The server settings of the benchmark: the server's defaults with the benchmark's overrides on top."""
    settings = SERVER_DEFAULTS["voicebench"]["resources_servers"]["voicebench"]
    settings = settings | BENCHMARK["voicebench_benchmark_resources_server"]["resources_servers"]["voicebench"]
    return {
        key: value for key, value in settings.items() if key in VoiceBenchConfig.model_fields and key != "entrypoint"
    }


def benchmark_server(server_client, **overrides) -> VoiceBenchServer:
    config = VoiceBenchConfig(
        name="voicebench_benchmark_resources_server",
        host="127.0.0.1",
        port=8105,
        entrypoint="app.py",
        **(benchmark_settings() | overrides),
    )
    return VoiceBenchServer(config=config, server_client=server_client)


def test_the_benchmark_settings_are_a_valid_all_subsets_config() -> None:
    config = benchmark_server(client()).config
    assert config.subset == "all" and config.scoring_type is None and config.expected_num_rows is None
    assert len(config.subsets) == 11 and sum(config.expected_rows.values()) == 20_508
    assert config.answer_extraction == "message" and config.sdqa_pedant is True


def test_the_benchmark_prompts_are_found_and_hashed_by_both_judges() -> None:
    server = benchmark_server(client())
    assert set(server._judges) == {"open", "reference_qa"}
    open_prompt, qa_prompt = server._judges["open"].prompt, server._judges["reference_qa"].prompt
    assert "[Instruction]: {prompt}" in open_prompt.user and "{reference}" in qa_prompt.user
    assert server._judges["open"].prompt_sha256 != server._judges["reference_qa"].prompt_sha256


def test_a_benchmark_without_a_judge_prompt_for_a_judged_subset_is_refused() -> None:
    prompts = benchmark_settings()["judge_prompt_files"]
    with pytest.raises(ValueError, match="judge scoring requires"):
        benchmark_server(client(), judge_prompt_files={"open": prompts["open"]})


def test_a_rule_row_and_a_judged_row_are_scored_through_the_benchmark_runtime() -> None:
    server_client = client(("4", "4", "4"))
    server = benchmark_server(server_client)
    mcq = asyncio.run(
        server.verify(request(subset="openbookqa", response=candidate("The answer is: C"), reference="C"))
    )
    assert mcq.reward == 1 and mcq.subset == "openbookqa" and mcq.scoring_type == "mcq"
    server_client.post.assert_not_called()

    judged = asyncio.run(server.verify(request(subset="alpacaeval", response=candidate("a fine answer"))))
    assert judged.scoring_type == "open" and judged.score_valid and judged.voicebench_score == 4
    server_client.post.assert_called_once()


def test_an_unaggregatable_subset_does_not_take_the_others_down() -> None:
    """One unparseable judge grade is that subset's ``aggregation_error``; the other subsets keep their metrics."""
    server = benchmark_server(client(("garbage", "garbage", "garbage")))
    ok = asyncio.run(
        server.verify(request(subset="openbookqa", response=candidate("The answer is: C"), reference="C"))
    ).model_dump()
    bad = asyncio.run(server.verify(request(subset="alpacaeval", response=candidate("an answer")))).model_dump()
    server.config.expected_rows = {"openbookqa": 1, "alpacaeval": 1}
    metrics = server.compute_metrics([[ok], [bad]])
    assert "voicebench/alpacaeval/aggregation_error" in metrics
    assert metrics["voicebench/openbookqa/legacy/acc"] == 100.0
    assert not any(key.startswith("voicebench/alpacaeval/legacy") for key in metrics)
