# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Auto Ontology pairing of BIRD-Interact: its catalog image's inputs and the configs that run it."""

import re
from pathlib import Path

import pytest
import yaml

from benchmarks.bird_interact.auto_ontology import build_catalog


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
BUILD_IMAGE = (BENCHMARK_DIR / "auto_ontology" / "build_image.sh").read_text()


def test_the_image_builds_the_auto_ontology_that_bird_dev_pins():
    """Gym pins Auto Ontology once, in the BIRD dev image's build script; this one reads that pin."""
    assert not re.search(r"^AUTO_ONTOLOGY_(REPOSITORY|COMMIT)=", BUILD_IMAGE, re.MULTILINE)
    assert "benchmarks/birdbench/auto_ontology/build_image.sh" in BUILD_IMAGE
    bird_dev = (BENCHMARK_DIR.parent / "birdbench" / "auto_ontology" / "build_image.sh").read_text()
    assert re.search(r"^AUTO_ONTOLOGY_COMMIT=[0-9a-f]{40}$", bird_dev, re.MULTILINE)


def test_the_agent_reads_the_servers_c_interact_texts():
    """The agent's tests run on copies of the ADK's texts; the copies must be the server's own."""
    from environment_servers.bird_interact import adk_prompts
    from responses_api_agents.auto_ontology_agent import c_interact
    from responses_api_agents.auto_ontology_agent.tests import adk_texts

    for name in [name for name in dir(adk_texts) if name.isupper()]:
        assert getattr(adk_texts, name) == getattr(adk_prompts, name), name
    instruction = adk_prompts.C_INTERACT_INSTRUCTION.format(
        db_name="alien", db_schema="CREATE TABLE t (a int);", external_kg="[]", max_turn=4
    )
    assert c_interact.read_task(instruction).database == "alien"
    phase_1 = adk_prompts.PHASE_1_MESSAGE.format(amb_user_query="Show the stations.", max_turn=4)
    assert c_interact.read_turn(phase_1).text == "Show the stations."
    follow_up = adk_prompts.FOLLOW_UP_MESSAGE.format(follow_up_query="Now by year.")
    assert c_interact.read_turn(follow_up).kind == "follow_up"


def test_the_catalog_has_each_dump_of_the_release_under_its_own_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("alien_template", "credit_template"):
        (tmp_path / name).mkdir()
    (tmp_path / "alien_table_orders.txt").write_text("")
    monkeypatch.setattr(build_catalog, "DUMPS_DIR", tmp_path)
    monkeypatch.setenv("POSTGRES_USER", "root")

    assert build_catalog.release_databases() == ["alien", "credit"]
    # The image's .env names the episode's port, which the episode runner fills in.
    assert build_catalog.source("alien", "${POSTGRES_PORT}") == "postgresql://root@localhost:${POSTGRES_PORT}/alien"


@pytest.mark.parametrize(
    ("config", "release"), [("auto_ontology_agent.yaml", "lite"), ("full_auto_ontology_agent.yaml", "full")]
)
def test_each_release_runs_its_prepared_rows_on_its_own_image(config: str, release: str):
    loaded = yaml.safe_load((BENCHMARK_DIR / config).read_text())
    [dataset] = loaded["auto_ontology_agent"]["responses_api_agents"]["auto_ontology_agent"]["datasets"]

    # prepare.py writes data/<release>/<mode>.jsonl.
    assert dataset["jsonl_fpath"] == f"benchmarks/bird_interact/data/{release}/c_interact.jsonl"
    assert dataset["taskset"] == "bird_interact:c_interact"
    assert loaded.get("bird_interact_release", "lite") == release
    # build_image.sh tags the image for the release, as the Lite config names it.
    lite = yaml.safe_load((BENCHMARK_DIR / "auto_ontology_agent.yaml").read_text())
    assert lite["auto_ontology_image"] == "nemo-gym/bird-interact-catalog:${bird_interact_release}"
    # A catalog with the column meanings gets its own tag, so it never stands in for the one without.
    assert (
        'TAG="${3:-nemo-gym/bird-interact-catalog:$RELEASE$([[ $COLUMN_MEANINGS == true ]] && echo -meanings)}"'
        in (BUILD_IMAGE)
    )


def test_the_pairing_runs_c_interact_on_the_auto_ontology_agent_alone():
    loaded = yaml.safe_load((BENCHMARK_DIR / "auto_ontology_agent.yaml").read_text())

    assert loaded["bird_interact_agent"] is None
    assert loaded["bird_interact_a_interact"] is None
    server = loaded["bird_interact_c_interact"]["environment_servers"]["bird_interact"]
    assert server["agent_server"] == {"name": "auto_ontology_agent"}


def test_the_interactive_pairings_run_auto_ontology_eval_s_pipeline_with_its_flags():
    fragment = yaml.safe_load((BENCHMARK_DIR / "auto_ontology" / "interactive.yaml").read_text())
    agent = fragment["auto_ontology_agent"]["responses_api_agents"]["auto_ontology_agent"]

    assert agent["pipeline"] == "interactive"
    assert agent["knowledge_dir"] == "benchmarks/bird_interact/data/${bird_interact_release}/databases"
    # auto-ontology-eval's BIRD-Interact flags, as strings for the sandbox's environment.
    assert agent["environment"]["INTERACTIVE"] == "true"
    assert all(isinstance(value, str) for value in agent["environment"].values())
    # Each release and mode chains the fragment after its own config, so the fragment's settings win.
    for config, base in (
        ("auto_ontology_interactive.yaml", "auto_ontology_agent.yaml"),
        ("full_auto_ontology_interactive.yaml", "full_auto_ontology_agent.yaml"),
        ("full_auto_ontology_a_interact.yaml", "full_a_interact.yaml"),
    ):
        paths = yaml.safe_load((BENCHMARK_DIR / config).read_text())["config_paths"]
        assert paths[0] == f"benchmarks/bird_interact/{base}"
        assert paths[-1] == "benchmarks/bird_interact/auto_ontology/interactive.yaml"


def test_the_a_interact_pairing_runs_the_full_tasks_on_the_auto_ontology_agent_alone():
    loaded = yaml.safe_load((BENCHMARK_DIR / "full_auto_ontology_a_interact.yaml").read_text())
    [dataset] = loaded["auto_ontology_agent"]["responses_api_agents"]["auto_ontology_agent"]["datasets"]

    assert (dataset["jsonl_fpath"], dataset["taskset"]) == (
        "benchmarks/bird_interact/data/full/a_interact.jsonl",
        "bird_interact:a_interact",
    )
    assert loaded["bird_interact_agent"] is None and loaded["bird_interact_c_interact"] is None
    server = loaded["bird_interact_a_interact"]["environment_servers"]["bird_interact"]
    assert server["agent_server"] == {"name": "auto_ontology_agent"}
    # The server waits past the agent's own bound on the task.
    assert server["turn_timeout_seconds"] > 1800
