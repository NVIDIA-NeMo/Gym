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
    assert 'TAG="${3:-nemo-gym/bird-interact-catalog:$RELEASE}"' in BUILD_IMAGE


def test_the_pairing_runs_c_interact_on_the_auto_ontology_agent_alone():
    loaded = yaml.safe_load((BENCHMARK_DIR / "auto_ontology_agent.yaml").read_text())

    assert loaded["bird_interact_agent"] is None
    assert loaded["bird_interact_a_interact"] is None
    server = loaded["bird_interact_c_interact"]["environment_servers"]["bird_interact"]
    assert server["agent_server"] == {"name": "auto_ontology_agent"}
