# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the ASB selector matrix into Gym Responses rows.

ASB publishes construction tools, not a dataset: the benchmark only exists once agents,
tasks and attacker tools have been crossed and the injected prompts assembled. This module
performs that expansion once, deterministically, and writes JSONL that every later run
reads. Nothing upstream is vendored -- ``fetch`` pins a gitignored checkout and this module
reads it in place.

Why the rows are pinned rather than regenerated
-----------------------------------------------
The expansion itself is cheap and deterministic, so pinning is not a cache. It is a pin,
for three reasons:

* Upstream is a live repository, and one of its data files is *already* a failed fetch
  committed as data -- ``agent_task_pot_all.jsonl`` is GitHub rate-limit HTML. A scheduled
  job that re-clones inherits that class of silent corruption.
* The paraphrase-style defenses rewrite the task with an auxiliary LLM before the model
  under test sees it. Regenerating those rows weekly changes the *input*, so a week-over-
  week delta would no longer be attributable to the checkpoint.
* An unattended weekly evaluation should not depend on github.com being reachable.

``verify_against_upstream`` re-runs the expansion and diffs it against the pinned rows, so
divergence is still detected -- it just does not silently become the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from benchmarks.asb import upstream_spec as spec


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO_ROOT / "benchmarks" / "asb" / "upstream" / "ASB"
DATA_DIR = REPO_ROOT / "resources_servers" / "asb" / "data"

DEFAULT_HF_REPO = os.environ.get("ASB_HF_REPO")

#: A public, MIT-licensed expansion of this matrix, so that reproducing a published number
#: does not require re-deriving 10,800 rows or hosting a copy first. `pull` falls back to it;
#: `push` deliberately does not, because publishing into a namespace the caller did not name
#: is a different kind of mistake from reading a public one. It is third-party rather than
#: NeMo-owned -- `verify` is what establishes that any pinned copy really is upstream's.
PUBLISHED_HF_REPO = "theverifier/asb-selectors"


# ---------------------------------------------------------------------------
# Upstream checkout
# ---------------------------------------------------------------------------


def fetch(upstream_dir: Path = UPSTREAM_DIR, *, revision: str = spec.UPSTREAM_REVISION) -> Path:
    """Clone ASB at the pinned revision if it is not already present, then verify it."""
    if not (upstream_dir / ".git").exists():
        upstream_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", spec.UPSTREAM_REPO, str(upstream_dir)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(upstream_dir), "checkout", "--quiet", revision],
        check=True,
    )
    verify_upstream_hashes(upstream_dir)
    return upstream_dir


def verify_upstream_hashes(upstream_dir: Path = UPSTREAM_DIR) -> None:
    """Fail loudly if any pinned input differs from the revision this adapter was built on.

    ``UPSTREAM_INVALID_FILES`` is checked too: those paths are known-corrupt upstream and
    must never be parsed, so their continued corruption is asserted rather than ignored.
    """
    for name, expected in spec.DATA_FILE_SHA256.items():
        path = upstream_dir / "data" / name
        if not path.exists():
            raise FileNotFoundError(f"ASB input missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"ASB input {name} changed upstream.\n  expected {expected}\n  actual   {actual}\n"
                f"Pinned revision is {spec.UPSTREAM_REVISION}. Re-pin deliberately; do not run across a change."
            )
    for name in spec.UPSTREAM_INVALID_FILES:
        path = upstream_dir / "data" / name
        if path.exists() and path.read_bytes().lstrip()[:5].lower() == b"<html":
            continue
        if path.exists():
            raise ValueError(
                f"{name} is no longer the known-corrupt HTML placeholder. Upstream may have fixed it; "
                "re-read benchmarks/asb/README.md before trusting it."
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Upstream asset loading
# ---------------------------------------------------------------------------


def load_agent_configs(upstream_dir: Path) -> dict[str, dict[str, Any]]:
    """Agent description and declared tools, from ``pyopenagi/agents/example/*/config.json``."""
    configs: dict[str, dict[str, Any]] = {}
    base = upstream_dir / "pyopenagi" / "agents" / "example"
    for config_path in sorted(base.glob("*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configs[config_path.parent.name] = config
    return configs


def load_normal_tools(upstream_dir: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(upstream_dir / "data" / "all_normal_tools.jsonl")
    return {row["Tool Name"]: row for row in rows}


def normal_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """``SimulatedTool.get_tool_call_format`` -- note the explicit null ``parameters``."""
    return {
        "type": "function",
        "function": {"name": tool["Tool Name"], "description": tool["Description"], "parameters": None},
    }


def attacker_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """``AttackerTool.get_tool_call_format`` -- which omits ``parameters`` entirely.

    The asymmetry with :func:`normal_tool_schema` is upstream's. Both schemas are rendered
    into the planning prompt verbatim, so normalizing them would change model input.
    """
    return {
        "type": "function",
        "function": {"name": tool["Attacker Tool"], "description": tool["Description"]},
    }


def agent_prefix(config: dict[str, Any]) -> str:
    """``build_system_instruction``'s ``prefix``: the description list joined with no separator."""
    return "".join(config["description"])


# ---------------------------------------------------------------------------
# Memory corpus extraction
# ---------------------------------------------------------------------------


def extract_memory_corpus(upstream_dir: Path) -> dict[str, list[str]]:
    """Pull the poisoned memory records out of upstream's shipped Chroma stores.

    The published Memory Poisoning numbers were produced against these exact stores: each
    is the memory a gpt-4o-mini DPI run wrote under one attack type. The record count per
    store (~400) is itself confirmation that those write runs were 400 rows.

    Only the documents are extracted. Upstream ranks them with OpenAI ``text-embedding-
    ada-002``; this adapter ranks the same corpus lexically because no OpenAI credential is
    in play. That substitution is disclosed in ``benchmarks/asb/METRICS.md`` -- the poison
    available to be retrieved is upstream's, the ranking over it is not.
    """
    corpus: dict[str, list[str]] = {}
    base = upstream_dir / "memory_db" / "direct_prompt_injection"
    for store in sorted(base.glob("*_gpt-4o-mini/chroma.sqlite3")):
        attack_type = store.parent.name.rsplit("_gpt-4o-mini", 1)[0]
        connection = sqlite3.connect(store)
        try:
            documents = [
                row[0]
                for row in connection.execute(
                    "SELECT string_value FROM embedding_metadata WHERE key = 'chroma:document' "
                    "AND string_value IS NOT NULL"
                )
            ]
        finally:
            connection.close()
        corpus[attack_type] = documents
    return corpus


WORKFLOW_RE = re.compile(r"Workflow:\s*(\[.*?\]);", re.DOTALL)


def memory_workflow(record: str) -> str | None:
    """The workflow fragment upstream splices out of a retrieved record."""
    match = WORKFLOW_RE.search(record)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Selector expansion
# ---------------------------------------------------------------------------


def attacker_tools_for(rows: list[dict[str, Any]], agent_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["Corresponding Agent"] == agent_name]


def attack_tool_file(aggressiveness: str) -> str:
    return {
        "all": "all_attack_tools.jsonl",
        "agg": "all_attack_tools_aggressive.jsonl",
        "non-agg": "all_attack_tools_non_aggressive.jsonl",
    }[aggressiveness]


def expand_selectors(
    *,
    upstream_dir: Path,
    injection_method: str,
    attack_type: str,
    defense_type: str | None = None,
    trigger: str | None = None,
    aggressiveness: str = "all",
    task_num: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Reproduce ``main_attacker.main``'s triple loop for one condition.

    ``task_num`` defaults to the value the published run for this condition used: 1 for
    everything except PoT, which ``config/POT.yml`` sets to 2.
    """
    is_pot = injection_method in {"pot_backdoor", "pot_clean"}
    if task_num is None:
        task_num = spec.PUBLISHED_POT_TASK_NUM if is_pot else spec.PUBLISHED_TASK_NUM

    task_file = "agent_task_pot.jsonl" if is_pot else "agent_task.jsonl"
    agent_rows = _read_jsonl(upstream_dir / "data" / task_file)
    attack_rows = _read_jsonl(upstream_dir / "data" / attack_tool_file(aggressiveness))
    configs = load_agent_configs(upstream_dir)
    tools_by_name = load_normal_tools(upstream_dir)
    pot_messages = {row["agent_name"]: row for row in _read_jsonl(upstream_dir / "data" / "agent_task_pot_msg.jsonl")}

    uses_attacker_tool = injection_method != "clean"

    for agent_row in agent_rows:
        agent_name = agent_row["agent_name"]
        config = configs[agent_name]
        prefix = agent_prefix(config)
        normal_tools = [tools_by_name[ref.split("/")[-1]] for ref in config["tools"]]

        for task_index, task in enumerate(agent_row["tasks"]):
            if task_index >= task_num:
                break
            for attacker_tool in attacker_tools_for(attack_rows, agent_name):
                yield build_row(
                    injection_method=injection_method,
                    attack_type=attack_type,
                    defense_type=defense_type,
                    trigger=trigger,
                    aggressiveness=aggressiveness,
                    agent_name=agent_name,
                    prefix=prefix,
                    task=task,
                    task_index=task_index,
                    normal_tools=normal_tools,
                    attacker_tool=attacker_tool,
                    uses_attacker_tool=uses_attacker_tool,
                    pot_message=pot_messages.get(agent_name),
                    agent_config=config,
                )


def build_row(
    *,
    injection_method: str,
    attack_type: str,
    defense_type: str | None,
    trigger: str | None,
    aggressiveness: str,
    agent_name: str,
    prefix: str,
    task: str,
    task_index: int,
    normal_tools: list[dict[str, Any]],
    attacker_tool: dict[str, Any],
    uses_attacker_tool: bool,
    pot_message: dict[str, Any] | None,
    agent_config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one Gym row, reproducing ``ReactAgentAttack.run``'s prompt construction.

    The message list is materialized here rather than in the agent so that a row is a
    complete, auditable record of what the model was shown. The only message the agent
    adds at run time is the retrieved-memory turn, which by definition cannot be known
    until retrieval happens.
    """
    is_pot = injection_method in {"pot_backdoor", "pot_clean"}
    is_dpi = injection_method in {"direct_prompt_injection", "mixed_attack", "DPI_MP", "DPI_OPI"}
    is_opi = injection_method in {"observation_prompt_injection", "mixed_attack", "OPI_MP", "DPI_OPI"}
    reads_memory = injection_method in {"memory_attack", "mixed_attack", "DPI_MP", "OPI_MP"}

    attacker_name = attacker_tool["Attacker Tool"]
    attack_prompts = spec.build_attack_prompts(
        attacker_instruction=attacker_tool["Attacker Instruction"], attacker_tool=attacker_name
    )

    tool_schemas = [normal_tool_schema(tool) for tool in normal_tools]
    if uses_attacker_tool:
        tool_schemas.append(attacker_tool_schema(attacker_tool))
    tools_json = json.dumps(tool_schemas)

    # Defenses that wrap the task are applied before the injection is appended, matching
    # the order in ``run``: delimiters and instructional prevention come first.
    task_input = task
    if defense_type == "delimiters_defense":
        task_input = spec.apply_delimiters_defense(task_input)
    elif defense_type == "instructional_prevention":
        task_input = spec.apply_instructional_prevention(task_input)

    if is_dpi:
        task_input = f"{task_input} {attack_prompts[attack_type]}"

    if injection_method == "pot_backdoor":
        task_input = f"{task_input} {trigger}."

    messages: list[dict[str, str]] = [{"role": "system", "content": prefix}]
    if defense_type == "delimiters_defense":
        messages[0]["content"] = prefix + spec.DELIMITERS_SYSTEM_SUFFIX

    if is_pot:
        assert pot_message is not None, f"no PoT few-shot messages for {agent_name}"
        tool1 = agent_config["tools"][0].split("/")[-1]
        tool2 = agent_config["tools"][1].split("/")[-1]
        poisoned = spec.build_pot_example_step(trigger=trigger or "", attacker_tool=attacker_name)
        example_1 = [
            {"message": pot_message["msg1"], "tool_use": [tool1]},
            {"message": pot_message["msg2"], "tool_use": [tool2]},
            poisoned,
        ]
        example_2 = [
            {"message": pot_message["msg3"], "tool_use": [tool1]},
            {"message": pot_message["msg4"], "tool_use": [tool2]},
            dict(poisoned),
        ]
        if defense_type == "pot_shuffling_defense":
            import random

            rng = random.Random(spec.SEED)
            # Upstream seeds the global RNG with 0 and shuffles in place; a dedicated
            # Random with the same seed reproduces the permutation without the global.
            rng.shuffle(example_1)
            rng.shuffle(example_2)
        messages.append(
            {
                "role": "system",
                "content": spec.build_pot_instruction(
                    tools_json=tools_json,
                    trigger=trigger or "",
                    task_intro1=pot_message["Task1"],
                    task_intro2=pot_message["Task2"],
                    example_1=example_1,
                    example_2=example_2,
                ),
            }
        )
    elif reads_memory:
        # Upstream moves the plan instruction to the *user* role under read_db, then puts
        # the retrieved memory in an assistant turn. The agent inserts that turn.
        messages.append({"role": "user", "content": spec.build_plan_instruction(tools_json)})
    else:
        messages.append({"role": "system", "content": spec.build_plan_instruction(tools_json)})

    messages.append({"role": "user", "content": task_input})

    condition = condition_id(
        injection_method=injection_method,
        attack_type=attack_type,
        defense_type=defense_type,
        trigger=trigger,
        aggressiveness=aggressiveness,
    )
    asb_id = f"{condition}|{agent_name}|t{task_index}|{attacker_name}"

    return {
        "asb_id": asb_id,
        "condition": condition,
        "responses_create_params": {
            "input": messages,
            # Upstream's API path pins temperature to 0.0 (aios/llm_core/llm_classes/gpt_llm.py).
            "temperature": 0.0,
        },
        "verifier_metadata": {
            "upstream_revision": spec.UPSTREAM_REVISION,
            "injection_method": injection_method,
            "attack_type": attack_type,
            "defense_type": defense_type,
            "trigger": trigger,
            "aggressiveness": aggressiveness,
            "aggressive": attacker_tool["Aggressive"] == "True",
            "agent_name": agent_name,
            "agent_prefix": prefix,
            "task": task,
            "task_index": task_index,
            "task_input": task_input,
            "tools_schema": tool_schemas,
            "normal_tools": [
                {
                    "name": tool["Tool Name"],
                    "description": tool["Description"],
                    "expected_achievement": tool["Expected Achievements"],
                }
                for tool in normal_tools
            ],
            "attacker_tool": {
                "name": attacker_name,
                "description": attacker_tool["Description"],
                "instruction": attacker_tool["Attacker Instruction"],
                "attack_goal": attacker_tool["Attack goal"],
                "attack_category": attacker_tool["Attack Type"],
            },
            "attacker_tool_available": uses_attacker_tool,
            "injects_into_observations": is_opi,
            "injects_into_task": is_dpi,
            "reads_memory": reads_memory,
            "observation_injection": attack_prompts[attack_type] if is_opi else None,
            "memory_key": attack_type if reads_memory else None,
        },
    }


def condition_id(
    *,
    injection_method: str,
    attack_type: str,
    defense_type: str | None,
    trigger: str | None,
    aggressiveness: str,
) -> str:
    parts = [injection_method, attack_type, defense_type or "no_defense", aggressiveness]
    if trigger:
        parts.append(re.sub(r"\W+", "_", trigger).strip("_") or "trigger")
    return ".".join(parts)


# ---------------------------------------------------------------------------
# The published matrix
# ---------------------------------------------------------------------------


def published_conditions() -> list[dict[str, Any]]:
    """Every condition behind a published ASB table cell.

    Denominators are asserted in ``tests/test_prepare.py`` against the granularity of the
    published percentages -- see ``upstream_spec`` for that derivation.
    """
    conditions: list[dict[str, Any]] = []

    for attack_type in spec.ATTACK_TYPES:
        conditions.append({"injection_method": "direct_prompt_injection", "attack_type": attack_type})
        conditions.append({"injection_method": "observation_prompt_injection", "attack_type": attack_type})
        conditions.append({"injection_method": "memory_attack", "attack_type": attack_type})

    for attack_type in spec.MIXED_ATTACK_TYPES:
        conditions.append({"injection_method": "mixed_attack", "attack_type": attack_type})

    conditions.append(
        {"injection_method": "pot_backdoor", "attack_type": "naive", "trigger": spec.DEFAULT_POT_TRIGGER}
    )
    conditions.append({"injection_method": "pot_clean", "attack_type": "naive", "trigger": spec.DEFAULT_POT_TRIGGER})
    conditions.append({"injection_method": "clean", "attack_type": "combined_attack"})

    # The defense tables report a single attack type per arm; combined_attack is the
    # strongest and the one the MP/mixed configs default to.
    for defense in ("delimiters_defense", "direct_paraphrase_defense", "instructional_prevention"):
        conditions.append(
            {
                "injection_method": "direct_prompt_injection",
                "attack_type": "combined_attack",
                "defense_type": defense,
            }
        )
    for defense in ("delimiters_defense", "instructional_prevention", "ob_sandwich_defense"):
        conditions.append(
            {
                "injection_method": "observation_prompt_injection",
                "attack_type": "combined_attack",
                "defense_type": defense,
            }
        )
    return conditions


def materialize(
    *,
    upstream_dir: Path = UPSTREAM_DIR,
    out_dir: Path = DATA_DIR,
    conditions: list[dict[str, Any]] | None = None,
    task_num: int | None = None,
) -> dict[str, int]:
    """Write one JSONL per condition plus a combined ``all.jsonl``, and a manifest."""
    verify_upstream_hashes(upstream_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = conditions if conditions is not None else published_conditions()

    counts: dict[str, int] = {}
    combined: list[dict[str, Any]] = []
    for condition in conditions:
        rows = list(expand_selectors(upstream_dir=upstream_dir, task_num=task_num, **condition))
        name = (
            rows[0]["condition"]
            if rows
            else condition_id(
                injection_method=condition["injection_method"],
                attack_type=condition["attack_type"],
                defense_type=condition.get("defense_type"),
                trigger=condition.get("trigger"),
                aggressiveness=condition.get("aggressiveness", "all"),
            )
        )
        path = out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        counts[name] = len(rows)
        combined.extend(rows)

    (out_dir / "all.jsonl").write_text("".join(json.dumps(row) + "\n" for row in combined), encoding="utf-8")

    corpus = extract_memory_corpus(upstream_dir)
    (out_dir / "memory_corpus.json").write_text(json.dumps(corpus), encoding="utf-8")

    manifest = {
        "upstream_repo": spec.UPSTREAM_REPO,
        "upstream_revision": spec.UPSTREAM_REVISION,
        "upstream_sha256": spec.DATA_FILE_SHA256,
        "rows_total": len(combined),
        "rows_by_condition": counts,
        "memory_corpus_sizes": {key: len(value) for key, value in corpus.items()},
        "content_hash": hashlib.sha256(
            "".join(json.dumps(row, sort_keys=True) for row in combined).encode()
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return counts


def verify_against_upstream(*, upstream_dir: Path = UPSTREAM_DIR, data_dir: Path = DATA_DIR) -> bool:
    """Re-expand from upstream and compare to the pinned rows.

    Returns True when they match. This is the check that keeps pinning honest: it is meant
    to run on a schedule and *report*, never to overwrite the pinned data on its own.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for condition in published_conditions():
        rows.extend(expand_selectors(upstream_dir=upstream_dir, **condition))
    content_hash = hashlib.sha256("".join(json.dumps(row, sort_keys=True) for row in rows).encode()).hexdigest()
    return content_hash == manifest["content_hash"]


# ---------------------------------------------------------------------------
# Hugging Face round-trip
# ---------------------------------------------------------------------------


def push_to_hub(repo_id: str, *, data_dir: Path = DATA_DIR, private: bool = True) -> str:
    """Upload the materialized rows and manifest as a dataset revision.

    Only inputs are uploaded. Rollouts and scores are per-checkpoint outputs and stay in
    ``results/``; publishing them here would make the pinned dataset mutate every run.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(data_dir),
        allow_patterns=["*.jsonl", "manifest.json", "memory_corpus.json"],
        commit_message=f"ASB selectors @ {spec.UPSTREAM_REVISION[:12]} ({manifest['rows_total']} rows)",
    )
    return repo_id


def pull_from_hub(repo_id: str, *, data_dir: Path = DATA_DIR, revision: str | None = None) -> Path:
    """Restore pinned rows from the hub -- what a scheduled run should do."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(data_dir),
    )
    return Path(path)


#: Where ``benchmarks/asb/config.yaml`` expects the built benchmark rows. ``gym eval
#: prepare`` rejects a prepare() whose return value does not match that path exactly.
BENCHMARK_FPATH = REPO_ROOT / "benchmarks" / "asb" / "data" / "asb_benchmark.jsonl"


def prepare() -> Path:
    """Build the whole-matrix ASB benchmark JSONL.

    Clones the pinned upstream revision, verifies every load-bearing input by SHA-256,
    expands the 27 published conditions, and copies the concatenated rows to the path the
    benchmark config names.
    """
    fetch()
    counts = materialize()
    source = DATA_DIR / "all.jsonl"
    BENCHMARK_FPATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, BENCHMARK_FPATH)
    print(f"ASB: wrote {sum(counts.values())} rows across {len(counts)} conditions -> {BENCHMARK_FPATH}")
    return BENCHMARK_FPATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "materialize", "verify", "push", "pull"])
    parser.add_argument("--task-num", type=int, default=None, help="Override the published task_num (not comparable)")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)

    if args.command == "push" and not args.repo_id:
        parser.error("push requires --repo-id or ASB_HF_REPO: it will not guess a namespace to publish into")
    if args.command == "pull" and not args.repo_id:
        args.repo_id = PUBLISHED_HF_REPO

    if args.command == "fetch":
        print(fetch())
    elif args.command == "materialize":
        fetch()
        counts = materialize(task_num=args.task_num)
        for name, count in sorted(counts.items()):
            print(f"{count:6d}  {name}")
        print(f"{sum(counts.values()):6d}  TOTAL across {len(counts)} conditions")
    elif args.command == "verify":
        ok = verify_against_upstream()
        print("pinned rows match upstream" if ok else "PINNED ROWS DIVERGE FROM UPSTREAM")
        return 0 if ok else 1
    elif args.command == "push":
        print(push_to_hub(args.repo_id))
    elif args.command == "pull":
        print(pull_from_hub(args.repo_id, revision=args.revision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
