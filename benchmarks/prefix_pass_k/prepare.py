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

"""Prepare prefix pass@K rows for SWE-bench Verified.

Each row pins one instance to its **decisive turn** T: the rollout replays the
captured turns 1..T-1 verbatim and the candidate model must produce turn T
onward. Scoring is the stock SWE-bench verification, so a row carries the full
SWE-bench Verified schema plus a `prefix_pass_k` block for the agent.

Three sources are joined:
  * SWE-bench Verified (HuggingFace) — repo/base_commit/FAIL_TO_PASS/...
  * a bisect `earliest_passing.json` — instance -> earliest resolving turn T
  * a captured trace directory — turn_NN.json, one per model call

The prompt is lifted from the trace's own first request rather than re-rendered
from a template: the prefix was produced against those exact system/user
messages, and a re-rendered prompt that differs by even a line puts the
candidate in a conversation the prefix never belonged to.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from datasets import load_dataset

from nemo_gym.global_config import get_hf_token


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "prefix_pass_k_verified_benchmark.jsonl"

DEFAULT_REFERENCE_ROOT = Path("/disk/tayu/swe-verified-conditional-pass-at-k")
DEFAULT_BATCH = "mini-gpt56sol-2026-08-10"

# mini-swe-agent backticks wire: the action is a bash command inside this fence
# and the agent parses the fence itself. Kept as one regex so the dataset and
# the agent cannot drift apart on what counts as an action.
ACTION_FENCE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n?```", re.DOTALL)

TURN_FILE = re.compile(r"^turn_(\d+)\.json$")


def parse_action(content: str) -> Optional[str]:
    """Return the single bash command carried by an assistant turn, if any.

    A turn with no fence is a prose/format-error turn: it carries no action, so
    it cannot be replayed as one.
    """
    matches = ACTION_FENCE.findall(content or "")
    if len(matches) != 1:
        return None
    command = matches[0].strip()
    return command or None


def load_turns(instance_dir: Path) -> list[tuple[int, dict[str, Any]]]:
    """Return (turn_index, turn_payload) sorted by the index in the filename."""
    turns = []
    for path in instance_dir.iterdir():
        match = TURN_FILE.match(path.name)
        if match:
            turns.append((int(match.group(1)), json.loads(path.read_text())))
    turns.sort(key=lambda pair: pair[0])
    return turns


def assistant_content(turn: dict[str, Any]) -> str:
    response = turn.get("response")
    if not isinstance(response, dict):
        # A failed call captures the raw error body instead of a response object
        # (e.g. a 504 HTML page). It carries no action.
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


def prompt_messages(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """The system prompt and task the agent opened the episode with.

    Taken from the first captured request, so it is by construction the prompt
    the replayed prefix was generated against. It stops at the task: a harness
    that drops an unparseable reply keeps only the format error it answered
    with, so a first request can carry that error as a second user message (20
    of the 130 Pro trajectories). No replay re-creates it -- the reply behind it
    was never captured as a turn -- and the reference replay, which re-renders
    the opening from the pinned config, does not show it to candidates either.
    """
    messages = (turn.get("request") or {}).get("messages") or []
    opening = []
    for message in messages:
        if message.get("role") not in ("system", "user"):
            break
        opening.append({"role": message["role"], "content": message.get("content") or ""})
        if message["role"] == "user":
            break
    return opening


def trailing_observation(turn: dict[str, Any]) -> Optional[str]:
    """The observation the agent sent back, read off the NEXT turn's request.

    The captured request is what the agent actually said, so it is exact for
    turns that produced no command — a format error is rendered text, not
    program output, and re-deriving it would mean reimplementing the harness's
    template.
    """
    messages = (turn.get("request") or {}).get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return None


def build_prefix(turns: list[tuple[int, dict[str, Any]]], target_turn: int) -> Optional[list[dict[str, Any]]]:
    """Replayable turns 1..target_turn-1, or None if the prefix is unusable.

    `target_turn` is INCLUSIVE — masking at T replays 1..T-1 and the candidate
    must produce T.

    A turn carrying no single parseable action is NOT an error: mini-swe-agent
    answers it with a rendered format error and lets the model retry, so such
    turns are a normal part of the conversation and are replayed as-is with the
    captured observation. Dropping them would discard a third of the pool.
    """
    by_index = dict(turns)
    prefix = []
    for index, turn in turns:
        if index >= target_turn:
            break
        if turn.get("response_status") not in (200, None):
            # A failed call (seen: 400 policy-flag, 504 gateway timeout) was
            # retried with the IDENTICAL request, so it contributed no
            # conversation turn and ran no command. Verified on
            # django__django-15916: turn_14 is a 400 and turn_15 re-sends the
            # same 28 messages. Skip it; dropping the instance would discard a
            # usable trajectory over a transport hiccup.
            continue
        content = assistant_content(turn)
        if not content:
            return None
        command = parse_action(content)
        entry: dict[str, Any] = {"turn": index, "action": command, "content": content}
        if command is None:
            # No command to run, so the observation is pure harness text: take
            # it verbatim rather than re-rendering it.
            observation = trailing_observation(by_index.get(index + 1) or {})
            if observation is None:
                return None
            entry["observation"] = observation
        prefix.append(entry)
    return prefix


class SkipInstance(Exception):
    """A bisected instance the pool cannot use; the message is the reason the build summary counts."""


def load_batch(reference_root: Path, batch: str) -> tuple[Path, dict[str, Any]]:
    """The batch's trace directory and its bisect (instance -> decisive-turn entry)."""
    trace_root = reference_root / "traces_decisive" / batch
    bisect_fpath = reference_root / "results" / f"{batch}.bisect" / "earliest_passing.json"
    if not trace_root.is_dir():
        raise SystemExit(f"trace directory not found: {trace_root}")
    if not bisect_fpath.is_file():
        raise SystemExit(f"bisect file not found: {bisect_fpath}")
    return trace_root, json.loads(bisect_fpath.read_text())


def read_pool(instances_file: Optional[Path]) -> Optional[set[str]]:
    """The instance ids listed in `instances_file`, or None to keep the whole batch."""
    if instances_file is None:
        return None
    return {
        line.strip() for line in instances_file.read_text().splitlines() if line.strip() and not line.startswith("#")
    }


def decisive_turn_row(
    instance_dir: Path, entry: dict[str, Any], batch: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The opening messages and the `prefix_pass_k` block for one bisected instance.

    Every backticks pool (Verified, Pro) builds its block here, so the block the
    agent replays cannot drift between benchmarks. Raises `SkipInstance` when the
    instance has no usable decisive turn or prefix.
    """
    if not entry.get("full_patch_resolved"):
        # The captured trajectory never resolved, so there is no decisive
        # turn to hand over at — the prefix could not pass even in principle.
        raise SkipInstance("trace_did_not_resolve")

    target_turn = entry.get("earliest_turn")
    if not isinstance(target_turn, int) or target_turn < 1:
        raise SkipInstance("no_earliest_turn")

    if not instance_dir.is_dir():
        raise SkipInstance("no_trace_dir")

    turns = load_turns(instance_dir)
    if not turns:
        raise SkipInstance("no_turn_files")

    messages = prompt_messages(turns[0][1])
    if not messages:
        raise SkipInstance("no_prompt_messages")

    prefix = build_prefix(turns, target_turn)
    if prefix is None:
        raise SkipInstance("unparseable_prefix_action")
    # The prefix may be shorter than target_turn-1 because failed calls are
    # skipped, so what matters is that no turn file is MISSING: a gap would
    # mean replaying a conversation the trajectory never had.
    if {index for index, _ in turns if index < target_turn} != set(range(1, target_turn)):
        raise SkipInstance("prefix_turns_not_contiguous")

    # The captured decisive turn itself. Not replayed during an arm — the
    # candidate must produce it — but the gold check replays 1..T and should
    # reproduce the trajectory's resolving patch, which is the only way to
    # tell a broken replay from a weak model.
    target_turn_payload = dict(turns).get(target_turn) or {}
    target_content = assistant_content(target_turn_payload)
    target_entry = {
        "turn": target_turn,
        "action": parse_action(target_content),
        "content": target_content,
    }

    meta_fpath = instance_dir / "meta.json"
    meta = json.loads(meta_fpath.read_text()) if meta_fpath.is_file() else {}

    return messages, {
        "batch": batch,
        "target_turn": target_turn,
        "n_prefix_turns": len(prefix),
        "n_prefix_actions": sum(1 for step in prefix if step["action"]),
        "n_turns": entry.get("n_turns"),
        "candidate_turns": entry.get("candidate_turns"),
        "prefix": prefix,
        "target": target_entry,
        # Byte size of the trajectory's own resolving patch: the gold check
        # replays 1..T and should reproduce it.
        "reference_patch_bytes": meta.get("patch_bytes"),
    }


def write_rows(rows: list[dict[str, Any]], skipped: Counter, output_fpath: Path) -> None:
    """Write the rows as JSONL and report how many were built and why the rest were skipped."""
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} rows -> {output_fpath}")
    if skipped:
        print("skipped:")
        for reason, count in skipped.most_common():
            print(f"  {reason}: {count}")


def build(
    reference_root: Path,
    batch: str,
    instances_file: Optional[Path],
    output_fpath: Path,
) -> None:
    trace_root, bisect = load_batch(reference_root, batch)
    pool = read_pool(instances_file)

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", token=get_hf_token())
    by_instance = {row["instance_id"]: row for row in dataset}

    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for instance_id, entry in sorted(bisect.items()):
        # Bisect keys for some batches carry an `instance_` prefix the benchmark
        # ids do not use.
        bare_id = instance_id[len("instance_") :] if instance_id.startswith("instance_") else instance_id
        if pool is not None and bare_id not in pool and instance_id not in pool:
            continue
        swebench_row = by_instance.get(bare_id)
        if swebench_row is None:
            skipped["not_in_swebench_verified"] += 1
            continue
        try:
            messages, block = decisive_turn_row(trace_root / bare_id, entry, batch)
        except SkipInstance as reason:
            skipped[str(reason)] += 1
            continue

        row = dict(swebench_row)
        row["subset"] = row.get("subset") or "verified"
        row["split"] = row.get("split") or "test"
        row["responses_create_params"] = {"input": messages}
        row["prefix_pass_k"] = block
        rows.append(row)

    write_rows(rows, skipped, output_fpath)


def prepare() -> Path:
    """Gym entry point (`gym eval prepare`): build the rows, return their path.

    Gym calls this with no arguments, and importing nemo_gym installs Hydra's
    own CLI parser, so argparse is not available here. The trace corpus lives
    outside the repo and its location is machine-specific, hence the
    environment overrides.
    """
    reference_root = Path(os.environ.get("PREFIX_PASS_K_REFERENCE_ROOT", DEFAULT_REFERENCE_ROOT))
    batch = os.environ.get("PREFIX_PASS_K_BATCH", DEFAULT_BATCH)
    instances = os.environ.get("PREFIX_PASS_K_INSTANCES_FILE")
    output = Path(os.environ.get("PREFIX_PASS_K_OUTPUT", OUTPUT_FPATH))
    build(reference_root, batch, Path(instances) if instances else None, output)
    return output


if __name__ == "__main__":
    print(prepare())
