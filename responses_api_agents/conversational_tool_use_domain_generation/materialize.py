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

"""Convert domain-generation Gym rollouts into policy/tool-generation Gym inputs."""

from __future__ import annotations

import argparse
import copy
import json
import random
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


POLICY_TOOL_AGENT_REF = {
    "type": "responses_api_agents",
    "name": "conversational_tool_use_policy_tool_generation",
}
GenerationProfile = Literal["general", "proactive"]
GYM_IDENTITY_KEYS = (
    "id",
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
)


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: rollout row must be a JSON object")
            yield line_number, value


def _rollout_candidates(row: dict[str, Any], *, source: Path, line_number: int) -> list[Any]:
    result = row.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("candidates"), list):
        raise ValueError(f"{source}:{line_number}: rollout row is missing result.candidates")
    return result["candidates"]


def _source_artifacts(
    rollout: dict[str, Any],
    *,
    candidate_index: int,
) -> dict[str, Any]:
    existing = rollout.get("source_artifacts", {})
    if not isinstance(existing, dict):
        raise ValueError("source_artifacts must be a JSON object when present")
    identity = {key: rollout[key] for key in GYM_IDENTITY_KEYS if rollout.get(key) is not None}
    identity["candidate_index"] = candidate_index
    artifacts = copy.deepcopy(existing)
    artifacts["domain_generation"] = identity
    return artifacts


def _derived_rollout_id(
    rollout: dict[str, Any],
    *,
    fallback: str,
) -> str:
    base = str(rollout.get("id") or fallback)
    identity_parts = []
    if rollout.get(TASK_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"t{rollout[TASK_INDEX_KEY_NAME]}")
    if rollout.get(ROLLOUT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"r{rollout[ROLLOUT_INDEX_KEY_NAME]}")
    if rollout.get(ATTEMPT_INDEX_KEY_NAME) is not None:
        identity_parts.append(f"a{rollout[ATTEMPT_INDEX_KEY_NAME]}")
    if identity_parts:
        return f"{base}_ng_{'_'.join(identity_parts)}"
    return base


def materialize_policy_tool_rows(
    rollouts: Iterable[tuple[int, dict[str, Any]]],
    *,
    source: Path,
    profile: GenerationProfile,
    shuffle_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Preserve candidate objects while applying casefold-only first-wins deduplication."""
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    for line_number, rollout in rollouts:
        for candidate_index, candidate in enumerate(
            _rollout_candidates(rollout, source=source, line_number=line_number)
        ):
            if not isinstance(candidate, dict):
                raise ValueError(f"{source}:{line_number}: result.candidates[{candidate_index}] must be a JSON object")
            name = candidate.get("name")
            if not isinstance(name, str):
                raise ValueError(f"{source}:{line_number}: result.candidates[{candidate_index}].name must be a string")

            dedup_key = name.casefold()
            if dedup_key in seen_names:
                continue
            seen_names.add(dedup_key)
            row_id = (
                f"{_derived_rollout_id(rollout, fallback=f'domain_rollout_{line_number:06d}')}"
                f"_candidate_{candidate_index:06d}"
            )
            if row_id in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate materialized row id: {row_id}")
            seen_ids.add(row_id)
            rows.append(
                {
                    "id": row_id,
                    "responses_create_params": {"input": []},
                    "domain": candidate,
                    "profile": profile,
                    "source_artifacts": _source_artifacts(
                        rollout,
                        candidate_index=candidate_index,
                    ),
                    "agent_ref": POLICY_TOOL_AGENT_REF.copy(),
                }
            )

    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize policy/tool Gym inputs from domain-generation rollout JSONL."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Domain Gym rollout JSONL")
    parser.add_argument("--output-file", type=Path, required=True, help="Policy/tool Gym input JSONL")
    parser.add_argument(
        "--profile",
        choices=("general", "proactive"),
        required=True,
        help="Policy/tool generation profile stamped on every output row",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle deduplicated rows with this explicit integer seed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input_file.resolve() == args.output_file.resolve():
        raise ValueError("input and output paths must differ")
    rows = materialize_policy_tool_rows(
        read_jsonl(args.input_file),
        source=args.input_file,
        profile=args.profile,
        shuffle_seed=args.shuffle_seed,
    )
    write_jsonl(args.output_file, rows)
    print(f"Wrote {len(rows)} policy/tool input rows to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
