#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Assemble trajectories offline from a capture directory — the inspection
tool for validating the token path without a trainer in the loop.

  python scripts/build_trajectories.py --capture-dir RUN/ng_captures \
      --rollout-json RUN/rollout.json --out RUN/trajectories.jsonl \
      [--builder per_request|prefix_merging]

Prints, per rollout: chains, quarantined branches, mask coverage, and runs
the copy of NeMo-RL's contiguity check on the main-chain projection.
"""

from __future__ import annotations

import argparse
import json

from nemo_gym.observability.capture_reader import LocalCaptureReader
from nemo_gym.observability.capture_store import LocalJsonlCaptureStore
from nemo_gym.observability.records import ModelCallRecord
from nemo_gym.trajectory.builder import (
    assert_nemo_rl_contiguity,
    build_trajectories,
    project_main_chain_response,
)
from nemo_gym.trajectory.registry import get_builder
from nemo_gym.trajectory.sources import CaptureTokenSource


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--capture-dir", required=True)
    p.add_argument(
        "--rollout-json", default=None, help="rollout record (for reward + rollout_id); else use --rollout-id"
    )
    p.add_argument("--rollout-id", default=None)
    p.add_argument("--builder", default="prefix_merging")
    p.add_argument("--out", default="trajectories.jsonl")
    args = p.parse_args()

    reward, rid = 0.0, args.rollout_id
    if args.rollout_json:
        rec = json.load(open(args.rollout_json))
        rid = rec["rollout_id"]
        reward = float(rec.get("reward", 0.0))
    assert rid, "need --rollout-id or --rollout-json"

    store = LocalJsonlCaptureStore(args.capture_dir)
    reader = LocalCaptureReader(store)
    steps = [r for r in reader.records(rid, kinds={"model_call"}) if isinstance(r, ModelCallRecord)]
    source = CaptureTokenSource(reader)
    entries = source.entries(rid)
    print(f"rollout {rid}: {len(steps)} steps, {len(entries)} token entries")

    trajectories = build_trajectories(rid, source, model_call_records=steps, builder=args.builder, reward=reward)
    with open(args.out, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t.model_dump()) + "\n")

    for t in trajectories:
        n1 = sum(t.loss_mask)
        print(
            f"  chain={t.chain_id:10s} tokens={len(t.token_ids):5d} "
            f"trainable={n1:5d} ({100.0 * n1 / max(len(t.token_ids), 1):.1f}%) "
            f"quarantined={t.provenance.get('quarantined_branches', 0)}"
        )

    out = get_builder(args.builder)(entries)
    proj = project_main_chain_response(rid, out.chains, {s.request_id: s.response for s in steps})
    assert_nemo_rl_contiguity(proj)
    print(
        f"NeMo-RL contiguity check: PASS "
        f"({len(proj['output'])} projected items, "
        f"{proj['ng_projection']['n_branches']} branches held back)"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
