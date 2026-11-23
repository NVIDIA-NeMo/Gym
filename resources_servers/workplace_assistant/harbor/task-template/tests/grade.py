# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grade one episode with the Resources Server's own ``is_correct``.

``is_correct(predicted, ground_truth, None)`` replays both call lists on fresh tables and compares the final state, as
``WorkbenchResourcesServer.verify`` does. The predicted calls are the tool server's journal, in the shape verify builds
from the response's ``function_call`` items (``name`` and ``arguments`` as JSON text).

Usage: grade.py <journal.jsonl> <ground_truth.json> <reward.json>
"""

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "wa"))

from resources_servers.workplace_assistant.utils import is_correct  # noqa: E402


def main(journal_path: str, ground_truth_path: str, reward_path: str) -> None:
    lines = Path(journal_path).read_text(encoding="utf-8").splitlines()
    predicted = [json.loads(line) for line in lines if line.strip()]
    ground_truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    reward = is_correct(predicted, ground_truth, None) * 1.0
    print(f"journal calls: {len(predicted)}; ground-truth calls: {len(ground_truth)}; reward: {reward}")
    Path(reward_path).write_text(json.dumps({"reward": reward, "journal_calls": len(predicted)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
