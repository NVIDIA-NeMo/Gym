# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the committed example rows from freshly prepared validation rows.

The example rows are prepared UserSim rows with a synthetic persona, so they carry
no Nemotron-Personas data. After a UserSim bump, run
``gym eval prepare --config environments/nemo_user_sim/config.yaml`` and then this
script. For each example it takes the prepared row with the same probe and locale
and keeps the example's synthetic persona fields and task id, so the preparation
must include every example locale, as the default preparation does. Collect ``example_rollouts.jsonl``
again afterwards: each rollout records a digest of its example row.
"""

import argparse
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_FPATH = REPOSITORY_ROOT / "resources_servers/nemo_user_sim/data/example.jsonl"
PREPARED_FPATH = REPOSITORY_ROOT / "environments/nemo_user_sim/data/nemo_user_sim.jsonl"
# Row fields that describe the simulated user rather than the probe.
SYNTHETIC_FIELDS = ("behavioral_profile", "disclosure_style", "user_interaction_style")


def _is_synthetic_field(name: str) -> bool:
    return name.startswith("persona") or name in SYNTHETIC_FIELDS


def regenerate(examples: list[dict], prepared: list[dict]) -> list[dict]:
    """Return ``examples`` rebuilt on the prepared row of each example's probe and locale."""
    prepared_by_key = {
        (row["resolved_row"]["probe_type"], row["resolved_row"]["locale"]): row["resolved_row"] for row in prepared
    }
    rebuilt = []
    for example in examples:
        synthetic = example["resolved_row"]
        key = (synthetic["probe_type"], synthetic["locale"])
        if key not in prepared_by_key:
            raise ValueError(f"No prepared row for probe {key[0]!r} in locale {key[1]!r}")
        row = {name: value for name, value in prepared_by_key[key].items() if not _is_synthetic_field(name)}
        row.update({name: value for name, value in synthetic.items() if _is_synthetic_field(name)})
        row["trajectory_id"] = example["task_id"]
        provenance = json.loads(row["usersim_provenance"])
        provenance["nemotron_personas_version"] = "synthetic"
        row["usersim_provenance"] = json.dumps(provenance)
        rebuilt.append({"task_id": example["task_id"], "resolved_row": row})
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, default=PREPARED_FPATH)
    parser.add_argument("--examples", type=Path, default=EXAMPLES_FPATH)
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.examples.read_text().splitlines() if line.strip()]
    prepared = [json.loads(line) for line in args.prepared.read_text().splitlines() if line.strip()]
    rebuilt = regenerate(examples, prepared)
    args.examples.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rebuilt)
    )
    print(f"Wrote {len(rebuilt)} example rows to {args.examples}")


if __name__ == "__main__":
    main()
