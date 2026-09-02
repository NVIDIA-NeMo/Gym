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
"""Turn a validated sweep manifest into the artifacts `gym eval run` needs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from nemo_gym.sweep.manifest import AGENT_REF_KEY, SweepManifest, SweepValidationError


INPUT_JSONL_NAME = "input.jsonl"
CONFIG_NAME = "sweep_config.yaml"
REPORT_NAME = "build_report.json"


@dataclass
class BuildReport:
    input_jsonl: Path
    config_yaml: Path
    report_json: Path
    rows_per_entry: Dict[str, int] = field(default_factory=dict)
    overrides_applied: Dict[str, str] = field(default_factory=dict)
    num_shards: int = 1
    num_repeats: Dict[str, int] = field(default_factory=dict)
    config_paths: List[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_entry.values())

    def to_dict(self) -> Dict:
        return {
            "input_jsonl": str(self.input_jsonl),
            "config_yaml": str(self.config_yaml),
            "total_rows": self.total_rows,
            "rows_per_entry": self.rows_per_entry,
            "num_shards": self.num_shards,
            "overrides_applied": self.overrides_applied,
            "num_repeats": self.num_repeats,
            "config_paths": self.config_paths,
        }


def build_sweep(
    manifest: SweepManifest,
    out_dir: str | Path,
    *,
    limit_per_entry: Optional[int] = None,
    overwrite: bool = False,
) -> BuildReport:
    """Concatenate every entry's rows into one input JSONL and emit its companion config.

    Rows are streamed, so the training files are never held in memory. Each row keeps its own
    ``agent_ref``, which is what lets one input file fan out to many environments: rollout
    collection dispatches per row by that field.
    """
    out_dir = Path(out_dir) / manifest.nickname
    out_dir.mkdir(parents=True, exist_ok=True)

    input_jsonl = out_dir / INPUT_JSONL_NAME
    config_yaml = out_dir / CONFIG_NAME
    report_json = out_dir / REPORT_NAME
    if input_jsonl.exists() and not overwrite:
        raise SweepValidationError(f"{input_jsonl} already exists. Pass overwrite=True to replace it.")

    report = BuildReport(input_jsonl=input_jsonl, config_yaml=config_yaml, report_json=report_json)

    shards = manifest.num_shards or 1
    if shards > 1:
        sinks = [open(out_dir / f"input_{i:03d}.jsonl", "w") for i in range(shards)]
        input_jsonl = out_dir / "input_000.jsonl"
        report.input_jsonl = input_jsonl
    else:
        sinks = [open(input_jsonl, "w")]

    written_total = 0
    try:
        for entry in manifest.entries:
            override = entry.agent_ref_override
            written = 0
            with open(entry.data) as source:
                for line in source:
                    if limit_per_entry is not None and written >= limit_per_entry:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    if override:
                        row = json.loads(line)
                        ref = row.get(AGENT_REF_KEY)
                        if not isinstance(ref, dict):
                            raise SweepValidationError(
                                f"[{entry.label}] cannot apply agent_ref_override to a row without agent_ref"
                            )
                        ref["name"] = override
                        line = json.dumps(row, ensure_ascii=False)
                    # Round-robin over the whole run, not per entry, so every shard stays
                    # mixed-environment and exercises the full agent fan-out.
                    sinks[written_total % len(sinks)].write(line + "\n")
                    written += 1
                    written_total += 1
            report.rows_per_entry[entry.label] = written
            if override:
                report.overrides_applied[entry.label] = override
    finally:
        for s in sinks:
            s.close()
    report.num_shards = len(sinks)

    report.config_paths = manifest.config_paths()
    report.num_repeats = manifest.num_repeats_by_agent()

    with open(config_yaml, "w") as handle:
        yaml.safe_dump(
            {"config_paths": report.config_paths, **manifest.gym_env_start.overlay()},
            handle,
            default_flow_style=False,
            sort_keys=False,
        )
    with open(report_json, "w") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    return report


def run_command(
    report: BuildReport,
    *,
    output_jsonl: str,
    policy_base_url: str = "<router-ip>:8000/v1",
    policy_model_name: str = "<checkpoint-path>",
    num_samples_in_parallel: int = 64,
) -> str:
    """Render the `gym eval run` invocation for a built sweep.

    ``--no-serve`` is not optional: without it the input path is silently discarded and replaced
    by the collated split, which is not what a sweep is for.
    """
    repeats = json.dumps(report.num_repeats).replace('"', "")
    return "\n".join(
        [
            "gym eval run --no-serve \\",
            f"    --config {report.config_yaml} \\",
            f"    --input {report.input_jsonl} \\",
            f"    --output {output_jsonl} \\",
            "    --resume \\",
            f"    ++num_repeats='{repeats}' \\",
            f"    ++num_samples_in_parallel={num_samples_in_parallel} \\",
            f"    ++policy_base_url={policy_base_url} \\",
            "    ++policy_api_key=dummy_api_key \\",
            f"    ++policy_model_name={policy_model_name}",
        ]
    )


# Placeholders so the composed config resolves with no endpoint and no secrets. env.yaml, when
# present, overrides them; without it a container build still validates.
CONTAINER_PLACEHOLDERS = {
    "policy_base_url": "dummy",
    "policy_api_key": "dummy",  # pragma: allowlist secret
    "policy_model_name": "dummy",
    "nv_inference_api_key": "dummy",  # pragma: allowlist secret
}


def container_config(manifests: List["SweepManifest"]) -> Dict:
    """Union every manifest's config_paths, for baking one venv per server implementation.

    The container is built once and serves every lane, so it must cover every server any lane
    might start. Deriving that from the manifests keeps it from drifting out of sync with them.
    """
    seen: Dict[str, None] = {}
    overlay: Dict = {}
    for manifest in manifests:
        for config in manifest.config_paths():
            seen.setdefault(config, None)
        # Overlays declare servers too -- the judge lane's model server exists only there. Omitting
        # them bakes no venv for it, and a server with no baked venv installs at runtime and hangs
        # the lane behind connection retries rather than failing.
        overlay.update(manifest.gym_env_start.overlay())
    return {"config_paths": list(seen), **overlay, **CONTAINER_PLACEHOLDERS}
