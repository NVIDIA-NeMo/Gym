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
"""Dataset/agent decoupling migration (PR5 of the data-prep stack).

Migrates the repo from agent-coupled datasets to the task_source design:

  phase 1 (--move-datasets):   move `datasets:` lists from agent blocks into the
                               declaring resources-server block in the same file.
                               Agents with no structured resources_server.name edge
                               (self-contained harnesses: tau2, osworld, swe families,
                               ...) are left untouched and reported.
  phase 2 (--pin-benchmarks):  add an explicit `agent: <agent instance>` key to
                               `type: benchmark` dataset entries so the binding no
                               longer depends on which block declares the dataset.
  phase 3 (--convert-fanout):  report-only for now. Detects same-RS multi-agent
                               cross-product configs (genrm_compare,
                               jailbreak_detection) and emits the proposed run-level
                               fan-out YAML in the report. TODO: apply once the
                               fan-out schema is finalized in the resolver PR.
  phase 4 (--rewrite-jsonl):   strip `agent_ref` from committed dataset rows.
                               `--fold-task-data` additionally folds loose top-level
                               extras into `task_data` (off by default; schema pending
                               the unified-row RFC).
  phase 5 (always):            proof emitter. Writes a JSON report with per-file row
                               counts and content-hash manifests (hash over
                               responses_create_params + non-reserved extras, which is
                               invariant across this migration) for the golden-harness
                               bijection check.

Special cases:
  * remote-backed datasets (gitlab/huggingface identifiers, file not on disk) are
    skipped and reported — their uploaded rows need a separate re-upload pass or the
    strip-on-load shim.
  * legal_agent_bench: TODO stub (--fix-legal-agent-bench) — its harbor agent binds to
    the RS only via ${...} interpolations; needs a structured resources_server edge.
  * swe_agents_val/swe_agents_train alias configs: TODO stub (--drop-swe-aliases) —
    they exist only to satisfy rows with baked-in agent_ref and are deleted once rows
    are clean.

Default is a dry run: nothing is written except the report. Pass --apply to write.

Formatting note: when ruamel.yaml is importable we round-trip YAML preserving
comments and formatting. Otherwise we fall back to PyYAML, which normalizes
formatting and DROPS COMMENTS — do not run --apply with the PyYAML fallback on the
real repo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:  # pragma: no cover - environment-dependent
    from ruamel.yaml import YAML

    _ruamel = YAML()
    _ruamel.preserve_quotes = True
    _ruamel.width = 4096

    def load_yaml(path: Path) -> Any:
        with open(path) as f:
            return _ruamel.load(f)

    def dump_yaml(data: Any, path: Path) -> None:
        with open(path, "w") as f:
            _ruamel.dump(data, f)

    YAML_BACKEND = "ruamel"
except ImportError:  # pragma: no cover - environment-dependent
    import yaml as _pyyaml

    def load_yaml(path: Path) -> Any:
        with open(path) as f:
            return _pyyaml.safe_load(f)

    def dump_yaml(data: Any, path: Path) -> None:
        with open(path, "w") as f:
            _pyyaml.safe_dump(data, f, sort_keys=False)

    YAML_BACKEND = "pyyaml"


CONFIG_GLOBS = [
    "resources_servers/*/configs/*.yaml",
    "environments/*/config*.yaml",
    "benchmarks/**/config*.yaml",
    "responses_api_agents/*/configs/*.yaml",
]
DATA_GLOBS = [
    "resources_servers/*/data/*.jsonl",
    "environments/*/data/*.jsonl",
    "benchmarks/*/data/*.jsonl",
    "responses_api_agents/*/data/*.jsonl",
]
# Row keys that are part of the (current or reserved) row contract rather than
# task content. agent_ref is hashed OUT on both sides so the manifest is
# invariant across the migration.
RESERVED_ROW_KEYS = {
    "responses_create_params",
    "task_data",
    "task_id",
    "subset_for_metrics",
    "provenance",
    "task_source",
    "agent_ref",
}
# Known cross-product configs converted to run-level fan-out (phase 3).
FANOUT_CONFIG_HINTS = ("genrm_compare", "jailbreak_detection")


@dataclass
class Report:
    yaml_backend: str = YAML_BACKEND
    dry_run: bool = True
    configs_scanned: int = 0
    configs_changed: List[str] = field(default_factory=list)
    datasets_moved: int = 0
    benchmark_pins_added: int = 0
    self_contained_left: List[str] = field(default_factory=list)
    cross_file_rs_skipped: List[str] = field(default_factory=list)
    fanout_proposals: Dict[str, str] = field(default_factory=dict)
    jsonl_scanned: int = 0
    jsonl_changed: List[str] = field(default_factory=list)
    agent_refs_stripped: int = 0
    remote_backed_skipped: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    manifests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


def _agent_entries(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """[(top_level_key, inner_agent_config)] for every agent instance in a parsed file."""
    out = []
    for top, block in (cfg or {}).items():
        if isinstance(block, dict) and isinstance(block.get("responses_api_agents"), dict):
            inner_values = list(block["responses_api_agents"].values())
            if inner_values and isinstance(inner_values[0], dict):
                out.append((top, inner_values[0]))
    return out


def _rs_entries(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{top_level_key: inner_rs_config} for every resources-server instance in a parsed file."""
    out = {}
    for top, block in (cfg or {}).items():
        if isinstance(block, dict) and isinstance(block.get("resources_servers"), dict):
            inner_values = list(block["resources_servers"].values())
            if inner_values and isinstance(inner_values[0], dict):
                out[top] = inner_values[0]
    return out


def _structured_rs_name(agent_cfg: Dict[str, Any]) -> Optional[str]:
    rs = agent_cfg.get("resources_server")
    if isinstance(rs, dict):
        name = rs.get("name")
        if isinstance(name, str) and name and "???" not in name and "${" not in name:
            return name
    return None


def migrate_config_file(path: Path, report: Report, pin_benchmarks: bool, move_datasets: bool) -> bool:
    """Phases 1 + 2 for one file. Returns True if the parsed tree was modified."""
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        return False
    rs_by_name = _rs_entries(cfg)
    changed = False

    for agent_name, agent_cfg in _agent_entries(cfg):
        datasets = agent_cfg.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            continue

        if pin_benchmarks:
            for entry in datasets:
                if isinstance(entry, dict) and entry.get("type") == "benchmark" and "agent" not in entry:
                    entry["agent"] = agent_name
                    report.benchmark_pins_added += 1
                    changed = True

        if not move_datasets:
            continue
        rs_name = _structured_rs_name(agent_cfg)
        if rs_name is None:
            report.self_contained_left.append(f"{path}::{agent_name}")
            continue
        if rs_name not in rs_by_name:
            # Cross-file RS: the skeleton only rewrites same-file declarations.
            report.cross_file_rs_skipped.append(f"{path}::{agent_name}->{rs_name}")
            continue

        rs_cfg = rs_by_name[rs_name]
        existing = rs_cfg.get("datasets")
        if isinstance(existing, list):
            existing.extend(datasets)
        else:
            rs_cfg["datasets"] = datasets
        del agent_cfg["datasets"]
        report.datasets_moved += len(datasets)
        changed = True

    if changed:
        report.configs_changed.append(str(path))
        if not report.dry_run:
            dump_yaml(cfg, path)
    return changed


def propose_fanout(path: Path, report: Report) -> None:
    """Phase 3 (report-only): emit a proposed run-level fan-out block for same-RS
    multi-agent cross-product files. TODO: apply once the fan-out schema lands."""
    cfg = load_yaml(path)
    if not isinstance(cfg, dict):
        return
    by_rs: Dict[str, List[str]] = {}
    for agent_name, agent_cfg in _agent_entries(cfg):
        rs_name = _structured_rs_name(agent_cfg)
        if rs_name:
            by_rs.setdefault(rs_name, []).append(agent_name)
    for rs_name, agents in by_rs.items():
        if len(agents) >= 2:
            proposal = {"run": {"fan_out": {rs_name: agents}}}
            report.fanout_proposals[f"{path}::{rs_name}"] = json.dumps(proposal)
            report.todos.append(f"fan-out apply pending final schema: {path} rs={rs_name} agents={agents}")


def content_hash(row: Dict[str, Any]) -> str:
    """Hash of the task content: responses_create_params + task_data + non-reserved
    extras. Invariant to agent_ref removal and task_source stamping."""
    content = {k: v for k, v in row.items() if k not in RESERVED_ROW_KEYS}
    content["responses_create_params"] = row.get("responses_create_params")
    content["task_data"] = row.get("task_data")
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def rewrite_jsonl_file(path: Path, report: Report, fold_task_data: bool) -> bool:
    rows_before: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows_before.append(json.loads(line))
            except json.JSONDecodeError:
                report.todos.append(f"unparseable jsonl line skipped file-wide: {path}")
                return False

    rows_after = []
    stripped = 0
    for row in rows_before:
        new_row = copy.deepcopy(row)
        if "agent_ref" in new_row:
            del new_row["agent_ref"]
            stripped += 1
        if fold_task_data:
            extras = {k: new_row.pop(k) for k in list(new_row) if k not in RESERVED_ROW_KEYS}
            if extras:
                task_data = new_row.setdefault("task_data", {})
                task_data.update(extras)
        rows_after.append(new_row)

    report.manifests[str(path)] = {
        "rows_before": len(rows_before),
        "rows_after": len(rows_after),
        "content_hashes_before": sorted(content_hash(r) for r in rows_before),
        "content_hashes_after": sorted(content_hash(r) for r in rows_after),
    }
    if stripped == 0 and not fold_task_data:
        return False

    report.agent_refs_stripped += stripped
    report.jsonl_changed.append(str(path))
    if not report.dry_run:
        with open(path, "w") as f:
            for row in rows_after:
                f.write(json.dumps(row) + "\n")
    return True


def collect_remote_backed(config_paths: List[Path], report: Report, repo_root: Path) -> None:
    for path in config_paths:
        try:
            cfg = load_yaml(path)
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        for _, agent_cfg in _agent_entries(cfg):
            for entry in agent_cfg.get("datasets") or []:
                if not isinstance(entry, dict):
                    continue
                remote = entry.get("gitlab_identifier") or entry.get("huggingface_identifier") or entry.get("source")
                fpath = entry.get("jsonl_fpath")
                if remote and fpath and not (repo_root / fpath).exists():
                    report.remote_backed_skipped.append(f"{path}::{fpath}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is dry-run.")
    parser.add_argument("--report", type=Path, default=Path("migration_report.json"))
    parser.add_argument("--move-datasets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-benchmarks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--convert-fanout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rewrite-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fold-task-data", action="store_true", help="Fold loose extras into task_data (schema pending RFC)."
    )
    parser.add_argument("--fix-legal-agent-bench", action="store_true", help="TODO stub.")
    parser.add_argument("--drop-swe-aliases", action="store_true", help="TODO stub.")
    args = parser.parse_args(argv)

    report = Report(dry_run=not args.apply)
    if args.apply and YAML_BACKEND == "pyyaml":
        print(
            "REFUSING --apply with PyYAML fallback (comments would be dropped). Install ruamel.yaml.", file=sys.stderr
        )
        return 2

    config_paths = sorted({p for g in CONFIG_GLOBS for p in args.repo_root.glob(g) if p.is_file()})
    report.configs_scanned = len(config_paths)
    collect_remote_backed(config_paths, report, args.repo_root)
    for path in config_paths:
        try:
            if args.convert_fanout and any(h in str(path) for h in FANOUT_CONFIG_HINTS):
                propose_fanout(path, report)
                continue  # cross-product files are converted by hand once the schema lands
            migrate_config_file(path, report, pin_benchmarks=args.pin_benchmarks, move_datasets=args.move_datasets)
        except Exception as exc:  # keep the sweep going; surface in report
            report.todos.append(f"config migration failed: {path}: {exc}")

    if args.rewrite_jsonl:
        data_paths = sorted({p for g in DATA_GLOBS for p in args.repo_root.glob(g) if p.is_file()})
        report.jsonl_scanned = len(data_paths)
        for path in data_paths:
            try:
                rewrite_jsonl_file(path, report, fold_task_data=args.fold_task_data)
            except Exception as exc:
                report.todos.append(f"jsonl rewrite failed: {path}: {exc}")

    if args.fix_legal_agent_bench:
        report.todos.append("TODO: add structured resources_server edge to legal_agent_bench harbor agent")
    if args.drop_swe_aliases:
        report.todos.append("TODO: delete swe_agents_val/swe_agents_train alias configs once rows are clean")

    args.report.write_text(report.to_json())
    print(
        f"[{'DRY-RUN' if report.dry_run else 'APPLIED'}] configs: {len(report.configs_changed)}/{report.configs_scanned} would change, "
        f"datasets moved: {report.datasets_moved}, benchmark pins: {report.benchmark_pins_added}, "
        f"self-contained left: {len(report.self_contained_left)}, cross-file skips: {len(report.cross_file_rs_skipped)}, "
        f"fan-out proposals: {len(report.fanout_proposals)}, jsonl: {len(report.jsonl_changed)}/{report.jsonl_scanned} "
        f"({report.agent_refs_stripped} agent_refs stripped), remote-backed skipped: {len(report.remote_backed_skipped)}. "
        f"Report: {args.report}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
