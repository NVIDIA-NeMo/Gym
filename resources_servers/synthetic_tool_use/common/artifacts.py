# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest-backed storage for resumable seed generation."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, TextIO

from resources_servers.synthetic_tool_use.common.models import (
    CustomerScenarioArtifact,
    DomainCandidate,
    DomainManifestEntry,
    RunManifest,
    SeedGenerationConfig,
    SeedToolSignature,
    StageState,
    utc_now,
)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, values: Iterable[Any]) -> None:
    lines = []
    for value in values:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        lines.append(json.dumps(value, ensure_ascii=False))
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))


class AdvisoryFileLock(AbstractContextManager["AdvisoryFileLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None

    def __enter__(self) -> AdvisoryFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class RunArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.domains_dir = run_dir / "domains"
        self.manifest_path = run_dir / "run_manifest.json"
        self._manifest_lock_path = run_dir / ".manifest.lock"

    def _manifest_lock(self) -> AdvisoryFileLock:
        return AdvisoryFileLock(self._manifest_lock_path)

    @classmethod
    def create(
        cls,
        config: SeedGenerationConfig,
        asset_hashes: dict[str, str] | None = None,
    ) -> RunArtifactStore:
        store = cls(config.output_dir)
        store.run_dir.mkdir(parents=True, exist_ok=True)
        store.domains_dir.mkdir(parents=True, exist_ok=True)
        if store.manifest_path.exists():
            existing = store.load_manifest()
            expected = RunManifest.from_config(config, asset_hashes)
            if existing.run_id != expected.run_id:
                raise ValueError(
                    f"output directory belongs to run {existing.run_id}, expected {expected.run_id}; use a new directory"
                )
        else:
            store.save_manifest(RunManifest.from_config(config, asset_hashes))
        return store

    def load_manifest(self) -> RunManifest:
        return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest.updated_at = utc_now()
        with self._manifest_lock():
            atomic_write_text(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")

    def register_domains(self, candidates: list[DomainCandidate]) -> RunManifest:
        with self._manifest_lock():
            manifest = self.load_manifest()
            existing_ids = {entry.domain_id for entry in manifest.domains}
            next_index = max((entry.source_index for entry in manifest.domains), default=-1) + 1
            new_candidates = []
            for candidate in candidates:
                if candidate.domain_id not in existing_ids:
                    new_candidates.append(candidate)
                    existing_ids.add(candidate.domain_id)

            final_max_index = next_index + len(new_candidates) - 1
            index_width = max(1, len(str(final_max_index)))
            for entry in manifest.domains:
                artifact_dir = f"{entry.source_index:0{index_width}d}"
                current_dir = self.domains_dir / entry.artifact_dir
                desired_dir = self.domains_dir / artifact_dir
                if current_dir != desired_dir:
                    if desired_dir.exists():
                        if current_dir.exists():
                            raise FileExistsError(f"cannot move {current_dir} to existing directory {desired_dir}")
                    else:
                        current_dir.rename(desired_dir)
                    entry.artifact_dir = artifact_dir

            for candidate in new_candidates:
                artifact_dir = f"{next_index:0{index_width}d}"
                entry = DomainManifestEntry(
                    domain_id=candidate.domain_id,
                    source_index=next_index,
                    name=candidate.name,
                    normalized_name=candidate.normalized_name,
                    artifact_dir=artifact_dir,
                )
                manifest.domains.append(entry)
                domain_dir = self.domains_dir / artifact_dir
                domain_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_json(domain_dir / "domain.json", candidate.model_dump(mode="json"))
                next_index += 1
            manifest.updated_at = utc_now()
            atomic_write_text(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")
            return manifest

    def record_asset_hashes(self, asset_hashes: dict[str, str]) -> None:
        with self._manifest_lock():
            manifest = self.load_manifest()
            normalized_hashes = dict(sorted(asset_hashes.items()))
            if manifest.asset_hashes and manifest.asset_hashes != normalized_hashes and manifest.domains:
                raise ValueError(
                    "generation profile assets changed after domains were registered; use a new run directory"
                )
            manifest.asset_hashes = normalized_hashes
            manifest.updated_at = utc_now()
            atomic_write_text(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")

    def domain_entry(self, domain_id: str) -> DomainManifestEntry:
        manifest = self.load_manifest()
        for entry in manifest.domains:
            if entry.domain_id == domain_id:
                return entry
        raise KeyError(domain_id)

    def domain_dir(self, domain_id: str) -> Path:
        return self.domains_dir / self.domain_entry(domain_id).artifact_dir

    def load_domain(self, domain_id: str) -> DomainCandidate:
        return DomainCandidate.model_validate_json(
            (self.domain_dir(domain_id) / "domain.json").read_text(encoding="utf-8")
        )

    def update_stage(
        self,
        domain_id: str,
        stage: str,
        state: StageState,
        *,
        attempts: int | None = None,
        failure_category: str | None = None,
        failure_detail: str | None = None,
        scenario_count: int | None = None,
    ) -> None:
        with self._manifest_lock():
            manifest = self.load_manifest()
            entry = next(item for item in manifest.domains if item.domain_id == domain_id)
            status = entry.stages[stage]
            status.state = state
            status.updated_at = utc_now()
            status.failure_category = failure_category
            status.failure_detail = failure_detail
            if attempts is not None:
                status.attempts = attempts
            if scenario_count is not None:
                entry.scenario_count = scenario_count
            manifest.updated_at = utc_now()
            atomic_write_text(self.manifest_path, manifest.model_dump_json(indent=2) + "\n")
            atomic_write_json(
                self.domains_dir / entry.artifact_dir / "stage_status.json", entry.model_dump(mode="json")
            )

    def write_attempt(self, domain_id: str, stage: str, attempt: int, payload: dict[str, Any]) -> Path:
        path = self.domain_dir(domain_id) / "attempts" / stage / f"attempt_{attempt:04d}.json"
        atomic_write_json(path, payload)
        return path

    def promote_policy_tools(
        self,
        domain_id: str,
        policy: str,
        tools: list[SeedToolSignature],
        quality_report: dict[str, Any],
    ) -> None:
        domain_dir = self.domain_dir(domain_id)
        atomic_write_text(domain_dir / "policy.md", policy.strip())
        atomic_write_jsonl(domain_dir / "tools.jsonl", tools)
        atomic_write_json(domain_dir / "quality_report.json", quality_report)

    def promote_scenarios(
        self,
        domain_id: str,
        run_name: str,
        scenarios: list[CustomerScenarioArtifact],
        scenarios_per_file: int,
    ) -> None:
        scenario_dir = self.domain_dir(domain_id) / "scenarios" / run_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for old_path in scenario_dir.glob("scenarios_*.jsonl"):
            old_path.unlink()
        for file_index, start in enumerate(range(0, len(scenarios), scenarios_per_file)):
            atomic_write_jsonl(
                scenario_dir / f"scenarios_{file_index:04d}.jsonl",
                scenarios[start : start + scenarios_per_file],
            )

    def write_generation_report(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        raw_domains_path = self.run_dir / "domains.raw.jsonl"
        raw_domains = raw_domains_path.read_text(encoding="utf-8").splitlines() if raw_domains_path.is_file() else []
        report = {
            "run_id": manifest.run_id,
            "run_name": manifest.run_name,
            "source_name": manifest.source_name,
            "generation_profile": manifest.generation_profile,
            "domains": len(manifest.domains),
            "domain_candidates": len(raw_domains),
            "domain_candidates_rejected": max(0, len(raw_domains) - len(manifest.domains)),
            "policy_tools": {},
            "scenarios": {},
            "scenario_count": sum(entry.scenario_count for entry in manifest.domains),
        }
        for stage in ("policy_tools", "scenarios"):
            counts: dict[str, int] = {}
            for entry in manifest.domains:
                state = entry.stages[stage].state.value
                counts[state] = counts.get(state, 0) + 1
            report[stage] = counts
        atomic_write_json(self.run_dir / "generation_report.json", report)
        return report
