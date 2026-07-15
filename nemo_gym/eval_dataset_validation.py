# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline structural and coverage checks for representative evaluation task sets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


ROLLOUT_ONLY_FIELDS = {
    "response",
    "reward",
    "terminated",
    "truncated",
    "_ng_task_index",
    "_ng_rollout_index",
}


@dataclass(frozen=True)
class EvalDatasetDiagnostic:
    severity: str
    code: str
    message: str
    line: int | None = None


@dataclass(frozen=True)
class EvalDatasetValidationResult:
    manifest: str
    dataset: str | None
    rows: int
    diagnostics: tuple[EvalDatasetDiagnostic, ...]

    @property
    def passed(self) -> bool:
        return not any(diagnostic.severity == "error" for diagnostic in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifest": self.manifest,
            "dataset": self.dataset,
            "rows": self.rows,
            "errors": sum(d.severity == "error" for d in self.diagnostics),
            "warnings": sum(d.severity == "warning" for d in self.diagnostics),
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
        }


def _error(code: str, message: str, line: int | None = None) -> EvalDatasetDiagnostic:
    return EvalDatasetDiagnostic("error", code, message, line)


def _warning(code: str, message: str, line: int | None = None) -> EvalDatasetDiagnostic:
    return EvalDatasetDiagnostic("warning", code, message, line)


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[EvalDatasetDiagnostic]]:
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:
        return None, [_error("manifest_unreadable", f"Could not read manifest: {exc}")]
    if not isinstance(data, dict):
        return None, [_error("manifest_shape", "Manifest root must be a mapping.")]
    if data.get("schema_version") != 1:
        return None, [_error("schema_version", "schema_version must be 1.")]
    for field in ("name", "use_case", "dataset", "split", "quality"):
        if not data.get(field):
            return None, [_error("manifest_field", f"Manifest is missing required field '{field}'.")]
    return data, []


def _load_rows(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[EvalDatasetDiagnostic]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    diagnostics: list[EvalDatasetDiagnostic] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, [_error("dataset_unreadable", f"Could not read dataset: {exc}")]

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            diagnostics.append(_error("invalid_json", f"Invalid JSON: {exc.msg}.", line_number))
            continue
        if not isinstance(row, dict):
            diagnostics.append(_error("row_shape", "Each JSONL row must be an object.", line_number))
            continue
        rows.append((line_number, row))

    if not rows and not diagnostics:
        diagnostics.append(_error("empty_dataset", "Dataset contains no task rows."))
    return rows, diagnostics


def validate_eval_dataset(manifest_path: str | Path) -> EvalDatasetValidationResult:
    """Validate objective eval-set requirements declared in a YAML manifest.

    This intentionally does not certify semantic correctness, contamination safety,
    verifier soundness, or true representativeness. Those require manual review.
    """

    manifest_path = Path(manifest_path).resolve()
    manifest, diagnostics = _load_manifest(manifest_path)
    if manifest is None:
        return EvalDatasetValidationResult(str(manifest_path), None, 0, tuple(diagnostics))

    dataset_config = manifest["dataset"]
    if not isinstance(dataset_config, dict) or not dataset_config.get("path"):
        diagnostics.append(_error("dataset_config", "dataset.path is required."))
        return EvalDatasetValidationResult(str(manifest_path), None, 0, tuple(diagnostics))

    dataset_path = (manifest_path.parent / str(dataset_config["path"])).resolve()
    rows, row_diagnostics = _load_rows(dataset_path)
    diagnostics.extend(row_diagnostics)

    task_id_fields = dataset_config.get("task_id_fields")
    if (
        not isinstance(task_id_fields, list)
        or not task_id_fields
        or not all(isinstance(f, str) for f in task_id_fields)
    ):
        diagnostics.append(_error("task_id_fields", "dataset.task_id_fields must be a non-empty list of fields."))
        task_id_fields = []

    has_default_agent = bool(dataset_config.get("default_agent"))
    has_prompt_config = bool(dataset_config.get("prompt_config"))
    identities: dict[tuple[Any, ...], int] = {}
    prompt_counts: Counter[str] = Counter()

    for line_number, row in rows:
        rollout_fields = sorted(ROLLOUT_ONLY_FIELDS.intersection(row))
        if rollout_fields:
            diagnostics.append(
                _error(
                    "rollout_fields",
                    f"Task row contains rollout-only fields: {', '.join(rollout_fields)}.",
                    line_number,
                )
            )

        create_params = row.get("responses_create_params")
        if not isinstance(create_params, dict):
            diagnostics.append(
                _error("responses_create_params", "responses_create_params must be an object.", line_number)
            )
        elif not has_prompt_config and not create_params.get("input"):
            diagnostics.append(
                _error(
                    "missing_input",
                    "responses_create_params.input is required without dataset.prompt_config.",
                    line_number,
                )
            )
        elif create_params.get("input") is not None:
            prompt_counts[json.dumps(create_params["input"], sort_keys=True)] += 1

        agent_ref = row.get("agent_ref")
        if not has_default_agent and (not isinstance(agent_ref, dict) or not agent_ref.get("name")):
            diagnostics.append(
                _error("missing_agent", "agent_ref.name is required without dataset.default_agent.", line_number)
            )

        if task_id_fields:
            missing = [field for field in task_id_fields if field not in row]
            if missing:
                diagnostics.append(
                    _error(
                        "missing_task_identity", f"Missing task identity fields: {', '.join(missing)}.", line_number
                    )
                )
            else:
                identity = tuple(json.dumps(row[field], sort_keys=True) for field in task_id_fields)
                if identity in identities:
                    diagnostics.append(
                        _error(
                            "duplicate_task_identity",
                            f"Task identity duplicates line {identities[identity]}.",
                            line_number,
                        )
                    )
                else:
                    identities[identity] = line_number

    duplicate_prompts = sum(count - 1 for count in prompt_counts.values() if count > 1)
    if duplicate_prompts:
        diagnostics.append(_warning("duplicate_prompts", f"Found {duplicate_prompts} duplicate prompt payload(s)."))

    split_config = manifest["split"]
    split_field = split_config.get("field") if isinstance(split_config, dict) else None
    allowed_splits = split_config.get("allowed") if isinstance(split_config, dict) else None
    if not split_field or not isinstance(allowed_splits, list) or not allowed_splits:
        diagnostics.append(_error("split_config", "split.field and a non-empty split.allowed list are required."))
    else:
        for line_number, row in rows:
            value = row.get(split_field)
            if value not in allowed_splits:
                diagnostics.append(
                    _error(
                        "held_out_split",
                        f"'{split_field}' must be one of {allowed_splits}; found {value!r}.",
                        line_number,
                    )
                )

    quality = manifest["quality"]
    min_tasks = quality.get("min_tasks") if isinstance(quality, dict) else None
    if not isinstance(min_tasks, int) or min_tasks < 1:
        diagnostics.append(_error("min_tasks", "quality.min_tasks must be a positive integer."))
    elif len(rows) < min_tasks:
        diagnostics.append(_error("task_count", f"Expected at least {min_tasks} tasks; found {len(rows)}."))

    for coverage in quality.get("coverage", []) if isinstance(quality, dict) else []:
        if not isinstance(coverage, dict) or not coverage.get("field"):
            diagnostics.append(_error("coverage_config", "Each quality.coverage rule requires a field."))
            continue
        field = coverage["field"]
        counts = Counter(row.get(field) for _, row in rows)
        required_values = coverage.get("required_values", [])
        missing_values = [value for value in required_values if counts[value] == 0]
        if missing_values:
            diagnostics.append(
                _error("coverage_values", f"Coverage field '{field}' is missing required values: {missing_values}.")
            )
        min_distinct = coverage.get("min_distinct")
        distinct = len([value for value in counts if value is not None])
        if isinstance(min_distinct, int) and distinct < min_distinct:
            diagnostics.append(
                _error(
                    "coverage_distinct",
                    f"Coverage field '{field}' needs {min_distinct} distinct values; found {distinct}.",
                )
            )
        min_count = coverage.get("min_count_per_value")
        if isinstance(min_count, int):
            values = required_values or [value for value in counts if value is not None]
            sparse = {str(value): counts[value] for value in values if counts[value] < min_count}
            if sparse:
                diagnostics.append(
                    _error(
                        "coverage_count",
                        f"Coverage field '{field}' needs {min_count} rows per value; below target: {sparse}.",
                    )
                )

    edge_config = quality.get("edge_cases", {}) if isinstance(quality, dict) else {}
    if edge_config:
        tag_field = edge_config.get("tag_field")
        required_tags = edge_config.get("required", [])
        observed_tags: set[str] = set()
        for _, row in rows:
            tags = row.get(tag_field, []) if tag_field else []
            if isinstance(tags, str):
                observed_tags.add(tags)
            elif isinstance(tags, list):
                observed_tags.update(str(tag) for tag in tags)
        missing_tags = [tag for tag in required_tags if tag not in observed_tags]
        if missing_tags:
            diagnostics.append(_error("edge_case_coverage", f"Missing required edge-case tags: {missing_tags}."))

    for field in quality.get("recommended_fields", []) if isinstance(quality, dict) else []:
        missing_count = sum(field not in row for _, row in rows)
        if missing_count:
            diagnostics.append(
                _warning(
                    "recommended_field",
                    f"Recommended field '{field}' is missing from {missing_count}/{len(rows)} rows.",
                )
            )

    return EvalDatasetValidationResult(str(manifest_path), str(dataset_path), len(rows), tuple(diagnostics))


def print_eval_dataset_result(result: EvalDatasetValidationResult, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return
    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.rows} task rows in {result.dataset or '<unknown>'}")
    for diagnostic in result.diagnostics:
        location = f" line {diagnostic.line}" if diagnostic.line is not None else ""
        print(f"[{diagnostic.severity.upper()}] {diagnostic.code}{location}: {diagnostic.message}")
    print("Manual review is still required for semantics, verifier correctness, contamination, and priority evals.")
