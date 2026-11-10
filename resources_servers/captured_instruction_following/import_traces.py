# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import captured requests without publishing their private source material.

Text continuations use the embedded-registry/history protocol of the existing
Agentic IF Realworld environment. This importer creates a separate cohort; it
does not alter that environment's frozen release or manufacture reviewed criteria.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path


TOOLS_MARKER = "APPLICABLE TOOL DEFINITIONS AND PARAMETER SCHEMAS (JSON):\n"
HISTORY_MARKER = "\n\nCAPTURED PRE-TARGET HISTORY (role-tagged JSON):\n"
SCHEMA_VERSION = "captured-instruction-following/v1"
COMPARISON_PROFILES = {"case_insensitive_scalars", "normalized_scalars"}
SEMANTIC_REQUEST_FIELDS = {"input", "tools", "instructions", "tool_choice", "parallel_tool_calls", "text"}
ASSISTANT_OUTPUT_TYPES = {"message", "reasoning", "function_call"}


def canonical_json(value: object) -> str:
    """Return finite, stable JSON used for all request and record bindings."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_sha256(value: object) -> str:
    """Hash a JSON value independently of file whitespace."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash original source bytes before reading a selected record."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _read_json(text: str) -> object:
    return json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)


def _object_file(path: Path) -> dict:
    value = _read_json(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = _read_json(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at JSONL line {number}")
            rows.append(value)
    return rows


def _identity(row: dict) -> tuple[str, str]:
    values = row.get("benchmark"), row.get("orig_trajectory_id")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Each inventory row needs benchmark and orig_trajectory_id")
    return values


def _references(path: Path | None) -> dict[tuple[str, str], dict]:
    result = {}
    if path is not None:
        for row in _jsonl(path):
            identity = _identity(row)
            if identity in result:
                raise ValueError("Duplicate identity in reference map")
            result[identity] = row
    return result


def _request(trace: dict) -> dict:
    request = trace.get("responses_create_params")
    if not isinstance(request, dict):
        raise ValueError("Source trace needs responses_create_params")
    if not isinstance(request.get("input"), (list, str)):
        raise ValueError("Only materialized list or string request input is supported")
    if "tools" in request and request["tools"] is not None and not isinstance(request["tools"], list):
        raise ValueError("Source tool definitions must be a list")
    if request.get("previous_response_id") or request.get("prompt"):
        raise ValueError("Resolve remote response/prompt references before importing")
    return request


def _group(row: dict, record: dict, request: dict) -> str:
    task_index = row.get("task_index", record.get("task_index"))
    if task_index is None:
        task_index = {key: request.get(key) for key in ("input", "tools", "instructions")}
    return "group-" + value_sha256({"benchmark": row["benchmark"], "source_task": task_index})[:24]


def _source_outputs(trace: dict) -> list[dict]:
    response = trace.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("output"), list):
        raise ValueError("Source trace needs a structured response.output array")
    outputs = response["output"]
    for item in outputs:
        if not isinstance(item, dict) or item.get("type") not in ASSISTANT_OUTPUT_TYPES | {"function_call_output"}:
            raise ValueError("Unsupported structured output item; do not discard it silently")
    return outputs


def captured_prefix(request: dict, outputs: list[dict]) -> tuple[list[dict], dict]:
    """Hold out the complete assistant turn after the last recorded tool result.

    Earlier output items, including their IDs and nested content, remain unchanged
    in the text prefix. Reasoning belonging to the held-out turn is never supplied
    as context. A text replay cannot establish effects of the earlier actions.
    """
    raw_input = request["input"]
    insertions = []
    if isinstance(raw_input, str):
        history = [{"role": "user", "type": "message", "content": raw_input}]
        insertions.append(
            {
                "history_index": 0,
                "source_pointer": "/responses_create_params/input",
                "operation": "string_input_to_user_record",
            }
        )
    else:
        if any(not isinstance(item, dict) for item in raw_input):
            raise ValueError("Captured input items must be objects")
        history = copy.deepcopy(raw_input)
    instructions = request.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise ValueError("Source instructions must be text")
        if instructions:
            for insertion in insertions:
                insertion["history_index"] += 1
            history.insert(0, {"role": "system", "type": "message", "content": instructions})
            insertions.append(
                {
                    "history_index": 0,
                    "source_pointer": "/responses_create_params/instructions",
                    "operation": "explicit_captured_instruction_record",
                }
            )
    result_positions = [index for index, item in enumerate(outputs) if item.get("type") == "function_call_output"]
    cutoff = result_positions[-1] + 1 if result_positions else 0
    if not outputs[cutoff:]:
        raise ValueError("No assistant continuation exists after the final tool result")
    if not any(item.get("type") in ASSISTANT_OUTPUT_TYPES for item in outputs[cutoff:]):
        raise ValueError("Held-out turn contains no assistant output")
    history.extend(copy.deepcopy(outputs[:cutoff]))
    boundary = {
        "selection": "complete_assistant_turn_after_last_tool_result" if cutoff else "complete_initial_assistant_turn",
        "source_output_prefix_items": cutoff,
        "held_out_begin_index": cutoff,
        "held_out_output_items": len(outputs) - cutoff,
        "held_out_types": [item["type"] for item in outputs[cutoff:]],
        "history_insertions": insertions,
        "earlier_action_correctness_retested": False,
    }
    return history, boundary


def _native_reference(trace: dict, row: dict, reference: dict | None) -> dict | None:
    metadata = trace.get("verifier_metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Source verifier_metadata must be an object")
    available = "expected_function" in metadata or "expected_params" in metadata
    if not available:
        if reference is not None:
            raise ValueError("Reference map supplies a call absent from the source verifier")
        return None
    function, params = metadata.get("expected_function"), metadata.get("expected_params")
    if not isinstance(function, str) or not function or not isinstance(params, dict):
        raise ValueError("Native reference needs a nonempty expected_function and expected_params object")
    reference = reference or row
    if "expected_call" in reference:
        call = reference["expected_call"]
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Reference expected_call must contain name and arguments")
        if value_sha256(call) != value_sha256({"name": function, "arguments": params}):
            raise ValueError("Reference map differs from the original expected call")
    for key, expected in (("expected_function", function), ("expected_params", params)):
        if key in reference and value_sha256(reference[key]) != value_sha256(expected):
            raise ValueError("Reference map differs from the original expected call")
    profile = reference.get("comparison_profile", reference.get("native_reference_profile"))
    if profile not in COMPARISON_PROFILES:
        raise ValueError(
            "Native reference requires an explicit supported comparison_profile in inventory/reference map"
        )
    return {
        "expected_function": copy.deepcopy(function),
        "expected_params": copy.deepcopy(params),
        "comparison_profile": profile,
    }


def convert_record(
    row: dict, record: dict, *, reference: dict | None = None, max_output_tokens: int = 32768, temperature: float = 0.7
) -> tuple[dict, dict, dict]:
    """Build one Gym row, private baseline, and private transformation receipt."""
    benchmark, identifier = _identity(row)
    if record.get("benchmark") != benchmark or record.get("trajectory_id") != identifier:
        raise ValueError("Annotated source identity differs from its inventory row")
    trace = record.get("source_trajectory")
    if not isinstance(trace, dict):
        raise ValueError("Annotated source record needs source_trajectory")
    request = _request(trace)
    outputs = _source_outputs(trace)
    native = _native_reference(trace, row, reference)
    task_id = "captured-" + value_sha256({"benchmark": benchmark, "orig_trajectory_id": identifier})[:24]
    identity = {
        "benchmark": benchmark,
        "orig_trajectory_id": identifier,
        "task_group_id": _group(row, record, request),
    }
    params = {"max_output_tokens": max_output_tokens, "temperature": temperature}
    if native is not None:
        params.update(
            {
                key: copy.deepcopy(value)
                for key, value in request.items()
                if key in SEMANTIC_REQUEST_FIELDS and value is not None
            }
        )
        params.setdefault("tools", [])
        boundary = {
            "selection": "initial_captured_request",
            "source_output_prefix_items": 0,
            "held_out_begin_index": 0,
            "held_out_output_items": len(outputs),
            "earlier_action_correctness_retested": False,
        }
        mode, criteria_status = "native_next_turn", "native_reference_available"
    else:
        history, boundary = captured_prefix(request, outputs)
        tools = request.get("tools") or []
        content = (
            "Generate the next assistant turn.\n"
            + TOOLS_MARKER
            + canonical_json(tools)
            + HISTORY_MARKER
            + canonical_json(history)
        )
        params.update(input=[{"role": "user", "type": "message", "content": content}], tools=[])
        mode, criteria_status = "single_turn_text_replay", "pending_review"
    baseline = {
        "row_id": task_id,
        **identity,
        "source_request": copy.deepcopy(request),
        "source_request_sha256": value_sha256(request),
        "original_response": copy.deepcopy(trace["response"]),
        "original_reward": copy.deepcopy(trace.get("reward", record.get("reward"))),
        "original_verifier_metadata": copy.deepcopy(trace.get("verifier_metadata", {})),
        "effective_verdict": copy.deepcopy(record.get("effective_verdict")),
        "final_resolver_ladder_applied": row.get("final_resolver_ladder_applied"),
        "continuation_boundary": boundary,
        "use": "private historical baseline; never a new response, target answer, or new reward",
    }
    verifier = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": mode,
        "source_request_sha256": value_sha256(request),
        "evaluation_request_sha256": value_sha256(params),
        "criteria_status": criteria_status,
        "private_baseline_ref": {
            "path": "baseline_outputs.jsonl",
            "row_id": task_id,
            "sha256": value_sha256(baseline),
        },
        "source_identity": identity,
    }
    if native is not None:
        verifier.update(native)
    prepared = {"task_id": task_id, **identity, "verifier_metadata": verifier, "responses_create_params": params}
    retained = (
        sorted(set(request) & SEMANTIC_REQUEST_FIELDS) if native is not None else ["input", "tools", "instructions"]
    )
    receipt = {
        "task_id": task_id,
        **identity,
        "evaluation_mode": mode,
        "source_request_sha256": value_sha256(request),
        "evaluation_request_sha256": value_sha256(params),
        "source_request_fields": sorted(request),
        "retained_semantic_fields": retained,
        "omitted_request_fields": {key: copy.deepcopy(value) for key, value in request.items() if key not in retained},
        "evaluation_sampling": {"max_output_tokens": max_output_tokens, "temperature": temperature},
        "sampling_policy": "Explicit evaluation defaults replace captured sampling/provider settings; every original field remains in the private baseline and omission receipt.",
        "source_input_sha256": value_sha256(request["input"]),
        "source_tools_sha256": value_sha256(request.get("tools")),
        "continuation_boundary": boundary,
        "source_refs": copy.deepcopy(row.get("source_refs", {})),
        "source_record_sha256": row.get("source_record_sha256"),
        "criteria_eligibility": "deterministic_next_call_reference"
        if native is not None
        else "unreviewed; action effects and unsupported text-scope claims remain unscored",
        "actual_tool_execution": False,
    }
    return prepared, baseline, receipt


def _write_line(stream: object, value: dict) -> None:
    stream.write(canonical_json(value) + "\n")


def _private_file(path: Path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def prepare_dataset(
    manifest_path: Path | str,
    output_dir: Path | str,
    *,
    reference_map_path: Path | str | None = None,
    expected_count: int | None = None,
    max_output_tokens: int = 32768,
    temperature: float = 0.7,
) -> dict:
    """Write a fresh private dataset without inference or source-data publication.

    Relative source_record_path values resolve against the inventory's parent.
    Hash mismatches, duplicate identities and unsupported boundary projections fail
    before the completed directory is published. Existing outputs are never reused.
    """
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Output directory exists; choose a fresh private destination")
    markers = [parent / ".git" for parent in [output_dir, *output_dir.parents]]
    if any(marker.is_file() or (marker.is_dir() and (marker / "HEAD").is_file()) for marker in markers):
        raise ValueError("Keep private imported data outside Git checkouts")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    if type(temperature) not in {int, float} or not 0 <= temperature <= 2:
        raise ValueError("temperature must lie in [0,2]")
    rows = _jsonl(manifest_path)
    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 1 or len(rows) != expected_count
    ):
        raise ValueError("Inventory count differs from expected_count")
    if not rows:
        raise ValueError("Inventory is empty")
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Inventory repeats a benchmark/original trajectory identity")
    references = _references(Path(reference_map_path).resolve() if reference_map_path is not None else None)
    if set(references) - set(identities):
        raise ValueError("Reference map contains identities outside the inventory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output_dir.name + ".", dir=output_dir.parent))
    counts, benchmarks, groups = Counter(), {}, set()
    try:
        with (
            _private_file(temporary / "prepared.jsonl") as prepared,
            _private_file(temporary / "baseline_outputs.jsonl") as baseline,
            _private_file(temporary / "provenance.jsonl") as provenance,
        ):
            for inventory in rows:
                source = inventory.get("source_record_path")
                expected = inventory.get("source_record_sha256")
                if not isinstance(source, str) or not source or not isinstance(expected, str) or len(expected) != 64:
                    raise ValueError("Inventory needs a source_record_path and source_record_sha256")
                source_path = Path(source)
                if not source_path.is_absolute():
                    source_path = manifest_path.parent / source_path
                if file_sha256(source_path) != expected:
                    raise ValueError("Source record hash differs from its inventory binding")
                reference = references.get(_identity(inventory))
                if reference is not None and reference.get("source_record_sha256", expected) != expected:
                    raise ValueError("Reference map binds a different source record")
                row, historical, receipt = convert_record(
                    inventory,
                    _object_file(source_path),
                    reference=reference,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )
                row["verifier_metadata"]["cohort_size"] = len(rows)
                _write_line(prepared, row)
                _write_line(baseline, historical)
                _write_line(provenance, receipt)
                mode = row["verifier_metadata"]["evaluation_mode"]
                counts[mode] += 1
                benchmarks.setdefault(row["benchmark"], Counter())[mode] += 1
                groups.add(row["task_group_id"])
            for stream in (prepared, baseline, provenance):
                stream.flush()
                os.fsync(stream.fileno())
        release = {
            "schema_version": SCHEMA_VERSION,
            "cohort_unit": "original_trajectory",
            "num_trajectories": len(rows),
            "num_task_groups": len(groups),
            "evaluation_modes": dict(sorted(counts.items())),
            "by_benchmark": {name: dict(value) for name, value in sorted(benchmarks.items())},
            "source_inventory_sha256": file_sha256(manifest_path),
            "reference_map_sha256": file_sha256(Path(reference_map_path)) if reference_map_path is not None else None,
            "artifacts": {
                name: {"sha256": file_sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
                for name in ("prepared.jsonl", "baseline_outputs.jsonl", "provenance.jsonl")
            },
            "actual_tool_execution": False,
            "new_model_calls": 0,
            "reviewed_text_criteria": 0,
            "existing_frozen_release_modified": False,
        }
        with _private_file(temporary / "release_manifest.json") as stream:
            stream.write(json.dumps(release, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Reserve the final path exclusively, then move only our completed files.
        output_dir.mkdir(mode=0o700)
        for path in temporary.iterdir():
            path.replace(output_dir / path.name)
        temporary.rmdir()
        return {
            "output_dir": str(output_dir),
            **release,
            "release_manifest_sha256": file_sha256(output_dir / "release_manifest.json"),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    """Command-line entry point for offline conversion of private trace inventories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-map", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    result = prepare_dataset(
        args.manifest,
        args.output_dir,
        reference_map_path=args.reference_map,
        expected_count=args.expected_count,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
