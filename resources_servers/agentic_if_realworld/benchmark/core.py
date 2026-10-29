# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable adapter around the unchanged strict-v2.2 policy."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from .policy import validate_response_judge as judge


TOOLS_MARKER = "APPLICABLE TOOL DEFINITIONS AND PARAMETER SCHEMAS (JSON):\n"
HISTORY_MARKER = "\n\nCAPTURED PRE-TARGET HISTORY (role-tagged JSON):\n"
ROOT = Path(__file__).parent
DEFAULT_JUDGE_MODEL = "judge-model"
DEFAULT_DATA_DIR = Path.home() / ".cache/nemo-gym/agentic_if_realworld/1.0.0"
BASE_INPUT_STRINGS = judge.input_strings
TRANSPORT_VERSION = "exact-payload-binding-enum-transport-v1"
PHASES = ((17, 32768), (17, 65536), (17, 131072), (29, 131072), (43, 131072))


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_policy_bytes(name: str) -> bytes:
    """Return original policy bytes before the explicitly added public SPDX header."""
    raw = (ROOT / "policy" / name).read_bytes()
    if name in {"validate_response_judge.py", "test_contract.py"}:
        prefix = (
            b"# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            b"# SPDX-License-Identifier: Apache-2.0\n"
        )
        if not raw.startswith(prefix):
            raise ValueError("Missing public policy license header")
        return raw[len(prefix) :]
    return raw


def write(path: Path, value: object) -> None:
    """Atomically write private receipts; never write credentials."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def indexed(path: Path, key: str = "id") -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate IDs in dataset")
    return result


def captured_context(request: dict) -> tuple[list, list, str]:
    if set(request) != {"messages"} or len(request["messages"]) != 1:
        raise ValueError("Expected unchanged one-user-message wrapper")
    message = request["messages"][0]
    if message["role"] != "user" or not isinstance(message["content"], str):
        raise ValueError("Invalid wrapper transport")
    content = message["content"]
    if content.count(TOOLS_MARKER) != 1 or content.count(HISTORY_MARKER) != 1:
        raise ValueError("Missing or ambiguous wrapper boundaries")
    tool_text, history_text = content.split(TOOLS_MARKER, 1)[1].split(HISTORY_MARKER, 1)
    tools, history = json.loads(tool_text), json.loads(history_text)
    if not isinstance(tools, list) or not isinstance(history, list):
        raise TypeError("Invalid registry/history")
    return tools, history, tool_text


def decoded_strings(value: object, depth: int = 0):
    # Exact extraction from recover_agentic_if_969_unscored: no rewriting.
    if depth > 8:
        return
    if isinstance(value, str):
        yield value
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return
        if decoded != value:
            yield from decoded_strings(decoded, depth + 1)
    elif isinstance(value, dict):
        for child in value.values():
            yield from decoded_strings(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from decoded_strings(child, depth + 1)


def input_strings(request: dict, mode: str):
    yield from BASE_INPUT_STRINGS(request, mode)
    if mode == "single_turn_text_replay":
        tools, history, _ = captured_context(request)
        yield from judge.strings(tools)
        yield from judge.strings(history)
        yield from decoded_strings(tools)
        yield from decoded_strings(history)


judge.input_strings = input_strings


def provider_schema(canonical: dict) -> dict:
    result = copy.deepcopy(canonical)

    def visit(node):
        if isinstance(node, dict):
            node.pop("allOf", None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def bounded_schema(canonical: dict, payload: dict) -> dict:
    result = copy.deepcopy(canonical)
    for key in [
        "item_id",
        "benchmark_mode",
        "input_sha256",
        "response_sha256",
        "constraint_contract_sha256",
        "benchmark_manifest_sha256",
    ]:
        result["properties"][key]["enum"] = [payload[key]]
    result["properties"]["policy_version"]["enum"] = [judge.VERSION]
    result["properties"]["scoring_scope"]["enum"] = [judge.SCOPES[payload["benchmark_mode"]]]
    array = result["properties"]["constraints"]
    array["minItems"] = array["maxItems"] = len(payload["constraints"])
    props = array["items"]["properties"]
    for field, values in [
        ("constraint_id", [c["constraint_id"] for c in payload["constraints"]]),
        ("subtype", [c["subtype"] for c in payload["constraints"]]),
        (
            "support_quote",
            [q for c in payload["constraints"] for q in c["support_quotes"]],
        ),
        (
            "support_origin",
            [c.get("support_origin", "input") for c in payload["constraints"]],
        ),
    ]:
        props[field]["enum"] = sorted(set(values))
    return result


class Dataset:
    """Load the internal release, optionally overridden by an exact private copy."""

    def __init__(self, directory: Path | str | None = None, *, synthetic: bool = False):
        self.directory = directory = Path(directory or os.environ.get("AGENTIC_IF_DATA_DIR") or DEFAULT_DATA_DIR)
        self.synthetic = synthetic
        self.release = read(directory / "release_manifest.json")
        self.pin = read(ROOT / "source_manifest.json")
        if not synthetic and digest(directory / "release_manifest.json") != self.pin["release_manifest_sha256"]:
            raise ValueError("Release manifest hash mismatch")
        if synthetic and self.release.get("synthetic") is not True:
            raise ValueError("Synthetic mode requires an explicitly synthetic release")
        for name, expected in self.release["artifacts"].items():
            if digest(directory / name) != expected["sha256"]:
                with (directory / name).open("rb") as stream:
                    if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                        raise ValueError(
                            "Private artifact is not materialized; run Gym preparation with private registry access"
                        )
                raise ValueError("Dataset artifact hash mismatch: " + name)
        self.inputs = indexed(directory / "inference_inputs.jsonl")
        self.criteria = indexed(directory / "criteria.jsonl")
        self.provenance = indexed(directory / "source_provenance.jsonl")
        self.manifest_sha256 = digest(directory / "release_manifest.json")
        if set(self.inputs) != set(self.criteria) or set(self.inputs) != set(self.provenance):
            raise ValueError("Input/criteria/provenance cohort mismatch")
        if len(self.inputs) != self.release["requested_items"] or (not synthetic and len(self.inputs) != 969):
            raise ValueError("Declared task count mismatch")
        seen = set()
        tools_count = Counter()
        for identifier, row in self.inputs.items():
            request = {k: v for k, v in row.items() if k != "id"}
            tools, _, _ = captured_context(request)
            tools_count[bool(tools)] += 1
            p = self.provenance[identifier]
            if p["conversation_id"] in seen:
                raise ValueError("Duplicate source conversation")
            seen.add(p["conversation_id"])
            if (
                p["benchmark_evaluation_role"] != "development_exposed"
                or p["detection_training_or_development_allowed"] is not False
                or p["actual_tool_execution"] is not False
                or p["input_sha256"] != judge.sha(row)
            ):
                raise ValueError("Exposure or source input binding mismatch")
            if not self.criteria[identifier]["constraints"]:
                raise ValueError("Missing frozen criteria")
        with (directory / "benchmark_exposure.csv").open() as stream:
            exposure = list(csv.DictReader(stream))
        # The overlay covers 2170 source cases; select membership, never redefine cohort.
        selected = [r for r in exposure if r.get("conversation_id") in seen]
        if {r["conversation_id"] for r in selected} != seen:
            raise ValueError("Exposure overlay coverage mismatch")
        if any(r.get("training_or_detector_development_allowed", "").lower() != "false" for r in selected):
            raise ValueError("Exposure overlay permits prohibited development")
        self.report = {
            "items": len(self.inputs),
            "atomic_checks": sum(len(r["constraints"]) for r in self.criteria.values()),
            "views": dict(Counter(r["scoring_view"] for r in self.criteria.values())),
            "nonempty_registries": tools_count[True],
            "empty_registries": tools_count[False],
            "manifest_sha256": self.manifest_sha256,
            "evaluation_only": True,
            "input_sha256": digest(directory / "inference_inputs.jsonl"),
            "criteria_sha256": digest(directory / "criteria.jsonl"),
        }
        if not synthetic and (
            self.report["views"] != {"original_failure_focused": 912, "supplemental_active_single_turn": 57}
            or self.report["atomic_checks"] != 2803
            or tools_count != {True: 817, False: 152}
        ):
            raise ValueError("Frozen accounting mismatch")

    def judge_prompt(self) -> str:
        """Load the frozen private policy; synthetic fixtures use independent instructions."""
        if self.synthetic:
            return (ROOT / "synthetic-judge-prompt.md").read_text(encoding="utf-8")
        path = self.directory / "judge_prompt.md"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.pin["policy_assets"]["judge_prompt.md"]:
            raise ValueError("Frozen private judge prompt hash mismatch")
        return content.decode("utf-8")

    def request(self, identifier: str) -> dict:
        return {k: v for k, v in self.inputs[identifier].items() if k != "id"}

    def prepared_row(self, i: str) -> dict:
        return {
            "task_id": i,
            "verifier_metadata": {
                "item_id": i,
                "dataset_sha256": self.manifest_sha256,
                "input_sha256": judge.sha(self.request(i)),
            },
            "responses_create_params": {
                "input": self.request(i)["messages"],
                "tools": [],
                "temperature": 0.7,
                "max_output_tokens": 32768,
            },
        }

    def prepared(self) -> list[dict]:
        return [self.prepared_row(i) for i in sorted(self.inputs)]

    def payload(self, identifier: str, response: dict, binding: dict) -> dict:
        request = self.request(identifier)
        payload = {
            "item_id": identifier,
            "benchmark_mode": "single_turn_text_replay",
            "input_request": request,
            "response": response,
            "constraints": self.criteria[identifier]["constraints"],
            "bindings_verified": True,
            "benchmark_manifest_sha256": self.manifest_sha256,
            "input_sha256": judge.sha(request),
            "response_sha256": judge.sha(response),
            "constraint_contract_sha256": judge.contract_sha(self.criteria[identifier]["constraints"]),
            "inference_binding": binding,
        }
        tools, _, quote = captured_context(request)
        payload.update(
            text_tool_registry=tools,
            text_tool_registry_quote=quote,
            text_tool_registry_sha256=judge.sha(tools),
        )
        judge.validate_payload(payload)
        return payload


def unscored(identifier: str, reason: str) -> dict:
    return {
        "item_id": identifier,
        "policy_version": judge.VERSION,
        "scoring_status": "not_scored",
        "model_verdict": None,
        "item_verdict": None,
        "reason": reason,
        "reward": 0.0,
        "mask_sample": True,
        "failure_kind": "agentic_if_realworld:unscored",
        "failure_reason": reason,
    }


def summarize(records: list[dict], dataset: Dataset) -> dict:
    result = judge.summarize_records(records, sorted(dataset.inputs))
    by_id = {r["item_id"]: r for r in records}
    result["views"] = {}
    for view in sorted({r["scoring_view"] for r in dataset.criteria.values()}):
        ids = [i for i, r in dataset.criteria.items() if r["scoring_view"] == view]
        result["views"][view] = judge.summarize_records([by_id[i] for i in ids if i in by_id], ids)
    result["subtypes"] = {}
    for subtype in sorted({c["subtype"] for r in dataset.criteria.values() for c in r["constraints"]}):
        ids = [i for i, r in dataset.criteria.items() if any(c["subtype"] == subtype for c in r["constraints"])]
        selected = []
        for i in ids:
            r = by_id.get(i)
            if r and r["scoring_status"] == "scored":
                selected.append({**r, "item_verdict": r["subtype_verdicts"][subtype]})
            elif r:
                selected.append(r)
        result["subtypes"][subtype] = judge.summarize_records(selected, ids)
    result["subtype_denominators_overlap"] = True
    return result


def summary_metrics(summary: dict) -> dict:
    """Flatten frozen full-cohort accounting for standard Gym and adaptive-driver reports."""
    fields = ["expected_items", "pass", "fail", "not_scored", "unattempted", "scoring_coverage", "complete"]
    metrics = {key: summary[key] for key in fields}
    metrics.update(
        {
            "composite/complete_accuracy": summary["complete_accuracy"],
            "masked": summary["not_scored"] - summary["unattempted"],
            "subtype_denominators_overlap": summary["subtype_denominators_overlap"],
        }
    )
    for group, prefix in [("views", ""), ("subtypes", "subtypes/")]:
        for name, values in summary[group].items():
            label = prefix + name + "/"
            metrics.update({label + key: values[key] for key in [*fields, "complete_accuracy"]})
            metrics[label + "masked"] = values["not_scored"] - values["unattempted"]
    return metrics


def source_preflight(
    split_registry: Path | str | None = None,
    ground_truth: Path | str | None = None,
    *,
    directory: Path | str | None = None,
) -> dict:
    """Use the pinned release audit, or recheck both original source artifacts."""
    pin = read(ROOT / "source_manifest.json")
    if bool(split_registry) != bool(ground_truth):
        raise ValueError("Supply both source split registry and ground truth")
    receipt = Path(directory or os.environ.get("AGENTIC_IF_DATA_DIR") or DEFAULT_DATA_DIR) / "source_audit.json"
    if digest(receipt) != pin["source_audit_sha256"]:
        raise ValueError("Packaged source audit hash mismatch")
    audit = read(receipt)
    if not split_registry:
        return {**audit, "source_validation": "frozen_release_audit"}
    split_registry, ground_truth = Path(split_registry), Path(ground_truth)
    if digest(split_registry) != audit["split_sha256"] or digest(ground_truth) != audit["ground_truth_sha256"]:
        raise ValueError("Source split or ground truth hash mismatch")
    with split_registry.open() as stream:
        rows = list(csv.DictReader(stream))
    train = {r["conversation_id"] for r in rows if r["split"] == "train"}
    test = {r["conversation_id"] for r in rows if r["split"] in {"test", "historical_test"}}
    if train & test or len(train) != audit["canonical_train_ids"] or len(test) != audit["canonical_test_ids"]:
        raise ValueError("Prohibited split overlap or changed split counts")
    return {
        "canonical_train_ids": len(train),
        "canonical_test_ids": len(test),
        "canonical_train_test_overlap": 0,
        "split_sha256": digest(split_registry),
        "ground_truth_sha256": digest(ground_truth),
        "labeling_protocols_by_original_split": audit["labeling_protocols_by_original_split"],
        "new_exclusions": 0,
        "training_development_thresholds_prompt_examples": "none",
        "source_validation": "original_source_bytes",
    }


def runtime_identity() -> dict:
    """Record dependency versions and actual source bytes in resume fingerprints."""
    import importlib.metadata
    import sys

    from nemo_gym.orchestration.jobs import installed_gym_commit

    # Creation time is a receipt field, not part of the stable fingerprint.
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "nemo-gym",
            "jsonschema",
            "pydantic",
            "aiohttp",
            "openai",
        ]
    }
    sources = {str(p.relative_to(ROOT)): digest(p) for p in ROOT.rglob("*.py")}
    component_root = ROOT.parents[2]
    for boundary in ["benchmarks", "resources_servers", "responses_api_models", "responses_api_agents", "nemo_gym"]:
        for p in (component_root / boundary).rglob("*.py"):
            if ".venv" not in p.parts:
                sources["gym/" + str(p.relative_to(component_root))] = digest(p)
    return {
        "python": sys.version,
        "dependencies": versions,
        "sources": sources,
        "gym_commit": installed_gym_commit(),
        "source_gym_commit": read(ROOT / "source_manifest.json")["gym_commit"],
    }
