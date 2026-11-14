# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent synthetic task and judge double; never used for benchmark scoring."""

import json
from pathlib import Path

from .core import HISTORY_MARKER, TOOLS_MARKER, Dataset, digest, judge, write


EXAMPLE_LITERALS = {
    "synthetic": "OK.",
    "synthetic-ready": "READY.",
    "synthetic-done": "DONE.",
    "synthetic-yes": "YES.",
    "synthetic-no": "NO.",
}


def create(directory: Path, *, examples: bool = False) -> None:
    """Build one plumbing task, or five independent literal tasks for Gym's public CI gate."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    literals = EXAMPLE_LITERALS if examples else {"synthetic": "OK."}
    records = {name: [] for name in ["inference_inputs.jsonl", "criteria.jsonl", "source_provenance.jsonl"]}
    exposure = []
    for identifier, literal in literals.items():
        task = _task(identifier, literal)
        for name, record in task.items():
            records[name].append(record)
        exposure.append(task["source_provenance.jsonl"]["conversation_id"] + ",False\n")
    for name, rows in records.items():
        with (directory / name).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (directory / "benchmark_exposure.csv").write_text(
        "conversation_id,training_or_detector_development_allowed\n" + "".join(exposure)
    )
    write(
        directory / "release_manifest.json",
        {
            "synthetic": True,
            "requested_items": len(literals),
            "artifacts": {name: {"sha256": digest(directory / name)} for name in [*records, "benchmark_exposure.csv"]},
        },
    )


def _task(identifier: str, literal: str) -> dict:
    instruction = "Return exactly " + literal
    history = [{"role": "user", "content": json.dumps({"text": instruction})}]
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Generate the next assistant turn.\n"
                + TOOLS_MARKER
                + "[]"
                + HISTORY_MARKER
                + json.dumps(history),
            }
        ]
    }
    constraint = {
        "constraint_id": "synthetic.literal",
        "subtype": "IF-FORMAT",
        "support_quotes": [instruction],
        "evaluation_question": "Does the entire answer equal " + literal + "?",
        "eligibility_verified": True,
        "visibility_verified": True,
        "observation_scope": "delivered_text",
        "support_origin": "input",
    }
    return {
        "inference_inputs.jsonl": {"id": identifier, **request},
        "criteria.jsonl": {
            "id": identifier,
            "constraints": [constraint],
            "scoring_view": "original_failure_focused",
        },
        "source_provenance.jsonl": {
            "id": identifier,
            "conversation_id": identifier + "-conversation",
            "benchmark_evaluation_role": "development_exposed",
            "detection_training_or_development_allowed": False,
            "actual_tool_execution": False,
            "input_sha256": judge.sha({"id": identifier, **request}),
            "original_split": "synthetic",
        },
    }


def delivery(dataset: Dataset, directory: Path, text: str, finish: str = "stop") -> dict:
    """Create an independent synthetic request and its authoritative raw receipt."""
    if not dataset.synthetic or set(dataset.inputs) != {"synthetic"}:
        raise ValueError("Fixture delivery accepts only the synthetic task")
    receipt_id = judge.sha({"fixture_text": text, "finish_reason": finish})
    upstream = {
        "model": "synthetic-policy",
        **dataset.request("synthetic"),
        "temperature": 0.7,
        "seed": 17,
        "max_tokens": 32768,
        "stream": True,
    }
    response = {
        "choices": [
            {
                "finish_reason": finish,
                "message": {"role": "assistant", "content": text},
            }
        ]
    }
    raw = directory / "model" / (receipt_id + ".a1.http")
    write(raw, response)
    write(
        raw.with_name(receipt_id + ".json"),
        {
            "status": "generated",
            "request_body": upstream,
            "request_sha256": judge.sha(upstream),
            "content_type": "application/json",
            "response": response,
            "raw_attempts": [{"path": str(raw), "sha256": digest(raw)}],
        },
    )
    return {
        **dataset.prepared_row("synthetic"),
        "response": {
            "id": receipt_id,
            "created_at": 0,
            "model": "synthetic-policy",
            "object": "response",
            "metadata": {"receipt_id": receipt_id},
            "tools": [],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "output": [
                {
                    "id": "fixture_message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed" if finish == "stop" else "incomplete",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
        },
    }


def review(payload: dict) -> dict:
    """Deterministic double for the independent literal-only synthetic task."""
    if payload["item_id"] not in EXAMPLE_LITERALS:
        raise ValueError("Synthetic reviewer cannot grade benchmark rows")
    literal = EXAMPLE_LITERALS[payload["item_id"]]
    message = payload["response"]["message"]
    field = "content" if message.get("content") else "refusal"
    text = message[field]
    passed = text == literal
    c = payload["constraints"][0]
    return {
        "policy_version": judge.VERSION,
        "item_id": payload["item_id"],
        "benchmark_mode": payload["benchmark_mode"],
        "scoring_scope": judge.SCOPES[payload["benchmark_mode"]],
        "item_verdict": "pass" if passed else "fail",
        **{
            k: payload[k]
            for k in [
                "input_sha256",
                "response_sha256",
                "constraint_contract_sha256",
                "benchmark_manifest_sha256",
            ]
        },
        "constraints": [
            {
                "constraint_id": c["constraint_id"],
                "subtype": c["subtype"],
                "support_quote": c["support_quotes"][0],
                "support_origin": "input",
                "verdict": "pass" if passed else "fail",
                "violation_established": not passed,
                "deviation_severity": "none" if passed else "minor",
                "rationale": "Synthetic exact literal check.",
                "response_evidence": {
                    "kind": "satisfying_span" if passed else "conflicting_span",
                    "field": field,
                    "quote": text,
                    "expected": literal,
                    "observation": "Synthetic delivered literal.",
                },
            }
        ],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    create(args.directory, examples=args.examples)
