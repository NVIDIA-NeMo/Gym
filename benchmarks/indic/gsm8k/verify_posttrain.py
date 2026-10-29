# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check coverage, actual HTTP sampling, reasoning boundaries and GSM8K scores."""

import argparse
import json
from pathlib import Path

import yaml

from benchmarks.indic.gsm8k.prepare_posttrain import OUTPUT_TOKENS
from resources_servers.gsm8k_reference.app import exact_match, flexible_extract, strict_extract


PROMPT = Path("benchmarks/indic/gsm8k/prompts/posttrain.yaml")


def read_jsonl(path: Path) -> list[dict]:
    """Read LF-delimited JSON records without splitting Unicode inside strings."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(tasks: list[dict], rows: list[dict], model: dict, prompt: str) -> dict:
    """Recompute scores and check every rollout against its source and settings."""
    if not tasks:
        raise ValueError("Cannot validate an empty task set")
    errors, seen = [], set()
    strict = flexible = reasoning = capped = 0
    for row in rows:
        index = row.get("_ng_task_index")
        if type(index) is not int or index not in range(len(tasks)) or index in seen:
            errors.append(f"Duplicate/unknown index: {index}")
            continue
        seen.add(index)
        task = tasks[index]
        params, response = row.get("responses_create_params", {}), row.get("response", {})
        messages = params.get("input")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
            or messages[0].get("content") != prompt.format(question=task["question"])
            or params.get("instructions") != model.get("system_instruction")
            or row.get("expected_answer") != task["expected_answer"]
            or row.get("uuid") != task["uuid"]
        ):
            errors.append(f"Prompt/source mismatch: {index}")
        expected = {
            "max_output_tokens": OUTPUT_TOKENS,
            "temperature": model["sampling"]["temperature"],
            "top_p": model["sampling"]["top_p"],
        }
        if any(params.get(k) != v for k, v in expected.items()):
            errors.append(f"Generation settings mismatch: {index}")
        if (
            row.get("_ng_rollout_index") != 0
            or row.get("mask_sample")
            or row.get("failure_kind")
            or response.get("error")
            or response.get("status") not in {"completed", "incomplete"}
        ):
            errors.append(f"Failed rollout: {index}")
        outputs = response.get("output", [])
        reasoning += any(item["type"] == "reasoning" for item in outputs)
        final = "".join(
            part.get("text", "")
            for item in outputs
            if item["type"] == "message"
            for part in item.get("content", [])
            if part["type"] == "output_text"
        )
        if any(tag in final for tag in ("<think>", "</think>", "<channel|>")):
            errors.append(f"Reasoning boundary leaked into final answer: {index}")
        s = exact_match(strict_extract(final), task["expected_answer"])
        f = exact_match(flexible_extract(final), task["expected_answer"])
        if row.get("reward") != s or row.get("strict_match") != s or row.get("flexible_extract") != f:
            errors.append(f"Scorer mismatch: {index}")
        strict += s
        flexible += f
        tokens = (response.get("usage") or {}).get("output_tokens")
        if type(tokens) is not int or not 0 < tokens <= OUTPUT_TOKENS:
            errors.append(f"Invalid output token accounting: {index}")
        capped += (
            tokens == OUTPUT_TOKENS or (response.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
        )
    if len(rows) != len(tasks) or len(seen) != len(tasks):
        errors.append(f"Coverage mismatch: {len(rows)}/{len(tasks)}")
    return {
        "valid": not errors,
        "errors": errors,
        "expected": len(tasks),
        "observed": len(rows),
        "strict_correct": strict,
        "flexible_correct": flexible,
        "strict_accuracy_percent": 100 * strict / len(tasks),
        "flexible_accuracy_percent": 100 * flexible / len(tasks),
        "reasoning_responses": reasoning,
        "cap_hits": capped,
    }


def verify_run(run: Path, *, model_key: str, language: str) -> dict:
    """Validate a campaign shard, save its summary, and mark success with DONE."""
    work = run / "results" / model_key / language
    # Invalidate an earlier success if inputs have changed or a check now fails.
    (work / "DONE").unlink(missing_ok=True)
    tasks = read_jsonl(run / "data" / f"{language}.jsonl")
    rows = read_jsonl(work / "rollouts.jsonl")
    model = json.loads((run / "models.json").read_text(encoding="utf-8"))[model_key]
    prompt = yaml.safe_load((run / "source" / PROMPT).read_text(encoding="utf-8"))["user"]
    report = validate(tasks, rows, model, prompt)
    count = 0
    with (work / "transport.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") != "transport_request":
                continue
            body = event["request_payload"]
            expected = {"model": model["name"], "max_tokens": OUTPUT_TOKENS, **model["sampling"]}
            if any(body.get(k) != v for k, v in expected.items()):
                report["errors"].append("Actual HTTP sampling mismatch")
            if body.get("chat_template_kwargs", {}).get("enable_thinking") is not True or body.get("stop"):
                report["errors"].append("Actual HTTP thinking/stopping mismatch")
            count += 1
    if not count:
        report["errors"].append("No actual HTTP request evidence")
    if language == "smoke" and (report["reasoning_responses"] != len(tasks) or report["strict_correct"] == 0):
        report["errors"].append("Smoke must show separated reasoning and at least one correctly scored answer")
    report.update(valid=not report["errors"], actual_requests=count, model=model["name"], language=language)
    temporary = work / "summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(work / "summary.json")
    if report["errors"]:
        raise ValueError(report["errors"][:10])
    (work / "DONE").touch()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("model")
    parser.add_argument("language")
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run, model_key=args.model, language=args.language), indent=2))


if __name__ == "__main__":
    main()
