# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Ask the LLM judge to assess saved Gym results and public execution records.

The judge requests file sections and assesses the workflow. This module checks
its answer format and evidence references, then saves the assessment and diagnostics.
"""

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from read_execution import label_excerpt


STATUSES = ("supported", "contradicted", "insufficient_evidence")
# These limits bound request count and request size.
MAX_REVIEW_TURNS = 16
# Leave room for continuation data and the final request's control information.
FINAL_RESERVE_CHARS = 20000
REQUEST_MARGIN_CHARS = 1000
# Load the review procedure and task criteria separately so they can be edited
# independently. Send both to the judge and save their combined text for review.
JUDGE_PROMPT = Path(__file__).with_name("judge-prompt.md").read_text(encoding="utf-8")
GRADING_CRITERIA = Path(__file__).with_name("grading-criteria.md").read_text(encoding="utf-8")
RUBRIC = JUDGE_PROMPT + "\n\n" + GRADING_CRITERIA


def write_json(path, value):
    # Save a JSON report, creating its parent directory if needed.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def response_schema(contract):
    # Build the judge's JSON answer format with one finding per task requirement.
    citation = {
        "type": "object",
        "properties": {"excerpt_id": {"type": "string"}},
        "required": ["excerpt_id"],
        "additionalProperties": False,
    }
    finding = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": list(STATUSES)},
            "explanation": {"type": "string"},
            "evidence": {"type": "array", "items": citation},
        },
        "required": ["status", "explanation", "evidence"],
        "additionalProperties": False,
    }
    selected = {
        "type": ["object", "null"],
        "properties": {"file_id": {"type": "string"}, "row_index": {"type": "integer"}},
        "required": ["file_id", "row_index"],
        "additionalProperties": False,
    }
    names = list(contract["requirements"])
    return {
        "type": "object",
        "properties": {
            "selected_result": selected,
            "findings": {
                "type": "object",
                "properties": {name: finding for name in names},
                "required": names,
                "additionalProperties": False,
            },
        },
        "required": ["selected_result", "findings"],
        "additionalProperties": False,
    }


def evidence_tools():
    # Describe the judge's tools for listing, reading and searching evidence files.
    definitions = (
        ("list_evidence", "List the next 100 available evidence files.", {"offset": {"type": "integer"}}),
        (
            "read_evidence",
            "Read up to 16000 characters starting at offset in an evidence file.",
            {"file_id": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
        ),
        (
            "search_evidence",
            "Find literal text at or after offset. Start with limit=2000; expand if needed.",
            {
                "file_id": {"type": "string"},
                "query": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
        ),
    )
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        }
        for name, description, properties in definitions
    ]


def serialized_chars(value):
    # Count characters in JSON text, including escaped text inside tool results.
    return len(json.dumps(value, ensure_ascii=False))


def tool_message(call_id, output):
    # Label a file-tool result with the judge's matching call ID.
    return {"type": "function_call_output", "call_id": call_id, "output": json.dumps(output, ensure_ascii=False)}


def bounded_tool_output(call_id, output, max_chars):
    # Shorten a file-tool result to fit the request and mark any withheld content.
    # Recompute citation IDs for shortened text so they identify only what is sent.
    output = label_excerpt(output)
    if serialized_chars(tool_message(call_id, output)) <= max_chars:
        return output
    limited = {**output, "budget_limited": True}
    if "text" in output:
        text, offset = output["text"], output["offset"]

        def section(length):
            # Keep the shorter section near a search match, or at the start of a normal read.
            match = output.get("match_offset", offset) - offset
            start = max(0, min(len(text) - length, match - length // 2)) if "match_offset" in output else 0
            end = offset + start + length
            value = {
                **limited,
                "offset": offset + start,
                "text": text[start : start + length],
                "next_offset": end,
                "eof": end >= output["total_chars"],
            }
            if "match_offset" in output:
                value["match_in_text"] = offset + start <= output["match_offset"] and end >= output["match_end_offset"]
            return label_excerpt(value)

        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if serialized_chars(tool_message(call_id, section(middle))) <= max_chars:
                low = middle
            else:
                high = middle - 1
        limited = section(low)
    elif "files" in output:
        limited["files"] = list(output["files"])
        while limited["files"] and serialized_chars(tool_message(call_id, limited)) > max_chars:
            limited["files"].pop()
        limited["next_offset"] = output["offset"] + len(limited["files"])
        limited["eof"] = False
    if serialized_chars(tool_message(call_id, limited)) <= max_chars:
        return limited
    return {"unavailable": True, "budget_limited": True, "reason": "final_assessment_reserved"}


def review_request(messages, result, contract, budget):
    # Build the next judge request with the evidence conversation and remaining review limits.
    instructions = RUBRIC + "\nReview budget: " + json.dumps(budget, ensure_ascii=False)
    if budget["phase"] == "final_assessment":
        instructions += "\nFile reading is finished. Return the final JSON assessment now."
    return {
        "model": result["model"],
        "max_output_tokens": 8192,
        "stream": False,
        "store": False,
        "instructions": instructions,
        "input": messages,
        "tools": evidence_tools(),
        "tool_choice": "none" if budget["phase"] == "final_assessment" else "auto",
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": result["effort"]},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gym_fm_02_judgment",
                "strict": True,
                "schema": response_schema(contract),
            }
        },
    }


def retry_delay(error, attempt):
    # Choose a delay of about two or four seconds, or a longer valid Retry-After value.
    delay = 2**attempt + random.uniform(0, 0.25)
    value = error.headers.get("Retry-After") if isinstance(error, HTTPError) and error.headers else None
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                return delay
        if math.isfinite(seconds):
            delay = max(delay, seconds)
    return delay


def request_response(body, *, deadline, attempts, directory, opener=urlopen, sleeper=time.sleep, clock=time.monotonic):
    # Send one judge request and record its attempts, retrying eligible errors while time remains.
    key = os.environ.get("VERIFIER_JUDGE_API_KEY", "")
    if not key:
        raise ValueError("judge_key_missing")
    base = os.environ.get("VERIFIER_JUDGE_BASE_URL", "https://inference-api.nvidia.com/v1").rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("judge_https_required")
    request = Request(
        base + "/responses",
        data=json.dumps(body, ensure_ascii=False).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    for number in range(1, 4):
        started = clock()
        remaining = deadline - started
        if remaining <= 0:
            raise ValueError("judge_time_budget_exhausted")
        record = {"request": len(attempts) + 1, "attempt": number, "status": "pending", "timeout_sec": remaining}
        attempts.append(record)
        write_json(Path(directory) / "request-attempts.json", attempts)
        try:
            with opener(request, timeout=remaining) as stream:
                # Reject responses over two million bytes before decoding their JSON.
                raw = stream.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("judge_response_too_large")
            try:
                response = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                raise ValueError("judge_response_envelope_not_json") from error
            if (
                not isinstance(response, dict)
                or response.get("status") != "completed"
                or response.get("error")
                or response.get("incomplete_details")
            ):
                raise ValueError("judge_response_incomplete")
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            record.update(status="request_error", error_type=type(error).__name__)
            # Retry connection errors, timeouts and these HTTP statuses; other HTTP errors stop.
            retryable = not isinstance(error, HTTPError) or error.code in {408, 500, 502, 503, 504}
            if isinstance(error, HTTPError):
                record["http_status"] = error.code
                # Keep the server's explanation before closing the response so
                # a rejected judge request can be diagnosed from the saved report.
                try:
                    error_body = error.read(16001) or b""
                    record["response_body"] = error_body[:16000].decode("utf-8", errors="replace")
                    record["response_body_truncated"] = len(error_body) > 16000
                except Exception as read_error:
                    # A failed diagnostic read must leave the original HTTP error
                    # and its retry handling intact.
                    record["response_body_error"] = type(read_error).__name__
            delay = retry_delay(error, number) if retryable and number < 3 else 0
            if isinstance(error, HTTPError):
                error.close()
            if not retryable or number == 3:
                raise
            if delay >= deadline - clock():
                record["retry_skipped"] = "time_budget_exhausted"
                raise
            record["retry_delay_sec"] = delay
        except Exception as error:
            record.update(status="response_error", error_type=type(error).__name__)
            raise
        else:
            record["status"] = "completed"
            return response
        finally:
            record["elapsed_sec"] = clock() - started
            write_json(Path(directory) / "request-attempts.json", attempts)
        sleeper(delay)


def public_answer(response):
    # Return the judge's public answer text, rejecting incomplete messages and refusals.
    messages = [item for item in response.get("output", []) if item.get("type") == "message"]
    if not messages or any(item.get("status") != "completed" for item in messages):
        raise ValueError("judge_response_incomplete")
    blocks = [block for item in messages for block in item.get("content", [])]
    if any(block.get("type") == "refusal" for block in blocks):
        raise ValueError("judge_response_refused")
    text = "".join(block.get("text", "") for block in blocks if block.get("type") == "output_text")
    fenced = re.fullmatch(r"\s*```(?:json)?\s*\n?(.*?)\n?```\s*", text, re.DOTALL | re.IGNORECASE)
    return fenced[1] if fenced else text


def validate_judgment(judgment, evidence):
    # Check the answer format, selected row and references to delivered excerpts.
    # The judge decides whether those excerpts support the workflow.
    names = set(evidence.expected["verification_contract"]["requirements"])
    if not isinstance(judgment, dict) or set(judgment) != {"selected_result", "findings"}:
        raise ValueError("judge_answer_invalid")
    findings = judgment["findings"]
    if not isinstance(findings, dict) or set(findings) != names:
        raise ValueError("judge_requirements_mismatch")
    cited_files = set()
    for finding in findings.values():
        if (
            not isinstance(finding, dict)
            or set(finding) != {"status", "explanation", "evidence"}
            or finding["status"] not in STATUSES
            or not isinstance(finding["explanation"], str)
            or not isinstance(finding["evidence"], list)
        ):
            raise ValueError("judge_finding_invalid")
        if finding["status"] != "insufficient_evidence" and not finding["evidence"]:
            raise ValueError("judge_citation_missing")
        for citation in finding["evidence"]:
            if not isinstance(citation, dict) or set(citation) != {"excerpt_id"}:
                raise ValueError("judge_citation_invalid")
            excerpt_id = citation["excerpt_id"]
            if not isinstance(excerpt_id, str) or excerpt_id not in evidence.excerpts:
                raise ValueError("judge_citation_not_found")
            cited_files.add(evidence.excerpts[excerpt_id]["file_id"])
    supported = all(item["status"] == "supported" for item in findings.values())
    selected = judgment["selected_result"]
    candidate = None
    if selected is not None:
        if (
            not isinstance(selected, dict)
            or set(selected) != {"file_id", "row_index"}
            or type(selected["row_index"]) is not int
        ):
            raise ValueError("judge_selected_result_invalid")
        candidate = evidence.select(selected)
        if candidate is None:
            raise ValueError("judge_selected_result_not_found")
    if supported:
        if candidate is None or candidate["file_id"] not in cited_files:
            raise ValueError("judge_selected_result_not_cited")
    return supported, candidate


def review(evidence, *, report_dir, artifacts, requester=request_response, clock=time.monotonic, deadline=None):
    # Ask the judge to assess the saved results and execution records, then return its report.
    started = clock()
    report_dir, artifacts = Path(report_dir), Path(artifacts)
    contract = evidence.expected["verification_contract"]
    result = {
        "status": "pending",
        "hybrid_reward": None,
        "pare_success": False,
        "contract_version": contract["version"],
        "gym_reward": None,
        "gym_agent": evidence.expected["gym_agent"],
        "model": os.environ.get("VERIFIER_JUDGE_MODEL", "azure/openai/gpt-6.1-terra"),
        "effort": os.environ.get("VERIFIER_JUDGE_REASONING_EFFORT", "high"),
        "saved_results": [
            {key: item[key] for key in ("file_id", "row_index", "gym_reward", "facts")} for item in evidence.candidates
        ],
        "request_attempts": [],
        "responses": [],
    }
    write_json(report_dir / "report.json", result)
    try:
        # Keep the discovered results even if the judge request later fails.
        evidence.retain(artifacts)
        # Harbor supplies the task's timing settings; keep their defaults in task.toml.
        timeout = float(os.environ["VERIFIER_JUDGE_TIME_BUDGET_SEC"])
        final_reserve = float(os.environ["VERIFIER_JUDGE_FINAL_RESERVE_SEC"])
        max_chars = int(os.environ.get("VERIFIER_JUDGE_MAX_EVIDENCE_CHARS", "300000"))
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or not math.isfinite(final_reserve)
            or final_reserve <= 0
            or not 1000 <= max_chars <= 1000000
        ):
            raise ValueError("judge_budget_invalid")
        deadline = min(started + timeout, deadline) if deadline is not None else started + timeout
        reserve_chars = min(FINAL_RESERVE_CHARS, max_chars // 5)
        reserve_seconds = min(final_reserve, timeout / 4)
        reading_deadline = deadline - reserve_seconds
        initial = evidence.initial(max_chars // 3)
        write_json(report_dir / "evidence.json", initial)
        write_json(report_dir / "contract.json", contract)
        (report_dir / "rubric.txt").write_text(RUBRIC)
        messages = [{"role": "user", "content": json.dumps(initial, ensure_ascii=False)}]
        result["input_sha256"] = hashlib.sha256(messages[0]["content"].encode()).hexdigest()
        accesses = []
        final_reason = None
        # Let the judge request more files until it gives an answer or reaches a review limit.
        for turn in range(1, MAX_REVIEW_TURNS + 1):
            remaining = deadline - clock()
            if not final_reason and remaining <= reserve_seconds:
                final_reason = "time_reserved"
            if turn == MAX_REVIEW_TURNS:
                final_reason = final_reason or "last_request"
            budget = {
                "phase": "final_assessment" if final_reason else "reading",
                "remaining_seconds": max(0, round(remaining, 3)),
                "remaining_read_seconds": max(0, round(remaining - reserve_seconds, 3)),
                "final_reserve_seconds": reserve_seconds,
                "remaining_requests": MAX_REVIEW_TURNS + 1 - turn,
                "max_request_chars": max_chars,
                "final_request_reserve_chars": reserve_chars,
            }
            body = review_request(messages, result, contract, budget)
            capacity = max_chars - reserve_chars - serialized_chars(body) - REQUEST_MARGIN_CHARS
            if not final_reason and capacity < REQUEST_MARGIN_CHARS:
                final_reason = "input_reserved"
                budget["phase"] = "final_assessment"
            budget["remaining_evidence_chars"] = max(0, capacity)
            body = review_request(messages, result, contract, budget)
            request_chars = serialized_chars(body)
            if request_chars > max_chars:
                raise ValueError("judge_input_budget_exceeded")
            result.setdefault("review_budget", []).append(
                {**budget, "turn": turn, "request_chars": request_chars, "final_reason": final_reason}
            )
            request_deadline = deadline if final_reason else reading_deadline
            response = requester(
                body, deadline=request_deadline, attempts=result["request_attempts"], directory=report_dir
            )
            result["responses"].append(
                {
                    "response_id": response.get("id"),
                    "returned_model_label": response.get("model"),
                    "usage": response.get("usage"),
                }
            )
            calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
            if calls and final_reason:
                raise ValueError("judge_tools_after_final_assessment")
            # A response without file-tool calls must contain the final assessment.
            if not calls:
                answer = public_answer(response)
                (report_dir / "answer.txt").write_text(answer)
                try:
                    judgment = json.loads(answer)
                except ValueError as error:
                    raise ValueError("judge_answer_not_json") from error
                supported, candidate = validate_judgment(judgment, evidence)
                result.update(
                    status="completed",
                    hybrid_reward=int(supported),
                    pare_success=supported,
                    judgment=judgment,
                    citations={
                        citation["excerpt_id"]: evidence.excerpts[citation["excerpt_id"]]
                        for finding in judgment["findings"].values()
                        for citation in finding["evidence"]
                    },
                )
                statuses = {finding["status"] for finding in judgment["findings"].values()}
                # At a review limit, unresolved requirements leave the review ungraded
                # unless another requirement is contradicted.
                if final_reason and "insufficient_evidence" in statuses and "contradicted" not in statuses:
                    result.update(
                        status="judge_error",
                        hybrid_reward=None,
                        pare_success=False,
                        reason="judge_review_budget_exhausted",
                    )
                if candidate:
                    result.update(selected_result=judgment["selected_result"], gym_reward=candidate["gym_reward"])
                    if supported:
                        (artifacts / "rollout.jsonl").write_bytes(candidate["raw"])
                        digest = hashlib.sha256(candidate["raw"]).hexdigest()
                        metrics = artifacts / digest / "aggregate_metrics.json"
                        try:
                            if metrics.is_file():
                                shutil.copyfile(metrics, artifacts / "aggregate_metrics.json")
                        except OSError:
                            result.setdefault("artifact_diagnostics", []).append(
                                {"reason": "optional_metrics_unreadable"}
                            )
                break
            # Keep the response in memory so the next request can continue the same review.
            messages.extend(response["output"])
            first_output = len(messages)
            withheld = {"unavailable": True, "budget_limited": True, "reason": "final_assessment_reserved"}
            messages.extend(tool_message(call["call_id"], withheld) for call in calls)
            # Give every file-tool call a matching result before adding the requested evidence.
            for index, call in enumerate(calls):
                try:
                    arguments = json.loads(call["arguments"])
                except (ValueError, TypeError) as error:
                    raise ValueError("judge_tool_arguments_invalid") from error
                position = first_output + index
                capacity = max_chars - reserve_chars - serialized_chars(body) - REQUEST_MARGIN_CHARS
                allowance = max(0, capacity // (len(calls) - index)) + serialized_chars(messages[position])
                if clock() >= reading_deadline:
                    output = withheld
                    final_reason = "time_reserved"
                elif allowance < REQUEST_MARGIN_CHARS:
                    output = withheld
                else:
                    output = bounded_tool_output(call["call_id"], evidence.tool(call["name"], arguments), allowance)
                messages[position] = tool_message(call["call_id"], output)
                evidence.record_returned(output)
                access = {
                    "turn": turn,
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": arguments,
                    "output": output,
                }
                accesses.append(access)
            write_json(report_dir / "file-access.json", accesses)
        else:
            raise ValueError("judge_turn_budget_exhausted")
    except HTTPError as error:
        result.update(
            status="judge_error",
            hybrid_reward=None,
            pare_success=False,
            error_type="HTTPError",
            http_status=error.code,
        )
    except Exception as error:
        result.update(status="judge_error", hybrid_reward=None, pare_success=False, error_type=type(error).__name__)
        if isinstance(error, ValueError) and str(error).startswith("judge_"):
            result["reason"] = str(error)
    result["elapsed_sec"] = clock() - started
    write_json(report_dir / "report.json", result)
    return result
