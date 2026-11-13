# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Find possible Gym results and collect public execution records for the judge.

Keep other workspace files available so the judge can inspect scripts and logs.
"""

import hashlib
import json
import math
import os
import shutil
import stat
import time
from pathlib import Path


PAGE_CHARS = 16000
SEARCH_CHARS = 2000


# Skip credentials, system files and runner records during workspace discovery.
# Agent logs are added separately as execution evidence.
EXCLUDED_PATHS = tuple(
    Path(value)
    for value in (
        "/proc",
        "/sys",
        "/dev",
        "/tests/test.sh",
        "/tests/verify.py",
        "/tests/reference.json",
        "/solution/solve.sh",
        "/logs/agent/codex.txt",
        "/logs/agent/trajectory.json",
        "/logs/agent/sessions",
        "/logs/agent/setup",
        "/logs/agent/oracle.txt",
        "/logs/agent/exit-code.txt",
        "/logs/verifier/report.json",
        "/logs/verifier/reward.txt",
        "/logs/verifier/reward.json",
        "/logs/verifier/test-stdout.txt",
        "/logs/verifier/test-stderr.txt",
        "/root/.codex/auth.json",
        "/root/.config/gcloud",
        "/root/.docker/config.json",
        "/etc/shadow",
        "/etc/gshadow",
        "/etc/ssl/private",
        "/etc/ssh",
        "/run/secrets",
    )
)


def label_excerpt(output):
    # Label a section by its file ID and range in the supplied evidence text.
    # The judge cites that ID without copying text; shortened sections get a new ID.
    output = dict(output)
    output.pop("excerpt_id", None)
    if output.get("text", "").strip():
        start = output["offset"]
        output["excerpt_id"] = f"{output['file_id']}@{start}:{start + len(output['text'])}"
    return output


def check(checks, name, passed):
    # Store whether a result check passed; raise an error when it did not.
    checks[name] = bool(passed)
    if not passed:
        raise ValueError(name)


def _has_model_output(output):
    # Return whether the model produced a tool call, reply, refusal or reasoning summary.
    # Blank text and tool results alone do not count.
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            return True
        if item.get("type") == "message" and item.get("role") == "assistant":
            parts = item.get("content")
        elif item.get("type") == "reasoning":
            parts = item.get("summary")
        else:
            continue
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "summary_text"}:
                text = part.get("text")
            elif part.get("type") == "refusal":
                text = part.get("refusal")
            else:
                continue
            if isinstance(text, str) and text.strip():
                return True
    return False


def _is_carlos(row, expected):
    # Return whether the row has the exact Carlos request as its only user message.
    # Read direct text and input_text parts so Gym's text layouts compare the same way.
    if not isinstance(row, dict):
        return False
    params = row.get("responses_create_params")
    inputs = params.get("input") if isinstance(params, dict) else None
    if isinstance(inputs, str):
        return inputs == expected["task_request"]
    if not isinstance(inputs, list) or any(not isinstance(item, dict) for item in inputs):
        return False
    user_messages = []
    for item in inputs:
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, list):
            if any(
                not isinstance(part, dict) or part.get("type") != "input_text" or not isinstance(part.get("text"), str)
                for part in content
            ):
                return False
            # Gym joins text parts without a separator. Do the same without
            # trimming or changing the request, so the comparison remains exact.
            content = "".join(part["text"] for part in content)
        user_messages.append(content)
    return user_messages == [expected["task_request"]]


def scored_rollout(rows, times, expected, checks):
    # Check the saved file's task, score and model response, then return its Carlos row.
    # Store each check for the judge and stop at the first failure.
    check(checks, "one_scored_carlos_rollout_saved", len(rows) == 1 and _is_carlos(rows[0], expected))
    row = rows[0]
    reward = row.get("reward")
    check(
        checks,
        "gym_reward_is_valid",
        not isinstance(reward, bool)
        and isinstance(reward, (int, float))
        and math.isfinite(reward)
        and 0 <= reward <= 1,
    )
    check(checks, "gym_verification_not_skipped", not row.get("verification_skipped", False))
    check(
        checks,
        "rollout_is_not_collection_failure",
        not row.get("failure_kind") and not row.get("failure_reason") and row.get("mask_sample", 0) == 0,
    )
    response = row.get("response")
    check(
        checks,
        "model_interaction_saved",
        isinstance(response, dict)
        and bool(response.get("id"))
        and bool(response.get("model"))
        and _has_model_output(response.get("output")),
    )
    digest = hashlib.sha256(json.dumps(response["output"], sort_keys=True).encode()).hexdigest()
    check(checks, "rollout_not_copied_from_bundled_samples", digest not in expected["bundled_output_hashes"])
    created = response.get("created_at")
    # Accept timestamps from one second before container start through
    # 120 seconds after verification begins.
    check(
        checks,
        "response_belongs_to_attempt",
        bool(times) and isinstance(created, (int, float)) and min(times) - 1 <= created <= max(times) + 120,
    )
    calls = {item.get("call_id") for item in response["output"] if item.get("type") == "function_call"}
    results = {item.get("call_id") for item in response["output"] if item.get("type") == "function_call_output"}
    # Every recorded tool call needs a matching result, even if the tool reported an error.
    check(checks, "recorded_tool_calls_have_results", calls <= results)
    return row


def _excluded(path):
    # Return whether a path should be skipped, such as credentials or runner records.
    name = str(path)
    shared_memory = name == "/dev/shm" or name.startswith("/dev/shm/")
    for root in EXCLUDED_PATHS:
        prefix = str(root)
        if prefix == "/dev" and shared_memory:
            continue
        if name == prefix or name.startswith(prefix + "/"):
            return True
    for part in path.parts:
        if part == ".env" or part.startswith(".env."):
            return True
        if part in {".ssh", ".aws", ".azure", ".kube", ".codex", ".netrc", ".git-credentials", ".npmrc", ".pypirc"}:
            return True
    return False


def saved_files(roots, diagnostics):
    # Walk the search directories and yield files once, skipping excluded paths and cycles.
    roots = tuple(Path(root).resolve() for root in roots)
    pending = [(root, None) for root in roots]
    visited = set()
    while pending:
        path, entry = pending.pop()
        try:
            if _excluded(path):
                continue
            if entry is None or entry.is_symlink():
                path = path.resolve()
                if _excluded(path):
                    continue
                info = path.stat()
            else:
                info = entry.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in visited:
                continue
            visited.add(identity)
            if stat.S_ISDIR(info.st_mode):
                # Do not follow a directory link outside the search roots.
                if not any(path.is_relative_to(root) for root in roots):
                    continue
                with os.scandir(path) as entries:
                    children = sorted(entries, key=lambda item: item.name, reverse=True)
                pending.extend((Path(item.path), item) for item in children)
            elif stat.S_ISREG(info.st_mode):
                diagnostics["files_seen"] += 1
                yield path
        except (OSError, RuntimeError):
            diagnostics["unreadable_paths"] += 1


def container_window():
    # Return container start and verification start times so the saved response's
    # created_at timestamp can be checked against this attempt's time window.
    now = time.time()
    try:
        fields = Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()
        lower = now - (time.clock_gettime(time.CLOCK_BOOTTIME) - int(fields[19]) / os.sysconf("SC_CLK_TCK"))
        if not math.isfinite(lower) or lower > now:
            raise ValueError("invalid_container_clock")
        return {"container_started_at": lower, "verifier_started_at": now}
    except (OSError, ValueError, IndexError, ZeroDivisionError):
        return {"container_started_at": None, "verifier_started_at": now}


def public_trace(path):
    # Extract public tool calls, results and session details from an outer-agent log.
    raw = path.read_text()
    if path.name == "trajectory.json":
        value = json.loads(raw)
        header = {
            key: value[key]
            for key in ("trajectory_id", "session_id", "thread_id", "task_id", "trial_id", "agent")
            if key in value
        }
        records = [{"trajectory_metadata": header}] if header else []
        records.extend(
            {
                key: step[key]
                for key in ("step_id", "timestamp", "result_timestamp", "thread_id", "tool_calls", "observation")
                if key in step
            }
            for step in value.get("steps", [])
            if isinstance(step, dict)
        )
    else:
        records = []
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            item, payload = event.get("item", {}), event.get("payload", {})
            if event.get("type") in {"thread.started", "session_meta"}:
                records.append(event)
            elif isinstance(item, dict) and item.get("type") in {"command_execution", "mcp_tool_call"}:
                records.append(event)
            elif (
                event.get("type") == "response_item"
                and isinstance(payload, dict)
                and payload.get("type")
                in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}
            ):
                records.append(event)
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def _has_execution_records(text: str) -> bool:
    # public_trace keeps identity metadata too; only execution entries should
    # suppress the existing reference-solution log fallback.
    for line in text.splitlines():
        record = json.loads(line)
        if record.get("tool_calls") or record.get("observation"):
            return True
        item = record.get("item")
        if isinstance(item, dict) and item.get("type") in {"command_execution", "mcp_tool_call"}:
            return True
        # public_trace includes response_item events only for tool calls/results.
        if record.get("type") == "response_item":
            return True
    return False


def result_facts(rows, expected, window):
    # Return the result checks reached before any failure; leave unrun checks out.
    checks = {}
    try:
        scored_rollout(rows, [window["container_started_at"], window["verifier_started_at"]], expected, checks)
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
        pass
    return checks


class EvidenceFiles:
    def __init__(
        self,
        expected,
        *,
        workspace=Path("/workspace"),
        agent_root=Path("/logs/agent"),
        solution=Path("/solution/solve.sh"),
        window=None,
    ):
        # Find possible results and collect public execution records for the judge.
        # Keep other files listed so scripts and logs can be read when needed.
        self.expected = expected
        self.window = container_window() if window is None else window
        self.files, self.texts, self.candidates, self.diagnostics = {}, {}, [], []
        self.returned_ranges = {}
        self.excerpts = {}
        self.workspace = Path(workspace).resolve()
        discovery = {"files_seen": 0, "unreadable_paths": 0}
        # Keep every discovered file listed for the judge.
        # Look for response or reward in JSON objects stored one per line.
        for path in saved_files((self.workspace,), discovery):
            file_id = self.add(path, "workspace")
            try:
                with path.open("rb") as stream:
                    first = stream.read(1)
                    while first and first in b" \t\r\n":
                        first = stream.read(1)
                    if first != b"{":
                        continue
                    stream.seek(0)
                    raw = stream.read()
                rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if not any(isinstance(row, dict) and ("response" in row or "reward" in row) for row in rows):
                    continue
                self.texts[file_id] = raw.decode("utf-8")
                self.files[file_id]["kind"] = "saved_output"
                for index, row in enumerate(rows):
                    if isinstance(row, dict) and ("response" in row or "reward" in row):
                        response = row.get("response")
                        response = response if isinstance(response, dict) else {}
                        self.candidates.append(
                            {
                                "file_id": file_id,
                                "row_index": index,
                                "created_at": response.get("created_at"),
                                "gym_reward": row.get("reward"),
                                "facts": result_facts(rows, expected, self.window),
                                "row": row,
                                "raw": raw,
                                "path": path,
                            }
                        )
            except (OSError, ValueError, UnicodeError, RecursionError, TypeError):
                continue
        # Read the commands Harbor's coding agent ran and the output those commands
        # returned, so the judge can check that the saved result came from a completed
        # Gym evaluation using the requested Gym agent.
        agent_root = Path(agent_root)
        trace_paths = [agent_root / "trajectory.json", agent_root / "codex.txt"]
        trace_paths.extend(sorted((agent_root / "sessions").rglob("*.jsonl")))
        has_execution_trace = False
        for path in trace_paths:
            if path.is_file():
                try:
                    text = public_trace(path)
                    if text:
                        self.add(path, "public_trace", text)
                        has_execution_trace = has_execution_trace or _has_execution_records(text)
                except (OSError, ValueError, UnicodeError, AttributeError):
                    self.diagnostics.append({"path": str(path), "reason": "optional_trace_unreadable"})
        # If no coding-agent records were collected, add oracle.txt, exit-code.txt and
        # solve.sh so the judge can review Harbor's reference-solution (oracle) run.
        # This can also happen if a regular Codex run has no usable logs.
        if not has_execution_trace:
            for path in (agent_root / "oracle.txt", agent_root / "exit-code.txt", Path(solution)):
                if path.is_file():
                    self.add(path, "oracle")
        # Put newest results first, then execution records and other workspace files.
        self.candidates.sort(key=self.candidate_order)
        output_ids = list(dict.fromkeys(item["file_id"] for item in self.candidates))
        output_order = {file_id: index for index, file_id in enumerate(output_ids)}
        kinds = {"saved_output": 0, "public_trace": 1, "oracle": 2, "workspace": 3}
        self.catalog = sorted(
            self.files.values(),
            key=lambda item: (
                kinds[item["kind"]],
                output_order.get(item["id"], 0),
                len(Path(item["path"]).parts),
                item["path"],
            ),
        )
        if discovery["unreadable_paths"]:
            self.diagnostics.append({"reason": "workspace_paths_unreadable", **discovery})

    def candidate_order(self, candidate):
        # Return a sort key that puts newer possible results first, regardless of reward.
        try:
            created = float(candidate["created_at"])
            created = created if math.isfinite(created) else 0
        except (TypeError, ValueError, OverflowError):
            created = 0
        return (-created, str(candidate["path"]), candidate["row_index"])

    def add(self, path, kind, text=None):
        # Assign an ID such as file:0 and store its path, category and any supplied text.
        file_id = f"file:{len(self.files)}"
        self.files[file_id] = {"id": file_id, "path": str(path), "kind": kind}
        if text is not None:
            self.texts[file_id] = text
        return file_id

    def text(self, file_id):
        # Return cached file text, reading it from disk the first time it is needed.
        if file_id not in self.files:
            raise ValueError("unknown_evidence_file")
        if file_id not in self.texts:
            value = Path(self.files[file_id]["path"]).read_text()
            if "\0" in value:
                raise ValueError("binary_evidence_file")
            self.texts[file_id] = value
        return self.texts[file_id]

    def read(self, file_id, offset, limit):
        # Return a page of file text and the offsets needed to read more.
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= PAGE_CHARS:
            raise ValueError("invalid_evidence_range")
        text = self.text(file_id)
        end = min(len(text), offset + limit)
        return label_excerpt(
            {
                "file_id": file_id,
                "offset": offset,
                "text": text[offset:end],
                "total_chars": len(text),
                "next_offset": end,
                "eof": end >= len(text),
                "previously_returned": [
                    {"offset": start, "end_offset": stop}
                    for start, stop in self.returned_ranges.get(file_id, [])
                    if start < end and stop > offset
                ],
            }
        )

    def search(self, file_id, query, offset, limit=SEARCH_CHARS):
        # Find the next exact text match and return nearby text with its offsets.
        if (
            not isinstance(query, str)
            or not query
            or type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not len(query) <= limit <= PAGE_CHARS
        ):
            raise ValueError("invalid_evidence_search")
        text = self.text(file_id)
        match = text.find(query, offset)
        if match < 0:
            return {"file_id": file_id, "found": False, "total_chars": len(text)}
        start = max(0, match - min(400, (limit - len(query)) // 2))
        return {
            "found": True,
            "match_offset": match,
            "match_end_offset": match + len(query),
            "next_search_offset": match + len(query),
            **self.read(file_id, start, limit),
        }

    def record_returned(self, output):
        # Keep exact delivered sections for citations and remember their ranges
        # so later reads can show overlap. Call this after any budget shortening.
        if not output.get("text") or "file_id" not in output:
            return
        file_id = output["file_id"]
        ranges = self.returned_ranges.get(file_id, []) + [(output["offset"], output["offset"] + len(output["text"]))]
        merged = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.returned_ranges[file_id] = merged
        if "excerpt_id" in output:
            self.excerpts[output["excerpt_id"]] = {
                "file_id": file_id,
                "path": self.files[file_id]["path"],
                "offset": output["offset"],
                "end_offset": output["offset"] + len(output["text"]),
                "quote": output["text"],
            }

    def tool(self, name, arguments):
        # Answer the judge's file-list, read or search request, or return an error.
        try:
            if name == "read_evidence":
                return self.read(**arguments)
            if name == "search_evidence":
                return self.search(**arguments)
            if name == "list_evidence":
                offset = arguments["offset"]
                if type(offset) is not int or offset < 0:
                    raise ValueError("invalid_evidence_range")
                values = self.catalog
                return {
                    "files": values[offset : offset + 100],
                    "offset": offset,
                    "next_offset": offset + 100,
                    "eof": offset + 100 >= len(values),
                }
            raise ValueError("unknown_evidence_tool")
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
            return {"error_type": type(error).__name__, "unavailable": True}

    def initial(self, max_chars):
        # Prepare the first evidence bundle, including whole result and execution files that fit.
        candidates = [
            {key: item[key] for key in ("file_id", "row_index", "created_at", "gym_reward", "facts")}
            for item in self.candidates
        ]
        value = {
            "contract": self.expected["verification_contract"],
            "task_request": self.expected["task_request"],
            "gym_agent": self.expected["gym_agent"],
            "files": self.catalog[:100],
            "file_count": len(self.files),
            "candidates_newest_first": candidates,
            "container_window": self.window,
            "diagnostics": self.diagnostics,
            "contents": [],
        }
        for descriptor in self.catalog:
            file_id = descriptor["id"]
            if descriptor["kind"] not in {"saved_output", "public_trace", "oracle"}:
                continue
            try:
                text = self.text(file_id)
            except (OSError, UnicodeError, ValueError):
                continue
            content = label_excerpt(
                {
                    "file_id": file_id,
                    "path": descriptor["path"],
                    "kind": descriptor["kind"],
                    "offset": 0,
                    "text": text,
                    "total_chars": len(text),
                    "eof": True,
                }
            )
            if len(json.dumps(value, ensure_ascii=False)) + len(json.dumps(content, ensure_ascii=False)) < max_chars:
                value["contents"].append(content)
                self.record_returned(content)
        return value

    def retain(self, directory):
        # Save possible-result files and matching aggregate metrics for later inspection.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in self.candidates:
            digest = hashlib.sha256(candidate["raw"]).hexdigest()
            destination = directory / digest
            destination.mkdir(exist_ok=True)
            (destination / "rollout.jsonl").write_bytes(candidate["raw"])
            path = candidate["path"]
            metrics = path.with_name(path.stem + "_aggregate_metrics.json")
            try:
                if metrics.resolve().is_relative_to(self.workspace) and metrics.is_file():
                    shutil.copyfile(metrics, destination / "aggregate_metrics.json")
            except (OSError, RuntimeError):
                self.diagnostics.append({"reason": "optional_metrics_unreadable"})

    def select(self, selected):
        # Find the possible result matching the judge's file ID and row index, or return None.
        return next(
            (
                item
                for item in self.candidates
                if item["file_id"] == selected.get("file_id") and item["row_index"] == selected.get("row_index")
            ),
            None,
        )
