# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read canonical Temporal scenario data without a CLI, credentials or network."""

import json
from pathlib import PurePosixPath

from responses_api_agents.ngc_recorded_context.readonly_tools import CoverageGap


def temporal_show(tokens: list[str], files: dict[str, str], world: dict) -> tuple[str, str]:
    """Return explicitly labeled fixture data for an exact recorded workflow/run."""
    arguments = list(tokens[1:])
    output_path = None
    if ">" in arguments:
        index = arguments.index(">")
        if index != len(arguments) - 2:
            raise CoverageGap("Unmodeled Temporal redirection")
        output_path = arguments[index + 1]
        if not output_path.startswith("/workspace/logs/") or ".." in PurePosixPath(output_path).parts:
            raise CoverageGap("Temporal output escapes replay workspace")
        arguments = arguments[:index]
    options, action = {}, []
    i = 0
    allowed = {"--address", "--namespace", "--api-key", "--workflow-id", "--run-id", "--output"}
    while i < len(arguments):
        token = arguments[i]
        if token == "--tls":
            i += 1
        elif token in allowed and i + 1 < len(arguments) and token not in options:
            options[token] = arguments[i + 1]
            i += 2
        elif not token.startswith("-"):
            action.append(token)
            i += 1
        else:
            raise CoverageGap("Unmodeled Temporal arguments")
    if action != ["workflow", "show"] or options.get("--output") != "json":
        raise CoverageGap("Only readonly Temporal workflow show JSON is modeled")
    connection = world["connection"]
    if options.get("--address") != connection["address"] or options.get("--namespace") != connection["namespace"]:
        raise CoverageGap("Temporal recording belongs to another address/namespace")
    workflow = options.get("--workflow-id")
    detail = world["workflow_details"].get(workflow)
    if detail is None or options.get("--run-id", detail["run_id"]) != detail["run_id"]:
        raise CoverageGap("No recording for this Temporal workflow/run")
    listed = next(
        w
        for group in world["workflows"].values()
        for w in group
        if w["workflow_id"] == workflow and w["run_id"] == detail["run_id"]
    )
    count, reported = len(detail["events"]), listed["history_length"]
    value = {
        "_replay": {
            "representation": "canonical scenario recording; not native Temporal CLI output or a live service response",
            "captured_event_count": count,
            "reported_history_length": reported,
            "complete_history": count == reported,
        },
        **detail,
    }
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if output_path is not None:
        files[output_path] = text
        return (
            f"Replay wrote canonical Temporal data to {output_path}: {count} captured events; source reports {reported}. No live query or credential expansion occurred.",
            "temporal_fixture_result",
        )
    return text, "temporal_fixture_result"
