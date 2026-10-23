# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build one episode's ``SessionEvidence`` from the OpenCode exports and the plugins' receipts.

The steps are the OpenCode driver's after its run (exports enriched with hook receipts, evidence, model observations,
hook and plant receipts, continuation, model-visible projection), now fed by the plugin bundle instead of a driver.
"""

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import JsonValue

from resources_servers.agentic_if.episode.artifacts import save_json
from resources_servers.agentic_if.episode.bundle import CollectedBundle
from resources_servers.agentic_if.episode.opencode_context import actor_export
from resources_servers.agentic_if.episode.opencode_evidence import opencode_evidence
from resources_servers.agentic_if.episode.opencode_export import export_output
from resources_servers.agentic_if.episode.opencode_hooks import check_hook_receipts, enrich_exports
from resources_servers.agentic_if.episode.opencode_observations import bind_observations, project_output
from resources_servers.agentic_if.episode.opencode_replay import continued_session
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence, SessionInfo
from resources_servers.agentic_if.task_variants.builder import content_text
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec


LOG = logging.getLogger(__name__)

# An OpenCode session ID names the files its export is saved in.
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class EpisodeEvidence:
    """The graded evidence, the root session's export, and the receipt-enriched exports (None if merging failed)."""

    evidence: SessionEvidence
    root_export: dict | None
    enriched_exports: dict[str, dict] | None


def build_session_evidence(
    spec: RuntimeSpec,
    *,
    exports: dict[str, dict],
    sessions: Sequence[SessionInfo],
    hook_receipts: list[dict[str, JsonValue]],
    event_records: list[dict[str, JsonValue]],
    call_records: list[dict[str, JsonValue]],
    profile_sha256: str | None,
    replay: Mapping[str, JsonValue] | None,
    artifacts: Mapping[str, str],
    task_text: str | None = None,
) -> EpisodeEvidence:
    """Evidence of one episode; anything missing or inconsistent is an explicit issue, never a silent pass.

    ``artifacts`` names the saved OpenCode export of each session (by session ID) and the saved receipts. ``replay`` is
    the continuation receipt that seed recorded. ``task_text`` is the row's user text (``row_prompt_texts``); when it is
    given, the root session must have started from it.
    """
    artifacts = dict(artifacts)
    receipt_error = None
    try:
        enriched = merged = enrich_exports(exports, hook_receipts, spec=spec, observations=call_records)
    except (ValueError, TypeError, KeyError) as error:
        enriched, merged = exports, None
        receipt_error = f"hook_execution_receipts:{error}"
    roots = [session.id for session in sessions if session.parent_id is None]
    root = enriched.get(roots[0]) if len(roots) == 1 else None
    continued, continuation_issues = None, []
    if spec.continuation is not None:
        if len(roots) != 1:
            continuation_issues.append("missing_continued_root_session")
        elif replay is None:
            continuation_issues.append("missing_replay_receipt")
        else:
            continued = continued_session(
                spec.continuation,
                session_id=roots[0],
                export=root,
                receipt=replay,
                artifacts={"prefix_export": artifacts["prefix_export"]} if "prefix_export" in artifacts else {},
            )
    # The root is owned by this one non-interactive `opencode run` invocation:
    # it receives exactly the staged prompt, with no other live user ingress.
    # Delegated/frozen sessions do not gain this guarantee by resemblance.
    controlled = frozenset(
        (roots[0], message["info"]["id"])
        for message in (root or {}).get("messages", [])
        if continued is None or message["info"]["id"] not in continued.prefix_ids
    )
    evidence = opencode_evidence(
        root,
        child_exports=[value for key, value in enriched.items() if key not in roots],
        expected_sessions=sessions,
        tool_names=spec.tools.tool_names,
        interfaces=installed_interfaces(spec.tools.tool_interfaces),
        artifacts=artifacts,
        controlled_messages=controlled,
    )
    evidence = bind_observations(
        evidence,
        call_records,
        prefix_messages=frozenset((continued.session_id, identity) for identity in continued.prefix_ids)
        if continued is not None
        else frozenset(),
    )
    evidence = check_hook_receipts(
        evidence,
        records=hook_receipts,
        profile_sha256=profile_sha256,
        events=spec.events,
        event_records=event_records,
    )
    if receipt_error is not None:
        evidence.status = "incomplete"
        evidence.issues.append(receipt_error)
    if continuation_issues:
        evidence.status = "incomplete"
        evidence.issues.extend(continuation_issues)
    if task_text is not None and len(roots) == 1:
        issue = literal_task_issue(
            evidence, task_text, session_id=roots[0], prefix_ids=continued.prefix_ids if continued else frozenset()
        )
        if issue is not None:
            evidence.status = "incomplete"
            evidence.issues.append(issue)
    if continued is not None:
        evidence = continued.attach(evidence)
    for session_id, document in enriched.items():
        document = actor_export(
            document,
            controlled_ids=frozenset(identity for session, identity in controlled if session == session_id),
        )
        if continued is not None and session_id == continued.session_id:
            document = document | {
                "messages": [item for item in document["messages"] if item["info"]["id"] not in continued.prefix_ids]
            }
        projected, issues = project_output(
            export_output(document),
            call_records,
            session_id=session_id,
            compaction_messages=frozenset(
                (record["session_id"], record["message_id"])
                for record in evidence.facts.get("observed_opencode_compactions", [])
            ),
        )
        if issues:
            if evidence.status == "complete":
                evidence.status = "incomplete"
            evidence.issues.extend(f"training_capture:{session_id}:{issue}" for issue in issues)
        observed_outputs = {
            item.call_id: item.output for item in projected if getattr(item, "type", None) == "function_call_output"
        }
        for step in evidence.steps:
            if step.session_id == session_id and step.role == "tool" and step.call_id in observed_outputs:
                step.output = observed_outputs[step.call_id]
    return EpisodeEvidence(evidence=evidence, root_export=root, enriched_exports=merged)


def row_prompt_texts(messages: Sequence[Mapping[str, JsonValue]]) -> tuple[str | None, str]:
    """The text of a row's system message (None without one), and its user messages joined by a blank line."""
    systems = [content_text(message["content"]) for message in messages if message.get("role") == "system"]
    users = [content_text(message["content"]) for message in messages if message.get("role") == "user"]
    return (systems[0] if systems else None), "\n\n".join(users)


def literal_task_issue(
    evidence: SessionEvidence, task_text: str, *, session_id: str, prefix_ids: frozenset[str]
) -> str | None:
    """Check that the root session started from the row's literal user text.

    The first user message of the root session after any restored prefix must be that text, or contain it inside the
    harness's own framing. Returns the issue when it is not.
    """
    first = next(
        (
            step
            for step in evidence.steps
            if step.session_id == session_id and step.role == "user" and step.id not in prefix_ids
        ),
        None,
    )
    if first is not None and (first.text == task_text or (task_text and task_text in first.text)):
        return None
    return f"literal_task_not_presented:{session_id}"


def _file_name(session_id: str) -> str:
    """A file-name-safe form of an OpenCode session ID: the ID itself, or its digest."""
    return session_id if _SAFE_NAME.fullmatch(session_id) else hashlib.sha256(session_id.encode()).hexdigest()


def save_bundle_files(files: Mapping[str, bytes], directory: Path) -> dict[str, str]:
    """Keep the raw bundle files under ``directory/bundle``; returns each one's saved path by its bundle path."""
    saved = {}
    for relative, data in files.items():
        parts = PurePosixPath(relative).parts
        if not parts or PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in parts):
            LOG.warning("Not saving bundle file with an unsafe path %r", relative[:200])
            continue
        path = directory.joinpath("bundle", *parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        saved[relative] = str(path)
    return saved


def with_seed_facts(evidence: SessionEvidence, seed: Mapping[str, JsonValue]) -> SessionEvidence:
    """Add what seed observed: the working directory, and the initial workspace diff when it was taken."""
    if seed.get("workdir") is not None:
        evidence.facts.setdefault("workdir", seed["workdir"])
    if "workspace_initial_diff" in seed:
        evidence.facts["workspace_initial_diff"] = seed["workspace_initial_diff"]
    return evidence


def evidence_from_bundle(
    spec: RuntimeSpec,
    collected: CollectedBundle,
    *,
    profile_sha256: str | None,
    seed: Mapping[str, JsonValue],
    directory: Path,
    task_text: str | None = None,
) -> tuple[SessionEvidence, dict | None]:
    """Save what the bundle held under ``directory``, and build the episode's evidence from it.

    Returns the evidence and the root session's export. A model call the boundary could not forward makes the
    evidence an infrastructure error, as it made the episode's. A gap the bundle reported, a model call without the
    row's system prompt, and a root session that did not start from ``task_text`` make it incomplete. Runs in a worker
    thread: exports can be large.
    """
    files = save_bundle_files(collected.files, directory)
    artifacts = {}
    for session_id, document in collected.exports.items():
        path = files.get(f"exports/{session_id}.json")
        if path is None:
            path = str(directory / f"export-{_file_name(session_id)}.json")
            save_json(Path(path), document)
        artifacts[session_id] = path
    receipts = {
        name: files[relative]
        for name, relative in (
            ("hook_applied", "receipts/hook-applied.jsonl"),
            ("hook_events", "receipts/hook-events-fired.jsonl"),
            ("model_calls", "receipts/model-calls.jsonl"),
        )
        if relative in files and (name != "hook_events" or spec.events)
    }
    if spec.continuation is not None:
        save_json(directory / "prefix-export.json", spec.continuation.prefix_export)
        artifacts["prefix_export"] = str(directory / "prefix-export.json")
    try:
        built = build_session_evidence(
            spec,
            exports=collected.exports,
            sessions=collected.sessions,
            hook_receipts=collected.hook_receipts,
            event_records=collected.event_records,
            call_records=collected.call_records,
            profile_sha256=profile_sha256,
            replay=seed.get("replay"),
            artifacts=artifacts,
            task_text=task_text,
        )
    # Deliberate boundary: an export the builder cannot read is a recorded evidence failure, not a lost episode.
    except Exception as error:
        LOG.exception("Could not build agentic IF evidence in %s", directory)
        evidence = SessionEvidence(
            status="infrastructure_error",
            steps=[],
            issues=[f"evidence_build_failed:{type(error).__name__}: {error}"],
            artifacts=artifacts | receipts,
        )
        return with_seed_facts(evidence, seed), None
    evidence = built.evidence
    evidence.artifacts.update(receipts)
    for session_id, document in (built.enriched_exports or {}).items():
        path = directory / f"export-{_file_name(session_id)}-with-if-receipts.json"
        save_json(path, document)
        evidence.artifacts[f"raw_export:{session_id}"] = artifacts[session_id]
        evidence.artifacts[session_id] = str(path)
    if collected.boundary_issues:
        evidence.status = "infrastructure_error"
        evidence.issues = list(dict.fromkeys(evidence.issues + collected.boundary_issues))
    gaps = [f"bundle_gap:{gap}" for gap in collected.gaps] + collected.presentation_gaps
    if gaps:
        if evidence.status == "complete":
            evidence.status = "incomplete"
        evidence.issues = list(dict.fromkeys(evidence.issues + gaps))
    return with_seed_facts(evidence, seed), built.root_export


def collection_failed(reason: str, seed: Mapping[str, JsonValue]) -> SessionEvidence:
    """The evidence of an episode whose bundle could not be read at all."""
    return with_seed_facts(SessionEvidence(status="infrastructure_error", steps=[], issues=[reason]), seed)
