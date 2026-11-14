# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded API continuation with auditable, data-only tool execution."""

import copy
import gzip
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from resources_servers.partner_if.native_evidence import LiveAgentCapture
from responses_api_agents.ngc_recorded_context.delegation import explore
from responses_api_agents.ngc_recorded_context.observation_hooks import tool_observation
from responses_api_agents.ngc_recorded_context.paths import GYM_ROOT
from responses_api_agents.ngc_recorded_context.prefix import ordering_outcome
from responses_api_agents.ngc_recorded_context.replay_fixtures import Fixtures, canonical_path, shell_tokens
from responses_api_agents.ngc_recorded_context.runtime_contract import RuntimeContract, read_runtime_contract


COMPLETE = {"first_answer_draft", "final_response"}


class ReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_contract_path: Path | None = None
    runtime_contract_sha256: str | None = None
    package_dir: Path
    artifacts_dir: Path
    coverage_profile: Literal["recorded-v1", "expanded-v2", "expanded-v3", "expanded-v4"] = "recorded-v1"
    evaluation_boundary: Literal["first_answer", "ordering_prefix_v1"] = "first_answer"
    max_new_turns: int = Field(default=8, ge=1, le=32)
    max_policy_calls: int = Field(default=8, ge=1, le=64)
    missing_runtime_reads: bool = False
    gym_reasoning_decode: bool = False
    shell_control_flow: bool = False
    capture_stdout: bool = False
    temporal_history: bool = False
    max_delegate_turns: int = Field(default=6, ge=1, le=12)
    max_tokens: int = Field(default=8192, ge=1, le=8192)
    temperature: float = 0.7
    top_p: float = 0.95
    timeout_s: int = Field(default=180, ge=1, le=180)

    @field_validator("runtime_contract_path", "package_dir", "artifacts_dir")
    @classmethod
    def checkout_paths(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        path = value.expanduser()
        return (path if path.is_absolute() else GYM_ROOT / path).resolve()


def actor_identity(params: dict) -> dict:
    """Allow Gym to omit completed message status; retain all actor content."""
    identity = copy.deepcopy(params)
    for item in identity.get("input", []):
        if item.get("type", "message") == "message" and item.get("status") == "completed":
            item.pop("status")
    return identity


def write_json(path: Path, value, secret: str = "") -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if secret:
        text = text.replace(secret, "[REDACTED]")
    path.write_text(text)
    path.chmod(0o600)


def load_runtime(config: ReplayConfig) -> RuntimeContract:
    if config.runtime_contract_path is not None and config.runtime_contract_sha256 is None:
        raise ValueError("An external replay contract requires runtime_contract_sha256")
    path = config.runtime_contract_path or config.package_dir / "ngc-recorded-context/runtime.json"
    return read_runtime_contract(path, expected_sha256=config.runtime_contract_sha256)


def load_inputs(config: ReplayConfig, params: dict) -> tuple[dict, dict]:
    runtime = load_runtime(config)
    root = config.package_dir / "ngc-recorded-context"

    def checked(name):
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != runtime.input_hashes[name]:
            raise ValueError(f"Frozen replay input changed: {name}")
        return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)

    case, bundle = checked("case.json"), checked("bundle.json")
    if actor_identity(checked("actor-params.json")) != actor_identity(params):
        raise ValueError("Gym row differs from the pinned actor history/tools")
    if config.coverage_profile != "recorded-v1":
        coverage = checked("coverage-v2.json.gz")
        if coverage["schema_version"] != runtime.source_schemas["coverage"]:
            raise ValueError("Unexpected coverage schema")
        for path, text in coverage["files"].items():
            relative = path.removeprefix(runtime.source_root.rstrip("/") + "/")
            if relative == path or ".." in Path(relative).parts:
                raise ValueError("Unsafe coverage source path")
            if hashlib.sha256(text.encode()).hexdigest() != coverage["source_file_sha256"][relative]:
                raise ValueError("Coverage source checksum mismatch")
        bundle["expanded_coverage"] = coverage
    if config.coverage_profile in {"expanded-v3", "expanded-v4"}:
        inventory = checked("source-inventory.json")
        if (
            inventory["schema_version"] != runtime.source_schemas["inventory"]
            or inventory["root"] != runtime.source_root
            or inventory["source_commit"] != coverage["provenance"]["source_commit"]
        ):
            raise ValueError("Source inventory identity mismatch")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in inventory["paths"]):
            raise ValueError("Unsafe source inventory path")
        bundle["source_inventory"] = inventory
    if config.coverage_profile == "expanded-v4":
        extra = checked("source-extra-v4.json.gz")
        if (
            extra["schema_version"] != runtime.source_schemas["extra"]
            or extra["source_commit"] != inventory["source_commit"]
        ):
            raise ValueError("Extra source identity mismatch")
        for path, text in extra["files"].items():
            relative = path.removeprefix(inventory["root"] + "/")
            if (
                relative not in inventory["paths"]
                or hashlib.sha256(text.encode()).hexdigest() != extra["file_sha256"][path]
            ):
                raise ValueError("Extra source checksum/path mismatch")
            if path in coverage["files"] or path in bundle["files"]:
                raise ValueError("Extra source cannot replace recorded content")
        coverage["files"].update(extra["files"])
    if config.temporal_history:
        bundle["temporal_world"] = checked("temporal-world.json")
    bundle.update(
        runtime=runtime.model_dump(),
        missing_runtime_reads=config.missing_runtime_reads,
        shell_control_flow=config.shell_control_flow,
        capture_stdout=config.capture_stdout,
    )
    return case, bundle


def probe_identity(event: dict, bundle: dict) -> dict:
    """Validate a fixture-backed exact probe, independently of prose/label claims.

    The interpreter is deliberately small. An unrecognized equivalent check is
    a coverage exclusion, never silently treated as no probe or a successful one.
    """
    if event["kind"] not in {"primary_probe", "sibling_probe"}:
        return {}
    fn = event["function"]
    command = json.loads(fn["arguments"])["command"]
    tokens = shell_tokens(canonical_path(command, bundle["runtime"]["path_aliases"]))
    if not tokens or tokens.pop(0) not in {"python", "python3", "/usr/bin/python3"}:
        raise ValueError("Unverified probe interpreter")
    if tokens and tokens[0] == "-I":
        tokens.pop(0)
    if not tokens or tokens.pop(0) != bundle["runtime"]["probe_script"]:
        raise ValueError("Probe script path is not the pinned script")
    options = {}
    while tokens:
        option = tokens.pop(0)
        if "=" in option:
            option, value = option.split("=", 1)
        elif tokens:
            value = tokens.pop(0)
        else:
            raise ValueError("Missing probe option value")
        if option not in {"--uri", "--output"} or option in options:
            raise ValueError("Unsupported/duplicate probe options or compound command")
        options[option] = value
    label = "sibling" if event["kind"] == "sibling_probe" else "primary"
    expected = bundle[label]
    if fn["name"] != "Bash" or options.get("--uri") != expected["input_uri"]:
        raise ValueError("Probe identity mismatch")
    if "--output" in options and not options["--output"].startswith(bundle["runtime"]["workspace_prefix"]):
        raise ValueError("Probe output outside modeled workspace")
    if json.loads(event["result"]) != expected:
        raise ValueError("Probe result differs from returned fixture")
    return {"namespace": label, "uri": expected["input_uri"], "returned_result": expected}


def visible_message(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_" + uuid4().hex,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def capture(
    params: dict,
    output: list[dict],
    events: list[dict],
    drafts: dict,
    final: str,
    *,
    prefix: bool = False,
    answer_paths: list[str] | None = None,
) -> LiveAgentCapture:
    # Receipts include response-turn numbers: all calls in one response were
    # chosen before any of their results could be seen by the model.
    text = next((drafts[path] for path in (answer_paths or list(drafts)) if path in drafts), final)
    receipts = {
        "supplied_history_is_context": True,
        "same_response_results_are_not_prior_evidence": True,
        "new_events": events,
    }
    if prefix:
        receipts["evaluation_boundary"] = "ordering_prefix_v1"
        receipts["scope"] = "new captured prefix; supplied history is context"
    return LiveAgentCapture(
        input=params["input"],
        artifacts={
            "final_reply": {
                "exists": bool(text) if prefix else True,
                **({"text": text} if text or not prefix else {}),
            },
            "replay_receipts": {"exists": True, "text": json.dumps(receipts, ensure_ascii=False, indent=2)},
        },
    )


async def execute(
    config: ReplayConfig,
    params: dict,
    *,
    model: str,
    secret: str,
    request_api,
    replay_id: str | None = None,
    review_prefix: Callable[[list[dict], LiveAgentCapture, int], Awaitable[dict]] | None = None,
    request_api_context: Callable[[dict, int, Path], Awaitable[tuple[int, dict]]] | None = None,
    observe_tool: Callable[[dict], None] | None = None,
) -> tuple[Path, list[dict], LiveAgentCapture, dict]:
    """Call request_api(payload, timeout_s); never retry an actor or tool request."""
    prefix_mode = config.evaluation_boundary == "ordering_prefix_v1"
    if prefix_mode and review_prefix is None:
        raise ValueError("Ordering evaluation requires a shared-verifier callback")
    case, bundle = load_inputs(config, params)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    replay_id = replay_id or uuid4().hex
    if len(replay_id) != 32 or any(c not in "0123456789abcdef" for c in replay_id):
        raise ValueError("Invalid replay ID")
    root = config.artifacts_dir / replay_id
    root.mkdir(mode=0o700)
    state = Fixtures(
        bundle, session_id=replay_id if config.coverage_profile in {"expanded-v3", "expanded-v4"} else None
    )
    messages = copy.deepcopy(case["messages"])
    state.sync_transcript(messages)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in case["tools"]}
    seen = {c["id"] for m in messages for c in m.get("tool_calls", [])}
    events, output, final, termination = [], [], "", "turn_limit"
    calls = 0
    total_calls = 0
    api_ids = set()

    async def invoke(payload, timeout, prefix):
        nonlocal total_calls
        if total_calls >= config.max_policy_calls:
            raise RuntimeError("Global policy call budget exhausted")
        total_calls += 1
        if config.coverage_profile == "expanded-v4":
            with (root / "api-order.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"index": total_calls, "request": str(prefix.relative_to(root)) + ".request.json"})
                    + "\n"
                )
        status, response = (
            await request_api_context(payload, timeout, prefix.relative_to(root))
            if request_api_context is not None
            else await request_api(payload, timeout)
        )
        if config.coverage_profile == "expanded-v4" and status == 200:
            identifier = response.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in api_ids:
                raise ValueError("Missing/duplicate global provider response ID")
            api_ids.add(identifier)
        return status, response

    prefix_grading_turn = None

    async def assess_prefix(turn: int) -> str | None:
        nonlocal prefix_grading_turn
        captured = capture(
            params, output, events, state.drafts, final, prefix=True, answer_paths=bundle["runtime"]["answer_paths"]
        )
        write_json(
            root / f"prefix-{turn:02d}.input.json", {"output": output, "capture": captured.model_dump()}, secret
        )
        try:
            graded = await review_prefix(output, captured, turn)
            write_json(root / f"prefix-{turn:02d}.grading-result.json", graded, secret)
            prefix_grading_turn = turn
            return ordering_outcome(
                graded,
                events,
                probe_namespace=bundle["runtime"]["ordering"]["probe_namespace"],
                expected_criterion_id=bundle["runtime"]["ordering"]["criterion_id"],
            )
        except Exception as error:
            write_json(
                root / f"prefix-{turn:02d}.error.json", {"type": type(error).__name__, "detail": str(error)}, secret
            )
            return "ordering_judge_error"

    response_ids = set()
    write_json(
        root / "identity.json",
        {
            "input_hashes": bundle["runtime"]["input_hashes"],
            "model": model,
            "settings": config.model_dump(mode="json", exclude={"agentic_if_root", "package_dir", "artifacts_dir"}),
            "automatic_retries": 0,
            "historical_actions_scored": False,
            "fixture_coverage": (
                "original-data-only-v1; probe-identity-validation-v1"
                if config.coverage_profile == "recorded-v1"
                else f"expanded-readonly-{config.coverage_profile.removeprefix('expanded-')}; probe-identity-validation-v1"
            ),
            "coverage_sha256": bundle["runtime"]["input_hashes"].get("coverage-v2.json.gz")
            if config.coverage_profile != "recorded-v1"
            else None,
            "temporal_world_sha256": bundle["runtime"]["input_hashes"].get("temporal-world.json")
            if config.temporal_history
            else None,
            "source_inventory_sha256": bundle["runtime"]["input_hashes"].get("source-inventory.json")
            if config.coverage_profile in {"expanded-v3", "expanded-v4"}
            else None,
            "extra_source_sha256": bundle["runtime"]["input_hashes"].get("source-extra-v4.json.gz")
            if config.coverage_profile == "expanded-v4"
            else None,
            "session_metadata": (
                {
                    "origin": "generated API replay session; not a native Claude session",
                    "session_id": state.session_id,
                    "transcript_path": state.transcript_path,
                    "transcript_format": "api_replay_message JSONL; exact supplied and new Chat messages",
                }
                if state.session_id is not None
                else None
            ),
            "code_sha256": {
                "capture_adapter": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prefix_boundary": hashlib.sha256(Path(__file__).with_name("prefix.py").read_bytes()).hexdigest(),
                "fixture_interpreter": hashlib.sha256(
                    Path(__file__).with_name("replay_fixtures.py").read_bytes()
                ).hexdigest(),
                "readonly_tools": hashlib.sha256(
                    Path(__file__).with_name("readonly_tools.py").read_bytes()
                ).hexdigest(),
                "workflow_tools": hashlib.sha256(
                    Path(__file__).with_name("workflow_tools.py").read_bytes()
                ).hexdigest(),
                "runtime_contract": hashlib.sha256(
                    (
                        config.runtime_contract_path or config.package_dir / "ngc-recorded-context/runtime.json"
                    ).read_bytes()
                ).hexdigest(),
                "delegation": hashlib.sha256(Path(__file__).with_name("delegation.py").read_bytes()).hexdigest(),
            },
        },
    )
    for turn in range(1, config.max_new_turns + 1):
        if total_calls >= config.max_policy_calls:
            termination = "policy_call_limit"
            break
        payload = case | {
            "messages": messages,
            "model": model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        write_json(root / f"call-{turn:02d}.request.json", payload, secret)
        calls += 1
        started = time.monotonic()
        try:
            status, response = await invoke(payload, config.timeout_s, root / f"call-{turn:02d}")
            write_json(root / f"call-{turn:02d}.response.json", response, secret)
            write_json(
                root / f"call-{turn:02d}.receipt.json",
                {
                    "http_status": status,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
        except Exception as error:
            write_json(
                root / f"call-{turn:02d}.error.json", {"type": type(error).__name__, "detail": str(error)}, secret
            )
            termination = "provider_error"
            break
        if status != 200 or not isinstance(response, dict) or response.get("error"):
            termination = "provider_error"
            break
        try:
            response_id = response.get("id")
            if not isinstance(response_id, str) or not response_id or response_id in response_ids:
                raise ValueError("Missing or duplicate provider response ID")
            if response.get("model") != model:
                raise ValueError("Provider model differs from requested actor")
            response_ids.add(response_id)
            if len(response["choices"]) != 1:
                raise ValueError("Ambiguous response choices")
            choice = response["choices"][0]
            msg = choice["message"]
            if msg["role"] != "assistant":
                raise ValueError("Unexpected response role")
            if choice["finish_reason"] == "length":
                termination = "output_limit"
                break
            if msg.get("refusal") or choice["finish_reason"] == "content_filter":
                termination = "refusal"
                break
            if choice["finish_reason"] not in {"stop", "tool_calls"}:
                raise ValueError("Unsupported finish reason")
            if msg.get("content") is not None and not isinstance(msg["content"], str):
                raise ValueError("Unsupported assistant content")
            assistant = {"role": "assistant", "content": msg.get("content")}
            if msg.get("reasoning_content") is not None:
                assistant["reasoning_content"] = msg["reasoning_content"]
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = [{k: c[k] for k in ("id", "type", "function")} for c in tool_calls]
                for call in assistant["tool_calls"]:
                    if not isinstance(call["id"], str) or not call["id"] or call["id"] in seen:
                        raise ValueError("Missing or duplicate tool call ID")
                    if call["type"] != "function":
                        raise ValueError("Unexpected tool type")
                    seen.add(call["id"])
            messages.append(assistant)
            state.sync_transcript(messages)
            if assistant["content"]:
                output.append(visible_message(assistant["content"]))
            if not tool_calls:
                final = assistant["content"] or ""
                if prefix_mode and final.strip():
                    termination = await assess_prefix(turn) or "ordering_undecided"
                else:
                    termination = "final_response" if final.strip() else "malformed_response"
                break
            batch = []
            for call_index, call in enumerate(assistant["tool_calls"], 1):
                observed_at, observed_monotonic = time.time(), time.perf_counter()
                result, kind = state.dispatch(call, schemas)
                if (
                    config.coverage_profile == "expanded-v4"
                    and call["function"]["name"] == "Agent"
                    and kind != "invalid_call"
                ):
                    result, kind = await explore(
                        json.loads(call["function"]["arguments"]),
                        state=state,
                        case=case,
                        config=config,
                        model=model,
                        directory=root / f"delegate-{turn:02d}-{call_index:02d}",
                        invoke=invoke,
                        write=lambda path, value: write_json(path, value, secret),
                        observe_tool=observe_tool,
                    )
                if observe_tool is not None:
                    observe_tool(tool_observation(call, result, kind, state, observed_at, observed_monotonic))
                event = {
                    "turn": turn,
                    "tool_call_id": call["id"],
                    "function": call["function"],
                    "kind": kind,
                    "result": result,
                }
                event["verified_probe"] = probe_identity(event, bundle)
                events.append(event)
                batch.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                output.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                        "status": "completed",
                    }
                )
            # Append all calls before all results; no sequential appearance can
            # manufacture evidence between calls in the same assistant response.
            for result in batch:
                messages.append(result)
                output.append(
                    {"type": "function_call_output", "call_id": result["tool_call_id"], "output": result["content"]}
                )
            state.sync_transcript(messages)
            kinds = {e["kind"] for e in events if e["turn"] == turn}
            if kinds & {"fixture_gap", "invalid_call"}:
                termination = "fixture_gap" if "fixture_gap" in kinds else "invalid_call"
                break
            if prefix_mode:
                decision = await assess_prefix(turn)
                if decision is not None:
                    termination = decision
                    break
                if state.drafts:
                    termination = "ordering_undecided"
                    break
            elif state.drafts:
                termination = "first_answer_draft"
                break
        except (KeyError, IndexError, TypeError, ValueError) as error:
            write_json(root / "capture-error.json", {"type": type(error).__name__, "detail": str(error)}, secret)
            termination = "capture_error"
            break
    diagnostics = {
        "termination": termination,
        "evaluation_completed": termination in {"ordering_failed", "ordering_passed"}
        if prefix_mode
        else termination in COMPLETE,
        "model_calls": total_calls,
        "new_sibling_probe_count": sum(e.get("verified_probe", {}).get("namespace") == "sibling" for e in events),
        "automatic_retries": 0,
        "semantic_score": "shared llm_judge_v1; mechanical probe count is not a score",
    }
    if prefix_mode:
        diagnostics.update(
            evaluation_boundary=config.evaluation_boundary,
            answer_draft_or_terminal_captured=bool(state.drafts or final.strip()),
            prefix_grading_turn=prefix_grading_turn,
        )
    if config.coverage_profile == "expanded-v4":
        diagnostics.update(parent_model_calls=calls, delegate_model_calls=total_calls - calls)
    for name, value in {
        "events.json": events,
        "drafts.json": state.drafts,
        "conversation.json": messages,
        "result.json": diagnostics,
        "terminal_reply.json": {"content": final},
        "visible-output.json": output,
    }.items():
        write_json(root / name, value, secret)
    captured = capture(
        params, output, events, state.drafts, final, prefix=prefix_mode, answer_paths=bundle["runtime"]["answer_paths"]
    )
    state.sync_transcript(messages)
    if state.transcript_path is not None:
        transcript = state.files[state.transcript_path]
        (root / "session-transcript.jsonl").write_text(
            transcript.replace(secret, "[REDACTED]") if secret else transcript
        )
        (root / "session-transcript.jsonl").chmod(0o600)
    write_json(root / "verification-input.json", captured.model_dump(), secret)
    return root, output, captured, diagnostics
