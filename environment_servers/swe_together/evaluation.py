# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen-goal correctness and independent interaction diagnostics."""

import asyncio
import json
import math
from pathlib import Path
from shlex import quote
from time import monotonic
from uuid import uuid4

from environment_servers.swe_together.coverage import (
    build_user_message,
    compute_scores,
    normalize_match_table,
    parse_json,
)
from environment_servers.swe_together.model_client import AuxiliaryModel
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
)
from nemo_gym.config import AgentServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess, read_text, upload_text
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


PROMPTS = Path(__file__).with_name("prompts")
PASS_THRESHOLD = 0.85
BASE_TAGS = {"request", "question", "verification", "workflow", "approval", "context"}
ALL_TAGS = BASE_TAGS | {"correction", "nudge"}
VERDICT_PATH = "/tmp/judge_inputs/verdict.json"
# Root Claude Code invocation error types that explain a missing verdict.
JUDGE_STOP_CAUSES = {"refusal": "judge_refused", "error_max_turns": "judge_turn_limit", "timeout": "judge_timeout"}


class JudgeMeasurementError(RuntimeError):
    """A grading failure with a stable cause; the episode is masked, never scored."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause


def judge_stop_cause(close: dict) -> str:
    """Name why a closed judge left no verdict, from its root invocation observation."""
    records = (close.get("agent_observations") or {}).get("records") or []
    root = next(
        (r for r in records if r.get("kind") == "agent_invocation" and r.get("parent_invocation_id") is None), {}
    )
    return JUDGE_STOP_CAUSES.get(root.get("error_type"), "judge_no_verdict")


async def missing_verdict_cause(sandbox: AsyncSandbox, close: dict, evidence: dict) -> str:
    """Call a verdict missing only when the judge sandbox confirms the file is absent; otherwise the read failed."""
    try:
        probe = await sandbox.exec("test -e " + quote(VERDICT_PATH), timeout_s=15)
    except Exception as error:
        evidence["judge_verdict_probe"] = {"error": f"{type(error).__name__}: {error}"}
        return "judge_verdict_transfer_failed"
    evidence["judge_verdict_probe"] = {"return_code": probe.return_code, "error_type": probe.error_type}
    if probe.return_code == 1 and not probe.error_type:
        return judge_stop_cause(close)
    return "judge_verdict_transfer_failed"


def derive_score(rubric: dict, verdict: dict) -> dict:
    """Recompute the exact reference weight sum, preserving the judge's override separately."""
    rows = verdict.get("goal_results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Missing judge goal_results")
    known = {g["id"] for g in rubric["completeness_goals"]}
    seen = set()
    for row in rows:
        if row.get("id") not in known or row["id"] in seen or not isinstance(row.get("met"), bool):
            raise ValueError("Invalid or duplicate judge goal result")
        seen.add(row["id"])
    if seen != known:
        raise ValueError("Judge omitted frozen goals")
    met = {r["id"]: r["met"] for r in rows}
    score = round(sum(float(g["weight"]) * float(met[g["id"]]) for g in rubric["completeness_goals"]), 2)
    return verdict | {
        "judge_reported_score": verdict.get("judge_score"),
        "judge_reported_verdict": verdict.get("verdict"),
        "judge_score": score,
        "verdict": "equivalent" if score >= PASS_THRESHOLD else "partial" if score >= 0.3 else "incorrect",
    }


async def stage_judge(sandbox: AsyncSandbox, task_dir: Path, patch: str, workdir: str) -> str:
    """Add trusted grading inputs after the complete task filesystem was replicated."""

    async def setup(command: str) -> None:
        result = await sandbox.exec(command, timeout_s=15)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"Judge input setup failed: {result.error_type}; "
                f"stdout={(result.stdout or '')[-4000:]}; stderr={(result.stderr or '')[-4000:]}"
            )

    await setup("mkdir -p /tmp/judge_inputs/tests /tmp/judge_inputs/logs")
    for name in ["README.md", "user_simulation_prompt.md", "canonical_goals.json"]:
        if (task_dir / name).is_file():
            await sandbox.upload(task_dir / name, "/tmp/judge_inputs/" + name)
        elif name == "README.md":
            await upload_text(sandbox, path="/tmp/judge_inputs/README.md", text="")
        else:
            raise FileNotFoundError(task_dir / name)
    await upload_text(sandbox, path="/tmp/judge_inputs/agent.patch", text=patch)
    await sandbox.upload(task_dir / "tests/test.sh", "/tmp/judge_inputs/test.sh")
    for file in sorted((task_dir / "tests").rglob("*")):
        if file.is_file():
            dest = "/tmp/judge_inputs/tests/" + str(file.relative_to(task_dir / "tests"))
            await setup("mkdir -p " + quote(str(Path(dest).parent)))
            await sandbox.upload(file, dest)
    await setup("chmod +x /tmp/judge_inputs/tests/test.sh")
    return (PROMPTS / "judge-user.md").read_text().format(workdir=workdir).strip()


async def read_test_reward(sandbox: AsyncSandbox) -> dict:
    """Canonical scripts may hardcode /logs/verifier despite staged LOGS_DIR."""
    result = {"test_reward_raw": None, "test_reward_source": "judge_optional_tests"}
    try:
        output = await sandbox.exec(
            "for path in /logs/verifier/reward.txt /tmp/judge_inputs/logs/reward.txt; do "
            'if [ -r "$path" ]; then printf "%s\\n" "$path"; cat "$path"; exit; fi; done',
            timeout_s=15,
        )
        if output.return_code or output.error_type:
            raise RuntimeError("Optional test reward read failed")
        lines = (output.stdout or "").splitlines()
        if lines:
            result["test_reward_path"] = lines[0]
            reward = float("\n".join(lines[1:]).strip())
            if not math.isfinite(reward):
                raise ValueError("Optional test reward is not finite")
            result["test_reward_raw"] = reward
    except Exception as error:
        result["test_reward_error"] = str(error)
    return result


async def run_judge(
    *,
    client: ServerClient,
    agent: AgentServerRef,
    sandbox: AsyncSandbox,
    provider: str,
    task_dir: Path,
    patch: str,
    workdir: str,
    episode: EpisodeId,
    task_id: TaskId,
    timeout: float,
    cookies: dict[str, str],
    artifact_dir: Path,
    deadline: float | None = None,
    cleanup_timeout: float = 300,
    evidence: dict | None = None,
) -> dict:
    evidence = evidence if evidence is not None else {}
    prompt = await stage_judge(sandbox, task_dir, patch, workdir)
    session_id = "judge-" + uuid4().hex
    judge_cookies = dict(cookies)
    # The judge is a separate logical activation and has a separate capture identity.
    judge_episode = EpisodeId(rollout_id=episode.capture_key + "-judge")
    seed = AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id=judge_episode,
        task_id=task_id,
        sandbox_access=SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref=provider, descriptor=await sandbox.serialize()),
            # A trusted grading directory avoids both adapter runtime roots and
            # candidate-authored project instructions at harness startup.
            workdir="/tmp/judge_inputs",
        ),
    )
    original_error = None
    transcript = {}
    try:
        response = await client.post(agent.name, "/v1/agent_sessions", json=seed, cookies=judge_cookies)
        await raise_for_status(response)
        await get_response_json(response)
        judge_cookies.update({key: value.value for key, value in response.cookies.items()})
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": (PROMPTS / "judge-phase2-system.md").read_text()},
                {"role": "user", "content": prompt},
            ],
            metadata={
                "agent_timeout_sec": str(min(timeout, max(0.001, deadline - monotonic())) if deadline else timeout)
            },
        )
        response = await client.post(
            agent.name, f"/ng-rollout/{judge_episode.capture_key}/v1/responses", json=params, cookies=judge_cookies
        )
        await raise_for_status(response)
        transcript = await get_response_json(response)
        (artifact_dir / "judge-transcript.json").write_text(json.dumps(transcript, indent=2))
        terminal_text = ""
        for item in reversed(transcript.get("output", [])):
            if item.get("type") == "message":
                terminal_text = "\n".join(
                    part.get("text", "") for part in item.get("content", []) if part.get("type") == "output_text"
                )[-4000:]
                break
        evidence["judge_outcome"] = {
            "status": transcript.get("status"),
            "error": transcript.get("error"),
            "metadata": transcript.get("metadata"),
            "terminal_text": terminal_text,
        }
    except Exception as error:
        original_error = error
        evidence["judge_transport_error"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            async with asyncio.timeout(cleanup_timeout):
                response = await client.post(
                    agent.name,
                    "/v1/agent_sessions/close",
                    json=AgentCloseSessionRequest(agent_session_id=session_id, episode_id=judge_episode),
                    cookies=judge_cookies,
                    _control=True,
                )
                await raise_for_status(response)
                close = AgentCloseSessionResponse.model_validate(await get_response_json(response))
                receipt = close.model_dump(mode="json")
                evidence["judge_close"] = receipt
                (artifact_dir / "judge-close.json").write_text(json.dumps(receipt, indent=2))
                if close.agent_session_id != session_id or close.cleanup_confirmed is not True:
                    raise RuntimeError("Judge session close was not confirmed")
        except Exception as error:
            evidence["judge_close_error"] = str(error)
            (artifact_dir / "judge-close-error.json").write_text(json.dumps({"error": str(error)}))
            # A seed or execution failure is the primary cause; the close failure stays in the evidence.
            if original_error is None:
                raise RuntimeError("Judge cleanup was not confirmed") from error
            if isinstance(original_error, Exception):
                raise original_error
            # Otherwise a cancellation is already propagating; let it continue.
    try:
        raw = await read_text(sandbox, path=VERDICT_PATH)
    except Exception as error:
        # A saved verdict still wins after a lost response; otherwise preserve the primary exception.
        if original_error is not None:
            raise original_error
        raise JudgeMeasurementError(
            await missing_verdict_cause(sandbox, receipt, evidence),
            "Judge produced no readable verdict; outcome="
            + json.dumps(evidence.get("judge_outcome"))
            + "; transport="
            + str(original_error)
            + f"; read={type(error).__name__}: {error}",
        ) from error
    (artifact_dir / "judge-verdict-raw.json").write_text(raw)
    rubric = json.loads((task_dir / "canonical_goals.json").read_text())
    try:
        verdict = derive_score(rubric, json.loads(raw))
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise JudgeMeasurementError("judge_verdict_invalid", f"Judge verdict is invalid: {error}") from error
    verdict.update(await read_test_reward(sandbox))
    verdict["judge_close"] = receipt
    verdict["judge_runtime"] = (transcript.get("metadata") or {}).get("runtime_version")
    return verdict


async def interaction_metrics(
    model: AuxiliaryModel, *, task_dir: Path, messages: list[dict], artifact_dir: Path
) -> dict:
    result = {
        "n_simulator_messages": len(messages),
        "user_correction": None,
        "intent_coverage": None,
    }
    if messages:
        tagger_raw = None
        try:
            user = "Messages:\n" + "\n".join(
                f"trial_idx={m['trial_idx']}: {' '.join(m['text'].split())[:1000]}" for m in messages
            )
            response = await model.call((PROMPTS / "tag-messages-system.md").read_text() + "\n\n" + user)
            tagger_raw = response.content
            raw = parse_json(tagger_raw)
            entries = raw.get("results", [])
            if not isinstance(entries, list):
                raise ValueError("Message tag results must be a list")
            rows = {
                row["trial_idx"]: row for row in entries if isinstance(row, dict) and type(row.get("trial_idx")) is int
            }
            if len(entries) != len(rows) or set(rows) != {m["trial_idx"] for m in messages}:
                raise ValueError("Missing, duplicate or extra message tags")
            tags = []
            for message in messages:
                row = rows[message["trial_idx"]]
                values = row.get("tags")
                if (
                    not isinstance(values, list)
                    or not values
                    or any(not isinstance(value, str) or value not in ALL_TAGS for value in values)
                ):
                    raise ValueError(f"Invalid or missing tags for trial_idx={message['trial_idx']}")
                acts = set(values)
                if not acts & BASE_TAGS:
                    acts.add("request")
                tags.append(
                    {
                        "trial_idx": message["trial_idx"],
                        "tags": sorted(acts),
                        "frustration": int(bool(row.get("frustration"))),
                    }
                )
            result["message_tags"] = tags
            result["user_correction"] = round(
                sum(("correction" in r["tags"]) + 0.2 * ("nudge" in r["tags"]) for r in tags), 4
            )
        except Exception as error:
            result["tagger_error"] = str(error)
            if tagger_raw is not None:
                result["tagger_response"] = tagger_raw
    coverage_raw = None
    try:
        intents = json.loads((task_dir / "oracle_intents.json").read_text())["intents"]
        if not intents:
            result["intent_coverage"] = compute_scores({"per_intent": []}, 0, len(messages))
        else:
            trial = [
                {
                    "trial_idx": 0,
                    "turn": 0,
                    "action": "instruction",
                    "text": (task_dir / "instruction.md").read_text().strip(),
                }
            ] + messages
            response = await model.call(
                build_user_message(intents, trial),
                message_history=[{"role": "system", "content": (PROMPTS / "coverage-system.md").read_text()}],
            )
            coverage_raw = response.content
            table, warnings = normalize_match_table(parse_json(coverage_raw), len(intents), len(trial))
            if warnings:
                result["intent_coverage_diagnostics"] = {"match_table": table, "schema_warnings": warnings}
                raise ValueError("Intent coverage response required schema repair: " + "; ".join(warnings))
            result["intent_coverage"] = compute_scores(table, len(intents), len(trial)) | {
                "match_table": table,
                "schema_warnings": warnings,
            }
    except Exception as error:
        result["coverage_error"] = str(error)
        if coverage_raw is not None:
            result.setdefault("intent_coverage_diagnostics", {})["raw_response"] = coverage_raw
    (artifact_dir / "interaction.json").write_text(json.dumps(result, indent=2))
    return result
