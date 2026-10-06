# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen-goal correctness and independent interaction diagnostics."""

import json
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.config_types import AgentServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from resources_servers.swe_together.coverage import (
    build_user_message,
    compute_scores,
    normalize_match_table,
    parse_json,
)
from resources_servers.swe_together.model_client import AuxiliaryModel
from resources_servers.swe_together.patch_normalize import apply_candidates, main_repo_path


PROMPTS = Path(__file__).with_name("prompts")
BASE_TAGS = {"request", "question", "verification", "workflow", "approval", "context"}
ALL_TAGS = BASE_TAGS | {"correction", "nudge"}


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
        "verdict": "equivalent" if score >= 0.85 else "partial" if score >= 0.3 else "incorrect",
    }


async def stage_judge(sandbox: AsyncSandbox, task_dir: Path, patch: str, workdir: str) -> str:
    """Apply only the upstream main-repository projection; retain full artifacts on the host."""
    hint = main_repo_path(patch) or workdir
    if not hint.startswith("/") or ".." in Path(hint).parts:
        raise ValueError("Invalid patch repository")
    candidates = apply_candidates(patch)
    await sandbox.exec("mkdir -p /tmp/judge_inputs/tests /tmp/judge_inputs/logs", timeout_s=15)
    if candidates:
        for i, candidate in enumerate(candidates):
            await upload_text(sandbox, path=f"/tmp/agent.patch.{i}", text=candidate)
        applied = False
        for flags in [[], ["-C2"]]:
            for i in range(len(candidates)):
                command = (
                    'git -c safe.directory="*" -C ' + quote(hint) + " apply --whitespace=nowarn " + " ".join(flags)
                )
                result = await sandbox.exec(command + " --check " + quote(f"/tmp/agent.patch.{i}"), timeout_s=120)
                if result.return_code:
                    continue
                result = await sandbox.exec(command + " " + quote(f"/tmp/agent.patch.{i}"), timeout_s=120)
                if result.return_code:
                    raise RuntimeError("Patch application failed after successful check")
                applied = True
                break
            if applied:
                break
        if not applied:
            raise RuntimeError("Candidate patch cannot be applied to fresh task image")
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
            await sandbox.exec("mkdir -p " + quote(str(Path(dest).parent)), timeout_s=15)
            await sandbox.upload(file, dest)
    await sandbox.exec("chmod +x /tmp/judge_inputs/tests/test.sh", timeout_s=15)
    return (
        "Begin by reading /tmp/judge_inputs/canonical_goals.json — this is "
        "the FROZEN rubric. DO NOT re-derive goals; for each goal in "
        "the rubric, mark met:true/false with concrete evidence. Then "
        "read /tmp/judge_inputs/README.md and /tmp/judge_inputs/user_simulation_prompt.md "
        "for context, inspect the agent's patch at /tmp/judge_inputs/agent.patch "
        f"(already applied to {hint}), explore the workspace, and "
        "optionally run tests. Write your verdict to /tmp/judge_inputs/verdict.json."
    )


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
) -> dict:
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
            workdir="/tmp",
        ),
    )
    try:
        response = await client.post(agent.name, "/v1/agent_sessions", json=seed, cookies=judge_cookies)
        await raise_for_status(response)
        await get_response_json(response)
        judge_cookies.update({key: value.value for key, value in response.cookies.items()})
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": prompt}],
            instructions=(PROMPTS / "judge-phase2-system.md").read_text(),
            metadata={"timeout_seconds": str(timeout)},
        )
        response = await client.post(agent.name, "/v1/responses", json=params, cookies=judge_cookies)
        await raise_for_status(response)
        transcript = await get_response_json(response)
        (artifact_dir / "judge-transcript.json").write_text(json.dumps(transcript, indent=2))
    finally:
        response = await client.post(
            agent.name,
            "/v1/agent_sessions/close",
            json=AgentCloseSessionRequest(agent_session_id=session_id, episode_id=judge_episode),
            cookies=judge_cookies,
        )
        await raise_for_status(response)
        receipt = await get_response_json(response)
        if receipt.get("cleanup_confirmed") is not True:
            raise RuntimeError("Judge cleanup was not confirmed")
        (artifact_dir / "judge-close.json").write_text(json.dumps(receipt, indent=2))
    raw = await read_text(sandbox, path="/tmp/judge_inputs/verdict.json")
    (artifact_dir / "judge-verdict-raw.json").write_text(raw)
    verdict = derive_score(json.loads((task_dir / "canonical_goals.json").read_text()), json.loads(raw))
    try:
        verdict["test_reward_raw"] = float(await read_text(sandbox, path="/tmp/judge_inputs/logs/reward.txt"))
    except Exception:
        verdict["test_reward_raw"] = None
    verdict["judge_runtime"] = (transcript.get("metadata") or {}).get("runtime_version")
    return verdict


async def interaction_metrics(
    model: AuxiliaryModel, *, task_dir: Path, messages: list[dict], artifact_dir: Path
) -> dict:
    result = {
        "n_simulator_messages": len(messages),
        "user_correction": 0.0 if not messages else None,
        "intent_coverage": None,
    }
    if messages:
        try:
            user = "Messages:\n" + "\n".join(
                f"trial_idx={m['trial_idx']}: {' '.join(m['text'].split())[:1000]}" for m in messages
            )
            response = await model.call((PROMPTS / "tag-messages-system.md").read_text() + "\n\n" + user)
            raw = parse_json(response.content)
            rows = {
                row["trial_idx"]: row
                for row in raw.get("results", [])
                if isinstance(row, dict) and isinstance(row.get("trial_idx"), int)
            }
            if set(rows) != {m["trial_idx"] for m in messages}:
                raise ValueError("Missing/extra message tags")
            tags = []
            for message in messages:
                row = rows[message["trial_idx"]]
                acts = set(row.get("tags") or []) & ALL_TAGS
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
    try:
        intents = json.loads((task_dir / "oracle_intents.json").read_text())["intents"]
        if not intents:
            result["intent_coverage"] = {
                "coverage_rate": 1.0,
                "weighted_coverage": 1.0,
                "scope_precision": 1.0,
                "overall_score": 1.0,
            }
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
            table, warnings = normalize_match_table(parse_json(response.content), len(intents), len(trial))
            result["intent_coverage"] = compute_scores(table, len(intents), len(trial)) | {
                "match_table": table,
                "schema_warnings": warnings,
            }
    except Exception as error:
        result["coverage_error"] = str(error)
    (artifact_dir / "interaction.json").write_text(json.dumps(result, indent=2))
    return result
