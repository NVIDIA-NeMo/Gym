# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project host-observed SkillsBench execution evidence without changing grades."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from harbor.models.task import config as harbor_task_config

from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ContextCompactionObservation, ModelCallRef


if TYPE_CHECKING:
    from harbor.models.trial.result import TrialResult

    from nemo_gym.config import ModelServerRef
    from responses_api_agents.harbor_agent_general.app import HarborVerifyResponse


logger = logging.getLogger(__name__)
VERIFIER_EXECUTION_FILE = "gym-verifier-execution.json"


def verifier_command() -> str | None:
    """The single-step Linux verifier command used by prepared official tasks."""
    # The legacy adapter pins Harbor before TaskOS and the script builder.
    # Keep its environment usable without claiming modern verifier evidence.
    task_os = getattr(harbor_task_config, "TaskOS", None)
    if task_os is None:
        return None
    from harbor.utils.scripts import build_execution_command

    return build_execution_command(
        "/tests/test.sh", stdout_path="/logs/verifier/test-stdout.txt", task_os=task_os.LINUX
    )


def verifier_exit_code(trial_dir: Path, trial: TrialResult) -> int | None:
    """Accept only a host witness inside this trial's completed verification phase.

    A reward file or a finished phase alone does not establish that a test ran.
    The host witness is not in the sandbox's downloadable verifier directory.
    """
    command = verifier_command()
    if command is None:
        return None
    if trial.step_results or trial.verifier_result is None or trial.exception_info is not None:
        return None
    phase = trial.verifier
    if phase is None or phase.started_at is None or phase.finished_at is None:
        return None
    try:
        witness = json.loads((trial_dir / VERIFIER_EXECUTION_FILE).read_text())
        code = witness["return_code"]
        start, end = witness["started_at"], witness["finished_at"]
        if (
            witness["command"] == command
            and type(code) is int
            and code != -1
            and type(start) in (int, float)
            and type(end) in (int, float)
            and phase.started_at.timestamp() <= start <= end <= phase.finished_at.timestamp()
        ):
            return code
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def agent_observations(
    trial_dir: Path, trial: TrialResult, response: HarborVerifyResponse, model_ref: ModelServerRef
) -> AgentObservationBundle | None:
    """Join responses explicitly identified by OpenHands' saved events.

    Do not infer ownership from the rollout URL: auxiliary/condenser calls can
    share it. Multiple tool actions can refer to the same model response.
    """
    path = trial_dir / "agent" / "openhands-events.jsonl"
    compactions = []
    try:
        response_ids = set()
        with path.open() as events:
            invocation_id = f"openhands-{trial.id}"
            for line in events:
                event = json.loads(line)
                if event.get("kind") not in {"ActionEvent", "MessageEvent", "Condensation"}:
                    continue
                identity = event.get("llm_response_id")
                if isinstance(identity, str) and identity:
                    response_ids.add(identity)
                    if event["kind"] == "Condensation":
                        # SDK accounting includes these calls. Retain ownership
                        # while explicitly excluding them from policy turns.
                        compactions.append(
                            ContextCompactionObservation(
                                invocation_id=invocation_id,
                                outcome="completed",
                                model_calls=[ModelCallRef(model_ref=model_ref, response_id=identity)],
                            )
                        )
    except (OSError, ValueError, AttributeError):
        logger.warning("SkillsBench agent observations unavailable for %s", trial_dir.name)
        return None
    timed_out = response.failure_kind == "agent_timeout"
    invocation = AgentInvocation(
        invocation_id=invocation_id,
        status="incomplete" if timed_out else "completed",
        stop_reason="agent_timeout" if timed_out else None,
        model_calls=[ModelCallRef(model_ref=model_ref, response_id=identity) for identity in sorted(response_ids)],
        conversation=[
            item.model_dump(mode="json")
            for item in [*response.responses_create_params.input, *response.response.output]
        ],
    )
    return AgentObservationBundle(source="skillsbench-openhands-sdk", records=[invocation, *compactions])
