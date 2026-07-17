# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hypotest resource server with sandbox workspace and artifact handoff."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, cast

from aviary.core import Environment
from fastapi import Request
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.aviary.app import obs_msg_to_nemo_gym, tool_to_function_tool_param
from resources_servers.aviary.hypotest_app import HypotestResourcesServer, HypotestServerConfig
from resources_servers.aviary.schemas import AviarySeedSessionRequest, AviarySeedSessionResponse
from resources_servers.bbh_hypotest.evidence import (
    EVIDENCE_CANDIDATES,
    EvidenceError,
    build_hypotest_prompt,
    load_evidence_cells,
    safe_artifact_path,
)


LOG = logging.getLogger(__name__)
SANDBOX_ARTIFACTS_METADATA_KEY = "nemo_gym_sandbox_artifacts"


class BBHHypotestConfig(HypotestServerConfig):
    seed_concurrency: int = 8
    sandbox_workdir: str = "/data_workspace"
    artifact_paths: list[str] = Field(default_factory=lambda: list(EVIDENCE_CANDIDATES))
    max_artifact_bytes: int = 10 * 1024 * 1024


class BBHSeedSessionResponse(AviarySeedSessionResponse):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    sandbox_setup: dict[str, Any]
    verify_context: dict[str, Any]


class BBHVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    seed_session: dict[str, Any] = Field(default_factory=dict)


class BBHVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    mask_sample: bool = False
    evidence_path: str | None = None
    evidence_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def final_assistant_text(response) -> str:
    for item in reversed(response.output):
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        parts = [str(part.text) for part in (getattr(item, "content", None) or []) if getattr(part, "text", None)]
        if parts:
            return "\n".join(parts)
    return "[harness produced no final conclusion]"


def result_metadata(env: Environment) -> dict[str, Any]:
    """Extract stable scoring fields without depending on a Hypotest-private helper."""
    state = env.state
    return {
        "done": bool(getattr(state, "done", False)),
        "score": float(getattr(state, "score", 0.0)),
        "raw_score": getattr(state, "raw_score", None),
        "total_reward": float(getattr(state, "total_reward", 0.0)),
        "notebook_runtime_errors": list(getattr(state, "notebook_runtime_errors", [])),
    }


class BBHHypotestResourcesServer(HypotestResourcesServer):
    config: BBHHypotestConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, context: Any) -> None:
        if self.config.seed_concurrency < 1:
            raise ValueError("seed_concurrency must be positive")
        self._seed_sem = asyncio.Semaphore(self.config.seed_concurrency)
        super().model_post_init(context)

    async def seed_session(self, request: Request, body: AviarySeedSessionRequest) -> BBHSeedSessionResponse:
        del request
        env_id = str(uuid.uuid4())
        env = cast(Environment, self.dataset.get_new_env_by_idx(body.task_idx))
        self.env_id_to_env[env_id] = env
        try:
            async with self._seed_sem:
                observations, tools = await env.reset()
        except BaseException:
            self.env_id_to_total_reward.pop(env_id, None)
            self.env_id_to_env.pop(env_id, None)
            try:
                await asyncio.shield(env.close())
            except Exception:
                LOG.exception("Failed to close partially seeded Hypotest environment")
            raise

        obs = [message for item in observations for message in obs_msg_to_nemo_gym(item)]
        prompt = build_hypotest_prompt(obs)
        return BBHSeedSessionResponse(
            env_id=env_id,
            obs=obs,
            tools=[tool_to_function_tool_param(tool) for tool in tools],
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": prompt}],
                metadata={"workdir": self.config.sandbox_workdir},
            ),
            sandbox_setup={
                "workspace_path": str(env.work_dir),
                "workdir": self.config.sandbox_workdir,
                "artifact_paths": self.config.artifact_paths,
                "max_artifact_bytes": self.config.max_artifact_bytes,
            },
            verify_context={"env_id": env_id},
        )

    @staticmethod
    def _restore_artifacts(workspace: Path, response) -> str | None:
        metadata = response.metadata or {}
        encoded = metadata.get(SANDBOX_ARTIFACTS_METADATA_KEY, "{}")
        artifacts = json.loads(encoded) if isinstance(encoded, str) else dict(encoded)
        selected = None
        for relative_path in EVIDENCE_CANDIDATES:
            content = artifacts.get(relative_path)
            if not isinstance(content, str):
                continue
            destination = safe_artifact_path(workspace, relative_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
            if selected is None:
                selected = relative_path
        return selected

    async def verify(self, request: Request, body: BBHVerifyRequest) -> BBHVerifyResponse:
        del request
        env_id = str(body.seed_session.get("env_id") or "")
        env = self.env_id_to_env.get(env_id)
        if env is None:
            return BBHVerifyResponse(
                **body.model_dump(exclude={"seed_session"}),
                reward=0.0,
                mask_sample=True,
                evidence_error=f"Unknown or closed Hypotest environment: {env_id}",
            )

        evidence_path = None
        evidence_error = None
        mask_sample = False
        metadata = {}
        try:
            evidence_path = self._restore_artifacts(Path(env.work_dir), body.response)
            try:
                artifact, cells = load_evidence_cells(Path(env.work_dir), evidence_path)
                evidence_path = str(artifact.relative_to(env.work_dir))
            except EvidenceError as exc:
                evidence_error = str(exc)
                cells = [f"raise RuntimeError({json.dumps(evidence_error)})"]

            for source in cells:
                await env.run_cell(source)
            await env.submit_answer(final_assistant_text(body.response))
            metadata = result_metadata(env)
            metadata.update(
                {
                    "evidence_path": evidence_path,
                    "evidence_error": evidence_error,
                    "evidence_cell_count": len(cells),
                    "evidence_adapter": "verbatim-artifact-v1",
                }
            )
        except Exception as exc:
            LOG.exception("Hypotest artifact execution or scoring failed")
            mask_sample = True
            evidence_error = f"{type(exc).__name__}: {exc}"
            metadata = {"infrastructure_error": evidence_error}
            try:
                metadata.update(result_metadata(env))
            except Exception:
                pass
        finally:
            self.env_id_to_total_reward.pop(env_id, None)
            self.env_id_to_env.pop(env_id, None)
            try:
                await env.close()
            except Exception as exc:
                LOG.exception("Failed to close Hypotest artifact environment")
                mask_sample = True
                metadata["close_error"] = f"{type(exc).__name__}: {exc}"

        LOG.warning(
            "BBH verify env_id=%s reward=%.4f evidence_path=%s evidence_error=%r "
            "mask_sample=%s done=%s score=%s raw_score=%s notebook_runtime_errors=%s",
            env_id,
            float(env.state.total_reward),
            evidence_path,
            evidence_error,
            mask_sample,
            metadata.get("done"),
            metadata.get("score"),
            metadata.get("raw_score"),
            metadata.get("notebook_runtime_errors", []),
        )
        return BBHVerifyResponse(
            **body.model_dump(exclude={"seed_session"}),
            reward=float(env.state.total_reward),
            mask_sample=mask_sample,
            evidence_path=evidence_path,
            evidence_error=evidence_error,
            metadata=metadata,
        )


if __name__ == "__main__":
    BBHHypotestResourcesServer.run_webserver()
