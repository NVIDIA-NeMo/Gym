# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-turn reference image → Python → Enroot render → rubric reward."""

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from resources_servers.vision2code.evaluation import (
    aggregate_rating,
    build_rater_user_content,
    build_render_failure_rating,
    system_prompt,
)
from resources_servers.vision2code.prepare import Reference, read_reference
from resources_servers.vision2code.render import RenderResult, render
from resources_servers.vision2code.rubrics import inspect_candidate_image, parse_json_object
from resources_servers.vision2code.tests.verifier_fixture import VERIFIER_FIXTURE as VERIFIER_FIXTURE


class Vision2CodeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    sandbox_image: str
    sandbox_provider: dict[str, dict] = Field(default_factory=lambda: {"enroot": {}})
    execution_timeout: float = Field(default=60, gt=0)
    render_concurrency: int = Field(default=4, ge=1)
    artifact_dir: str = "artifacts/vision2code"

    @model_validator(mode="after")
    def validate_renderer(self) -> "Vision2CodeConfig":
        if set(self.sandbox_provider) != {"enroot"}:
            raise ValueError("vision2code requires Gym's Enroot provider")
        if not self.sandbox_image.endswith(".sqsh") or not Path(self.sandbox_image).is_absolute():
            raise ValueError("sandbox_image must be an absolute local .sqsh path")
        config = self.sandbox_provider["enroot"]
        # This environment only needs provider storage paths and startup timeouts.
        # Do not let provider overrides add host mounts, hooks, or commands.
        if set(config) - {"create"}:
            raise ValueError("Only Enroot create storage paths and timeouts may be configured")
        allowed = {"base_dir", "data_path", "cache_path", "runtime_path", "create_timeout_s", "start_timeout_s"}
        if set(config.get("create", {})) - allowed:
            raise ValueError(f"Allowed Enroot create settings: {sorted(allowed)}")
        return self


class Vision2CodeVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    verifier_metadata: Reference


class Vision2CodeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    verifier_metadata: Reference
    render: RenderResult | None = None
    rubric_result: dict[str, JsonValue] | None = None
    judge_diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


def response_text(response: NeMoGymResponse) -> str:
    """Extract only the final assistant message, excluding reasoning items."""
    for item in reversed(response.output):
        if item.type == "message" and item.role == "assistant":
            return "\n".join(c.text for c in item.content if c.type == "output_text")
    return ""


def mask_failure(result: Vision2CodeVerifyResponse, *, kind: str, reason: str) -> None:
    """Use both current masking fields and Gym's persisted failure-sidecar routing."""
    result.reward = 0.0
    result.mask_sample = True
    result.failure_kind = kind
    result.failure_reason = reason
    instance = result.model_extra.get("instance_config")
    result.model_extra.update(
        {
            "_ng_failure_class": kind,
            "instance_config": {**(instance if isinstance(instance, dict) else {}), "mask_sample": True},
        }
    )


class Vision2CodeServer(SimpleResourcesServer):
    ray_enabled = False
    config: Vision2CodeConfig
    _render_slots: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        self._render_slots = asyncio.Semaphore(self.config.render_concurrency)
        super().model_post_init(context)

    async def verify(self, body: Vision2CodeVerifyRequest) -> Vision2CodeVerifyResponse:
        ref = body.verifier_metadata
        directory = Path(self.config.artifact_dir).resolve() / uuid.uuid4().hex
        directory.mkdir(parents=True)
        # Preserve Gym task/rollout bookkeeping, but never reuse verifier outputs
        # when an existing rollout is submitted for reverification.
        derived = set(Vision2CodeVerifyResponse.model_fields) - set(Vision2CodeVerifyRequest.model_fields)
        result = Vision2CodeVerifyResponse(
            **body.model_dump(exclude=derived | {"_ng_failure_class", "_ng_failure_judge_error"}), reward=0.0
        )
        instance = result.model_extra.get("instance_config")
        if isinstance(instance, dict):
            result.model_extra["instance_config"] = {**instance, "mask_sample": False}
        result.artifacts = {"directory": str(directory), "result": str(directory / "result.json")}
        stage = "reference"
        start = time.perf_counter()
        try:
            reference_path = Path(ref.image_path)
            if not reference_path.is_absolute():
                # Gym starts servers from their component directory; committed examples
                # use repository-relative paths so they remain portable across checkouts.
                reference_path = Path(__file__).resolve().parents[2] / reference_path
            raw, _, width, height = read_reference(reference_path)
            if hashlib.sha256(raw).hexdigest() != ref.sha256 or (width, height) != (ref.width, ref.height):
                raise ValueError("Reference changed since task preparation; regenerate the JSONL")
            stage = "render"
            provider = {
                "enroot": {
                    **self.config.sandbox_provider["enroot"],
                    "create": {
                        **self.config.sandbox_provider["enroot"].get("create", {}),
                        "mount_point": "/workspace",
                    },
                    "probe": {"command": "test -d /workspace", "expected_stdout": None},
                }
            }
            async with self._render_slots:
                result.render = await render(
                    code=response_text(body.response),
                    width=ref.width,
                    height=ref.height,
                    image=self.config.sandbox_image,
                    provider=provider,
                    artifact_dir=directory,
                    timeout=self.config.execution_timeout,
                )
            result.artifacts.update(
                {"code": str(directory / "code.py"), "execution": str(directory / "execution.json")}
            )
            if result.render.status == "infrastructure_failure":
                mask_failure(result, kind="provider_unavailable", reason=json.dumps(result.render.diagnostics))
                return result
            if result.render.status != "ok":
                result.rubric_result = build_render_failure_rating(result.render.status, dataset_name=ref.rubric)
                return result
            candidate = directory / "output.png"
            result.artifacts["image"] = str(candidate)
            inspection = inspect_candidate_image(candidate)
            # Reuse the exact upstream prompt, translating only Chat content types
            # to Gym Responses types. The model adapter owns endpoint translation.
            content = build_rater_user_content(
                source_image_path=reference_path,
                rendered_image_path=candidate,
                dataset_name=ref.rubric,
            )
            response_content = [
                {"type": "input_text", "text": part["text"]}
                if part["type"] == "text"
                else {"type": "input_image", "image_url": part["image_url"]["url"], "detail": "auto"}
                for part in content
            ]
            params = self.config.judge_responses_create_params.model_dump(exclude_none=True)
            params.update(
                input=[
                    {"role": "system", "content": system_prompt(dataset_name=ref.rubric)},
                    {"role": "user", "content": response_content},
                ],
                tools=[],
            )
            stage = "judge"
            judge_start = time.perf_counter()
            try:
                judge = await call_judge(
                    self.server_client,
                    server_name=self.config.judge_model_server.name,
                    url_path="/v1/responses",
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
                    response_model=NeMoGymResponse,
                )
            finally:
                result.timings["judge_s"] = time.perf_counter() - judge_start
            result.judge_diagnostics = {"response": judge.model_dump(mode="json"), "candidate_inspection": inspection}
            stage = "judge_parse"
            parsed = parse_json_object(response_text(judge))
            result.rubric_result = aggregate_rating(
                parsed,
                dataset_name=ref.rubric,
                candidate_inspection=inspection,
                reference_image_path=str(reference_path),
                candidate_image_path=str(candidate),
            )
            result.reward = result.rubric_result["final_rating_0_to_5"] / 5.0
            return result
        except asyncio.CancelledError:
            mask_failure(result, kind="cancelled", reason=f"Cancelled during {stage}")
            raise
        except JudgeError as exc:
            mask_failure(result, kind="judge_failed", reason=str(exc))
            result.judge_diagnostics["error"] = str(exc)
            return result
        except Exception as exc:
            kind = "judge_unparseable" if stage == "judge_parse" else "verifier_error"
            mask_failure(result, kind=kind, reason=f"{stage}: {type(exc).__name__}: {exc}")
            return result
        finally:
            result.timings["total_s"] = time.perf_counter() - start
            execution_path = directory / "execution.json"
            if result.render is None and execution_path.exists():
                result.render = RenderResult.model_validate_json(execution_path.read_text())
                result.artifacts.update({"code": str(directory / "code.py"), "execution": str(execution_path)})
            (directory / "judge.json").write_text(json.dumps(result.judge_diagnostics, indent=2))
            result.artifacts["judge"] = str(directory / "judge.json")
            # Avoid duplicating base64 reference images and full policy inputs.
            saved = result.model_dump(mode="json", exclude={"response", "responses_create_params"})
            (directory / "result.json").write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    Vision2CodeServer.run_webserver()
