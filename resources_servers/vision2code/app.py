# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 Vision2Code Authors
# SPDX-License-Identifier: Apache-2.0
# Judge recovery adapted from https://github.com/image2code/vision2code
# Source: vision2code/benchmark/run_benchmark.py; revision 000f2e162dd62fac5637c718e018d7123b3aff1e.
# License: Apache-2.0; see NOTICE.
"""Stateless verification of reference image → Python → Enroot render → rubric reward."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from PIL import Image
from pydantic import ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NgFailureEvent
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import JUDGE_FAILURES, is_judge_failure
from resources_servers.vision2code.evaluation import (
    aggregate_rating,
    build_rater_user_content,
    build_render_failure_rating,
    rating_to_reward,
    system_prompt,
)
from resources_servers.vision2code.image_metrics import apply_blend_a, image_metrics
from resources_servers.vision2code.judge_recovery import REJUDGE_PROMPT, recovery_schema, repair_prompt
from resources_servers.vision2code.judge_transport import request_judgment
from resources_servers.vision2code.prepare import PreparationConfig, Reference, read_reference
from resources_servers.vision2code.render import RenderResult, finish_task, render, stop_sandbox
from resources_servers.vision2code.rubrics import encode_image_to_data_url, inspect_candidate_image, parse_json_object
from resources_servers.vision2code.tests.verifier_fixture import VERIFIER_FIXTURE as VERIFIER_FIXTURE


LOGGER = logging.getLogger(__name__)


class Vision2CodeConfig(BaseResourcesServerConfig):
    judge_model_server: ModelServerRef
    judge_responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    judge_repair_attempts: int = Field(default=2, ge=0, strict=True)
    judge_repair_mode: Literal["retry", "repair", "rejudge"] = "retry"
    judge_transport_attempts: int = Field(default=3, ge=1, strict=True)
    preparation: PreparationConfig = Field(default_factory=PreparationConfig)
    judge_repair_max_output_tokens: int | None = Field(default=None, gt=0, strict=True)
    sandbox_image: str
    sandbox_provider: dict[str, dict] = Field(default_factory=lambda: {"enroot": {}})
    execution_timeout: float = Field(default=60, gt=0)
    render_concurrency: int = Field(default=8, ge=1)
    image_metrics_concurrency: int = Field(default=2, ge=1, strict=True)
    inconclusive_verification_retries: int = Field(default=1, ge=0, strict=True)
    sandbox_cleanup_attempts: int = Field(default=2, ge=1, strict=True)
    reward_power: float = Field(default=2.0, ge=1, allow_inf_nan=False)
    use_blended_reward: bool = Field(
        default=False,
        description="Use reference Blend A / 5: 10% foreground RGB, 40% mean F1, 50% raw LLM / 5; ignores reward_power.",
    )
    blended_reward_apply_judge_caps: bool = Field(
        default=False,
        description="When using Blend A, apply rubric caps to the unrounded judge term; keep pixel terms unchanged.",
    )
    spatial_uncapped_judge_squared: bool = False
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
    include_rendered_image: bool = False
    include_pixel_scores: bool = False


class Vision2CodeVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    id: str | None = None
    verifier_metadata: Reference
    rendered_image: str | None = None
    pixel_scores: dict[str, JsonValue] | None = None
    render: RenderResult | None = None
    render_attempts: list[dict[str, JsonValue]] = Field(default_factory=list)
    rubric_result: dict[str, JsonValue] | None = None
    reward_power: float = 2.0
    use_blended_reward: bool = False
    blended_reward_apply_judge_caps: bool = False
    spatial_uncapped_judge_squared: bool = False
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
    config: Vision2CodeConfig
    _render_slots: asyncio.Semaphore = PrivateAttr()
    _image_metrics_slots: asyncio.Semaphore = PrivateAttr()
    _pending_cleanup: dict[Path, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _render_tasks: set[asyncio.Task[RenderResult]] = PrivateAttr(default_factory=set)
    _shutting_down: bool = PrivateAttr(default=False)

    def model_post_init(self, context: object) -> None:
        self._render_slots = asyncio.Semaphore(self.config.render_concurrency)
        self._image_metrics_slots = asyncio.Semaphore(self.config.image_metrics_concurrency)
        super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await finish_task(asyncio.create_task(self._cleanup_sandboxes()))

        app.router.lifespan_context = lifespan
        return app

    async def _cleanup_sandboxes(self) -> None:
        self._shutting_down = True
        # Request cancellation can still be waiting for bounded startup/exec/stop operations.
        # Drain them before taking the snapshot of handles needing another teardown attempt.
        await asyncio.gather(*list(self._render_tasks), return_exceptions=True)

        async def cleanup(directory: Path, sandbox: AsyncSandbox) -> None:
            diagnostics: dict[str, JsonValue] = {}
            async with self._render_slots:
                stopped = await stop_sandbox(
                    sandbox, attempts=self.config.sandbox_cleanup_attempts, diagnostics=diagnostics
                )
            if stopped:
                self._pending_cleanup.pop(directory, None)
            else:
                LOGGER.error("Sandbox cleanup remains unresolved; diagnostics: %s/shutdown-cleanup.json", directory)
            try:
                (directory / "shutdown-cleanup.json").write_text(json.dumps(diagnostics, indent=2))
            except OSError:
                LOGGER.exception("Could not persist shutdown cleanup diagnostics for %s: %s", directory, diagnostics)

        await asyncio.gather(*(cleanup(path, sandbox) for path, sandbox in list(self._pending_cleanup.items())))

    async def _compute_image_metrics(self, reference: Path, candidate: Path) -> dict[str, JsonValue]:
        # Cancellation cannot stop a thread: retain its slot until the work has actually finished.
        async with self._image_metrics_slots:
            return await finish_task(asyncio.create_task(asyncio.to_thread(image_metrics, reference, candidate)))

    async def verify(self, body: Vision2CodeVerifyRequest) -> Vision2CodeVerifyResponse:
        ref = body.verifier_metadata
        spatial_squared = self.config.spatial_uncapped_judge_squared and ref.rubric == "spatialvlm_qa"
        use_blended_reward = self.config.use_blended_reward and not spatial_squared
        directory = Path(self.config.artifact_dir).resolve() / uuid.uuid4().hex
        directory.mkdir(parents=True)
        # Preserve Gym task/rollout bookkeeping, but never reuse verifier outputs
        # when an existing rollout is submitted for reverification.
        derived = set(Vision2CodeVerifyResponse.model_fields) - set(Vision2CodeVerifyRequest.model_fields)
        result = Vision2CodeVerifyResponse(
            **body.model_dump(exclude=derived | {"_ng_failure_class", "_ng_failure_id", "_ng_failure_judge_error"}),
            reward=0.0,
            reward_power=2.0 if spatial_squared else (1.0 if use_blended_reward else self.config.reward_power),
            use_blended_reward=use_blended_reward,
            blended_reward_apply_judge_caps=use_blended_reward and self.config.blended_reward_apply_judge_caps,
            spatial_uncapped_judge_squared=spatial_squared,
        )
        instance = result.model_extra.get("instance_config")
        if isinstance(instance, dict):
            result.model_extra["instance_config"] = {**instance, "mask_sample": False}
        result.artifacts = {"directory": str(directory), "result": str(directory / "result.json")}
        stage = "reference"
        render_directory = directory
        start = time.perf_counter()
        try:
            reference_path = Path(ref.image_path)
            if not reference_path.is_absolute():
                # Gym starts servers from their component directory; committed examples
                # use repository-relative paths so they remain portable across checkouts.
                reference_path = Path(__file__).resolve().parents[2] / reference_path
            raw, mime, width, height = read_reference(reference_path)
            if hashlib.sha256(raw).hexdigest() != ref.sha256 or (width, height) != (ref.width, ref.height):
                raise ValueError("Reference changed since task preparation; regenerate the JSONL")
            source_path = directory / "image.png"
            if mime == "image/png":
                source_path.write_bytes(raw)
            else:
                with Image.open(BytesIO(raw)) as source:
                    if source.mode == "CMYK":
                        source = source.convert("RGB")
                    source.save(source_path, format="PNG")
            result.artifacts["source_image"] = str(source_path)
            stage = "render"
            for attempt in range(self.config.inconclusive_verification_retries + 1):
                render_directory = directory if attempt == 0 else directory / f"render-attempt-{attempt + 1}"
                async with self._render_slots:
                    if self._shutting_down:
                        raise asyncio.CancelledError("Resources server is shutting down")
                    render_task = asyncio.create_task(
                        render(
                            code=response_text(body.response),
                            width=ref.width,
                            height=ref.height,
                            image=self.config.sandbox_image,
                            provider=self.config.sandbox_provider,
                            artifact_dir=render_directory,
                            timeout=self.config.execution_timeout,
                            cleanup_attempts=self.config.sandbox_cleanup_attempts,
                            pending_cleanup=self._pending_cleanup,
                        )
                    )
                    self._render_tasks.add(render_task)
                    try:
                        result.render = await render_task
                    finally:
                        self._render_tasks.discard(render_task)
                result.render_attempts.append(
                    {"directory": str(render_directory), "render": result.render.model_dump(mode="json")}
                )
                # Do not accumulate abandoned sandboxes while recovering an infrastructure failure.
                if (
                    result.render.status != "infrastructure_failure"
                    or render_directory in self._pending_cleanup
                    or self._shutting_down
                ):
                    break
            result.artifacts.update(
                {"code": str(render_directory / "code.py"), "execution": str(render_directory / "execution.json")}
            )
            if result.render.status == "infrastructure_failure":
                mask_failure(result, kind="provider_unavailable", reason=json.dumps(result.render.diagnostics))
                return result
            if result.render.status != "ok":
                result.rubric_result = build_render_failure_rating(result.render.status, dataset_name=ref.rubric)
                if result.use_blended_reward:
                    result.rubric_result = apply_blend_a(
                        result.rubric_result, apply_judge_caps=result.blended_reward_apply_judge_caps
                    )
                return result
            candidate = render_directory / "output.png"
            result.artifacts["image"] = str(candidate)
            if body.include_rendered_image:
                result.rendered_image = encode_image_to_data_url(candidate)
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
            stage = "judge_parse"
            judge_start = time.perf_counter()
            attempts = []
            result.judge_diagnostics = {"candidate_inspection": inspection, "attempts": attempts}
            request_params = params
            try:
                for attempt in range(self.config.judge_repair_attempts + 1):
                    judge = await request_judgment(
                        self.server_client,
                        server_name=self.config.judge_model_server.name,
                        params=NeMoGymResponseCreateParamsNonStreaming.model_validate(request_params),
                        attempts=self.config.judge_transport_attempts,
                    )
                    raw_response = response_text(judge)
                    result.judge_diagnostics["response"] = judge.model_dump(mode="json")
                    record = {
                        "mode": "initial" if attempt == 0 else self.config.judge_repair_mode,
                        "response": judge.model_dump(mode="json"),
                    }
                    attempts.append(record)
                    try:
                        parsed = parse_json_object(raw_response)
                        result.rubric_result = aggregate_rating(
                            parsed,
                            dataset_name=ref.rubric,
                            candidate_inspection=inspection,
                            reference_image_path=str(reference_path),
                            candidate_image_path=str(candidate),
                        )
                        break
                    except (ValueError, TypeError, KeyError) as exc:
                        record["error"] = str(exc)
                        if attempt == self.config.judge_repair_attempts:
                            raise
                        if self.config.judge_repair_mode == "retry":
                            request_params = params
                            continue
                        prompt = (
                            REJUDGE_PROMPT
                            if self.config.judge_repair_mode == "rejudge"
                            else repair_prompt(raw_response, str(exc), dataset_name=ref.rubric)
                        )
                        request_params = {
                            **params,
                            "temperature": 0.0,
                            "input": [
                                params["input"][0],
                                {
                                    "role": "user",
                                    "content": [*response_content, {"type": "input_text", "text": prompt}],
                                },
                            ],
                        }
                        if self.config.judge_repair_max_output_tokens is not None:
                            request_params["max_output_tokens"] = self.config.judge_repair_max_output_tokens
                        if self.config.judge_repair_mode == "rejudge":
                            request_params["text"] = {
                                "format": {
                                    "type": "json_schema",
                                    "name": "vision2code_rubric",
                                    "strict": True,
                                    "schema": recovery_schema(ref.rubric),
                                }
                            }
            finally:
                result.timings["judge_s"] = time.perf_counter() - judge_start
            if result.use_blended_reward or body.include_pixel_scores:
                stage = "image_metrics"
                result.pixel_scores = await self._compute_image_metrics(reference_path, candidate)
            if result.use_blended_reward:
                stage = "image_metrics"
                result.rubric_result = apply_blend_a(
                    result.rubric_result, result.pixel_scores, apply_judge_caps=result.blended_reward_apply_judge_caps
                )
            if spatial_squared:
                result.rubric_result["scoring_method"] = "uncapped_judge_squared"
            reward_rating = result.rubric_result["raw_score_0_to_5" if spatial_squared else "final_rating_0_to_5"]
            result.reward = rating_to_reward(reward_rating, power=result.reward_power)
            return result
        except asyncio.CancelledError:
            mask_failure(result, kind="cancelled", reason=f"Cancelled during {stage}")
            raise
        except JUDGE_FAILURES as exc:
            if not is_judge_failure(exc):
                raise
            mask_failure(result, kind="judge_failed", reason=str(exc))
            if isinstance(exc, NgFailureEvent):
                result.model_extra["_ng_failure_id"] = exc.event_id
            result.judge_diagnostics["error"] = str(exc)
            return result
        except Exception as exc:
            kind = "judge_unparseable" if stage == "judge_parse" else "verifier_error"
            mask_failure(result, kind=kind, reason=f"{stage}: {type(exc).__name__}: {exc}")
            return result
        finally:
            result.timings["total_s"] = time.perf_counter() - start
            execution_path = render_directory / "execution.json"
            if stage == "render" and execution_path.exists():
                result.render = RenderResult.model_validate_json(execution_path.read_text())
                result.artifacts.update({"code": str(render_directory / "code.py"), "execution": str(execution_path)})
                if not result.render_attempts or result.render_attempts[-1]["directory"] != str(render_directory):
                    result.render_attempts.append(
                        {"directory": str(render_directory), "render": result.render.model_dump(mode="json")}
                    )
            (directory / "judge.json").write_text(json.dumps(result.judge_diagnostics, indent=2))
            result.artifacts["judge"] = str(directory / "judge.json")
            # Avoid duplicating base64 reference images and full policy inputs.
            saved = result.model_dump(mode="json", exclude={"response", "responses_create_params", "rendered_image"})
            (directory / "result.json").write_text(json.dumps(saved, indent=2))


if __name__ == "__main__":
    Vision2CodeServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = Vision2CodeServer.run_webserver()
