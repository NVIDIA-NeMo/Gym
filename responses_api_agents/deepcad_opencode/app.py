# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode CAD episodes: edit model.py, inspect images, grade only final code."""

import base64
import copy
import json
import tempfile
from pathlib import Path

from anyio import CancelScope
from fastapi import Request
from pydantic import Field

from nemo_gym.client.openai_client import NeMoGymEasyInputMessage
from nemo_gym.server_runtime.base_server import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.deepcad.dataset.augment_inputs import view_prompt
from resources_servers.deepcad.task_data import TaskData
from responses_api_agents.opencode_agent.observability import scope_opencode_trajectory
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


HERE = Path(__file__).resolve().parent
GYM = HERE.parents[1]
INSTRUCTIONS = """Reconstruct the target CAD geometry by writing and editing /workspace/model.py.
The final model.py is the ONLY artifact graded; prose, code fences, preview meshes and earlier files are not graded.
Start model.py with import cadquery as cq explicitly and assign the final solid to result. No imports are supplied for you.
Target images are attached to this request. Inspect ALL of them. Use cad_cad_render to execute your current program and receive rendered images
in the same cameras as the target. Before finishing, you MUST call cad_cad_render and inspect the returned images. Revise the program when needed. Errors return tracebacks.
You may also run your own Python rendering code with bash, save PNGs under /workspace, and inspect them with cad_cad_view.
A helper is available: python /workspace/.cad/cad_render.py /workspace/model.py --output /workspace/my_views
For importing the helper from your own script, add /workspace/.cad to sys.path and import cad_render.
No target STEP file, reference code, or reward oracle is available. Do not attempt to access anything outside the workspace.
When satisfied, leave your final complete code in model.py and finish. Do not delegate to subagents or compact history.
"""


class DeepCADOpenCodeConfig(OpenCodeSandboxedAgentConfig):
    max_steps: int = Field(default=10, ge=1)
    max_code_bytes: int = Field(default=200_000, ge=1)
    max_input_image_bytes: int = Field(default=8 * 1024 * 1024, ge=1)
    token_id_capture: bool = True


class DeepCADOpenCodeRunRequest(OpenCodeSandboxedAgentRunRequest):
    verifier_metadata: TaskData


def image_content(attachment: dict) -> dict:
    """Preserve OpenCode image attachments as Responses tool image content."""
    mime = attachment.get("mime", attachment.get("mimeType", ""))
    url = attachment.get("url", "")
    if not mime.startswith("image/") or not url.startswith("data:image/"):
        raise ValueError("CAD image attachments must contain inline image data")
    return {"type": "input_image", "image_url": url, "detail": "high"}


class DeepCADOpenCodeAgent(OpenCodeSandboxedAgent):
    config: DeepCADOpenCodeConfig

    async def _create_opencode_config(self, request: Request, *, image_input: bool = False) -> dict:
        config = await super()._create_opencode_config(request, image_input=image_input)
        config["compaction"] = {"auto": False, "prune": False}
        config.setdefault("permission", {}).update({"task": "deny", "webfetch": "deny", "websearch": "deny"})
        agents = config.setdefault("agent", {})
        agents.setdefault("build", {}).update({"steps": self.config.max_steps, "prompt": INSTRUCTIONS})
        for name in ("general", "explore", "title", "summary", "compaction"):
            agents.setdefault(name, {})["disable"] = True
        config.setdefault("provider", {}).setdefault("nemo_gym", {}).setdefault("models", {}).setdefault(
            "dummy_model", {}
        )["modalities"] = {"input": ["text", "image"], "output": ["text"]}
        config.setdefault("mcp", {})["cad"] = {
            "type": "local",
            "command": ["python", "/workspace/.cad/cad_tools.py"],
            "enabled": True,
            "timeout": 150000,
        }
        return config

    def _opencode_export_to_output_items(self, exported: dict) -> list:
        # OpenCode exports MCP image results as tool attachments; the base text-only converter drops them.
        clean = copy.deepcopy(exported)
        attachments = {}
        for message in clean["messages"]:
            for part in message["parts"]:
                if part["type"] == "tool":
                    images = part["state"].get("attachments", [])
                    if images:
                        attachments[part["callID"]] = [image_content(image) for image in images]
        output = super()._opencode_export_to_output_items(clean)
        for item in output:
            if item.type == "function_call_output" and item.call_id in attachments:
                item.output = [{"type": "input_text", "text": item.output or ""}, *attachments[item.call_id]]
        return output

    async def _prepare_workspace(self, sandbox, body: DeepCADOpenCodeRunRequest, directory: Path) -> str:
        metadata = body.verifier_metadata
        images = []
        descriptions = []
        for message in body.responses_create_params.input:
            if getattr(message, "role", None) != "user":
                continue
            content = message.content
            if isinstance(content, str):
                descriptions.append(content)
                continue
            for part in content:
                part = part.model_dump() if hasattr(part, "model_dump") else part
                if part["type"] == "input_text":
                    descriptions.append(part["text"])
                elif part["type"] == "input_image":
                    url = part["image_url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise ValueError(
                            "Materialize CAD target images as inline data URLs before starting the episode"
                        )
                    data = base64.b64decode(url.split(",", 1)[1], validate=True)
                    if len(data) > self.config.max_input_image_bytes:
                        raise ValueError("Target image exceeds max_input_image_bytes")
                    local = directory / f"target-{len(images)}.png"
                    import io

                    from PIL import Image

                    with Image.open(io.BytesIO(data)) as image:
                        image.convert("RGB").save(local)
                    remote = f"/workspace/target-{len(images)}.png"
                    await sandbox.upload(local, remote)
                    images.append(remote)
        if not images:
            raise ValueError("CAD OpenCode requires target images")
        for name in ("cad_tools.py", "cad_render.py"):
            await sandbox.upload(HERE / name, f"/workspace/.cad/{name}")
        for name in ("prepare_gym.py", "augment_inputs.py"):
            relative = f"resources_servers/deepcad/dataset/{name}"
            await sandbox.upload(GYM / relative, f"/workspace/.cad/{relative}")
        settings = directory / "settings.json"
        settings.write_text(
            json.dumps({"view_mode": metadata.view_mode, "color": metadata.foreground_color or "teal"})
        )
        await sandbox.upload(settings, "/workspace/.cad/settings.json")
        views = view_prompt(metadata.view_mode, engineering_annotations=metadata.engineering_annotations).partition(
            "Output ONLY"
        )[0]
        return (
            INSTRUCTIONS
            + "\n"
            + views
            + "\nTarget images:\n"
            + "\n".join(images)
            + "\n"
            + "\n".join(text.partition("Output ONLY")[0] for text in descriptions)
        )

    async def _collect_code(self, sandbox, directory: Path) -> tuple[str, str]:
        check = await sandbox.exec(
            'python -c \'import os,stat; p="/workspace/model.py"; s=os.lstat(p) if os.path.lexists(p) else None; '
            f"print(bool(s and stat.S_ISREG(s.st_mode) and 0 < s.st_size <= {self.config.max_code_bytes}))'",
            timeout_s=30,
        )
        if check.error_type or check.return_code:
            raise RuntimeError(f"Final workspace inspection failed: {check.stderr}")
        if check.stdout.strip() != "True":
            return "", "missing_or_invalid_model_file"
        path = directory / "model.py"
        await sandbox.download("/workspace/model.py", path)
        if path.stat().st_size > self.config.max_code_bytes:
            return "", "oversized_model_file"
        try:
            return path.read_text(encoding="utf-8"), "collected"
        except UnicodeError:
            return "", "invalid_code_encoding"

    async def run(self, request: Request, body: DeepCADOpenCodeRunRequest) -> dict:
        async with self._sem:
            key = request.session[SESSION_ID_KEY]
            rollout_id = self.rollout_id_from_run(body)
            sandbox = await self._start_sandbox()
            self._sandbox_id_to_sandbox[key] = sandbox
            request._cookies = request.cookies | {"sandbox_id": key}
            request.state._ng_observation_invocation_id = rollout_id
            try:
                with tempfile.TemporaryDirectory(prefix="cad-opencode-") as tmp:
                    directory = Path(tmp)
                    prompt = await self._prepare_workspace(sandbox, body, directory)
                    model_body = body.responses_create_params.model_copy(deep=True)
                    # Keep inline images in the model request so the upstream harness attaches them.
                    images = []
                    for message in model_body.input:
                        if getattr(message, "role", None) != "user" or isinstance(message.content, str):
                            continue
                        for part in message.content:
                            part = part.model_dump() if hasattr(part, "model_dump") else part
                            if part["type"] == "input_image":
                                images.append(part)
                    model_body.input = [
                        NeMoGymEasyInputMessage(role="user", content=[{"type": "input_text", "text": prompt}, *images])
                    ]
                    response = await self.responses(request, model_body)
                    run_result = self._sandbox_id_to_run_result.get(key, {}).copy()
                    observations = run_result.pop("_ng_agent_observations", None)
                    trajectory = run_result.pop("_ng_trajectory", None)
                    code, file_status = await self._collect_code(sandbox, directory)
                    # Freeze the artifact before destroying the episode workspace. Never grade its STEP files.
                    if self.config.artifacts_dir:
                        artifact = Path(self.config.artifacts_dir) / key
                        artifact.mkdir(parents=True, exist_ok=True)
                        (artifact / "model.py").write_text(code)
                        (artifact / "file-status.json").write_text(json.dumps({"status": file_status}))
                    await sandbox.stop()
                    sandbox = None
                    forced_zero = self.config.execution_failure_reward_zero and run_result.get(
                        "opencode_failed", False
                    )
                    grading = body.model_dump(mode="json") | {
                        "response": response.model_dump(mode="json"),
                        "final_code": "" if forced_zero else code,
                    }
                    verified = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path="/verify",
                        json=grading,
                        cookies=request.cookies,
                    )
                    await raise_for_status(verified)
                    result = await get_response_json(verified)
                    result.update(
                        response=response.model_dump(mode="json"), final_file_status=file_status, **run_result
                    )
                    if forced_zero:
                        if result["reward"] != 0 or result.get("mask_sample", False):
                            raise ValueError("CAD verifier must return unmasked zero for an empty final program")
                        result["failure_kind"] = "deepcad:agent_incomplete"
                    if trajectory is not None:
                        result["ng_trajectory"] = scope_opencode_trajectory(trajectory, body, rollout_id).model_dump(
                            mode="json"
                        )
                    if observations is not None:
                        result["ng_agent_observations"] = observations.model_dump(mode="json")
                    if self.config.artifacts_dir:
                        (artifact / "result.json").write_text(json.dumps(result))
                    return result
            finally:
                try:
                    with CancelScope(shield=True):
                        if sandbox is not None:
                            await sandbox.stop()
                finally:
                    self._sandbox_id_to_sandbox.pop(key, None)
                    self._sandbox_id_to_run_result.pop(key, None)
                    del request.state._ng_observation_invocation_id


if __name__ == "__main__":
    DeepCADOpenCodeAgent.run_webserver()
