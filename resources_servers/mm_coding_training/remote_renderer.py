# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh remote execution for each answer; references and judge credentials stay on the server."""

import asyncio
import base64
import io
import shlex
import tempfile
from pathlib import Path

import anyio
from PIL import Image

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec


LANGUAGES = {
    "chart2code": ("python", "answer.py"),
    "mmsvg_img2svg": ("svg", "answer.svg"),
    "mmsvg_text2svg": ("svg", "answer.svg"),
    "image2struct_latex": ("latex", "answer.tex"),
    "design2code_full": ("html", "answer.html"),
}
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class RendererSetupError(RuntimeError):
    """The pinned renderer image did not pass its dependency preflight."""


def image_uri(data: bytes) -> str:
    """Validate a bounded renderer/reference image before including it in a judge request."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or exceeds 32 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"} or image.width * image.height > 32_000_000:
            raise ValueError("Unsupported or oversized image")
        mime = Image.MIME[image.format]
        image.verify()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


class RemoteRenderer:
    """Execute the pinned renderer through Gym's provider-neutral sandbox API."""

    def __init__(self, provider: dict, spec: SandboxSpec, *, execution_timeout: float, ready_timeout: float) -> None:
        self.provider = provider
        self.spec = spec
        self.execution_timeout = execution_timeout
        self.ready_timeout = ready_timeout

    async def render(self, benchmark: str, source: str) -> bytes | None:
        """Return validated PNG bytes, or None only for a measured candidate execution failure."""
        artifact = LANGUAGES[benchmark][1]
        spec = SandboxSpec(
            image=self.spec.image,
            ttl_s=self.spec.ttl_s,
            ready_timeout_s=self.ready_timeout,
            workdir="/workspace",
            env={"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"},
            resources=self.spec.resources,
            metadata=self.spec.metadata,
            provider_options=self.spec.provider_options,
            files={
                "/workspace/renderer.py": Path(__file__).with_name("renderer.py").read_text(),
                f"/workspace/{artifact}": source,
            },
        )
        sandbox = AsyncSandbox(self.provider, spec)
        try:
            await sandbox.start()
            setup = await sandbox.exec("python /workspace/renderer.py --check", timeout_s=60)
            if setup.error_type is not None or setup.return_code != 0:
                raise RendererSetupError("Renderer image failed its dependency preflight")
            command = shlex.join(["python", "/workspace/renderer.py", benchmark, artifact, "render.png"])
            result = await sandbox.exec(command, cwd="/workspace", timeout_s=self.execution_timeout)
            if result.error_type is not None:
                raise RuntimeError(f"Renderer execution was inconclusive ({result.error_type})")
            if result.return_code is None or result.return_code < 0 or result.return_code in {137, 139, 143}:
                raise RuntimeError("Renderer has no measured exit or an unknown-cause signal exit")
            if result.return_code != 0:
                return None
            with tempfile.TemporaryDirectory(prefix="mm-coding-render-") as directory:
                output = Path(directory) / "render.png"
                await sandbox.download("/workspace/render.png", output)
                data = output.read_bytes()
                image_uri(data)
                return data
        except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as error:
            raise RuntimeError("Remote renderer could not produce a valid measurement") from error
        finally:
            # A client disconnect cancels this request, and an unshielded stop then never runs.
            # 120 s is the bound that LentSandboxes gives its own stop (STOP_TIMEOUT_S).
            with anyio.CancelScope(shield=True), anyio.fail_after(120):
                await sandbox.stop()
