# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-turn ABC composition with the released MiMo music reward."""

import asyncio
import json
import logging
import os
import re
import signal
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import ConfigDict, Field, JsonValue, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.mimo_music.scorer.pipeline import extract_abc
from resources_servers.mimo_music.setup_abc2midi import ensure_abc2midi


logger = logging.getLogger(__name__)


class MimoMusicResourcesServerConfig(BaseResourcesServerConfig):
    """Bound scorer execution independently of model generation."""

    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    num_processes: int = Field(default=4, gt=0)
    score_timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)


class MimoMusicVerifyRequest(BaseVerifyRequest):
    """Retain task provenance without making it part of the music reward."""

    model_config = ConfigDict(extra="allow")


class MimoMusicVerifyResponse(BaseVerifyResponse):
    """Native graded reward plus diagnostics from the released scorer."""

    model_config = ConfigDict(extra="allow")
    abc_extracted: bool
    scorer_details: dict[str, JsonValue] = Field(default_factory=dict)


def assistant_text(response: NeMoGymResponse) -> str:
    """Use only the final assistant message, excluding reasoning channels/tags."""
    for item in reversed(response.output):
        if item.type != "message" or item.role != "assistant":
            continue
        text = "\n".join(part.text for part in item.content if part.type == "output_text")
        text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
        # Some providers omit the opening tag; an unclosed block is not an answer.
        text = re.split(r"</(?:think|thinking)>", text)[-1]
        return re.split(r"<(?:think|thinking)>", text)[0].strip()
    return ""


class MimoMusicResourcesServer(SimpleResourcesServer):
    """Run the CPU scorer in cancellable, concurrency-bounded process groups."""

    config: MimoMusicResourcesServerConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()
    _abc2midi: str = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        self._abc2midi = ensure_abc2midi()
        self._semaphore = asyncio.Semaphore(self.config.num_processes)

    async def verify(self, body: MimoMusicVerifyRequest) -> MimoMusicVerifyResponse:
        text = assistant_text(body.response)
        abc = extract_abc(text)
        # Reverification may include old reward/diagnostic fields; never trust them.
        payload = body.model_dump()
        payload.update(
            reward=0.0,
            abc_extracted=abc is not None,
            scorer_details={},
            mask_sample=False,
            failure_kind=None,
            failure_reason=None,
        )
        result = MimoMusicVerifyResponse(**payload)
        if not abc:
            result.scorer_details = {"skip": "empty_abc"}
            return result

        async with self._semaphore:
            try:
                # A thread timeout cannot stop feature extraction or its renderer child.
                env = dict(os.environ)
                env["ABC2MIDI_BIN"] = self._abc2midi
                root = str(Path(__file__).resolve().parents[2])
                env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
                # The parent owns scratch storage, so SIGKILL cannot leak tune files.
                with tempfile.TemporaryDirectory(prefix="mimo-music-") as work_dir:
                    env["TMPDIR"] = work_dir
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "resources_servers.mimo_music.score_worker",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                        env=env,
                    )
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(json.dumps({"abc": abc}).encode()),
                            timeout=self.config.score_timeout_seconds,
                        )
                    except (TimeoutError, asyncio.CancelledError):
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await proc.wait()
                        raise
                if proc.returncode:
                    raise RuntimeError(stderr.decode(errors="replace")[-1500:])
                scored = json.loads(stdout.decode(errors="replace"))
                result.scorer_details = scored["scorer_details"]
                reward = float(scored["reward"])
                if not 0.0 <= reward <= 1.0:
                    raise ValueError("Scorer returned a reward outside [0, 1]")
                result.reward = reward
            except TimeoutError:
                result.mask_sample = True
                result.failure_kind = "mimo_music:scorer_timeout"
                result.failure_reason = f"Scorer exceeded {self.config.score_timeout_seconds}s"
            except Exception as error:
                # This is the process boundary, not a bad-composition rejection.
                logger.exception("Music scorer failed")
                result.mask_sample = True
                result.failure_kind = "mimo_music:scorer_error"
                result.failure_reason = f"{type(error).__name__}: {error}"
        return result


if __name__ == "__main__":
    MimoMusicResourcesServer.run_webserver()
