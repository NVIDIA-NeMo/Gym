# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the installed EmberBridge episode engine and verify its native evidence."""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path

from pydantic import BaseModel, Field, StrictBool


HORIZON_AUDIT = """
import json
import sys
from pathlib import Path

from nanoverse.simulators.robodojo.horizon import endpoints

root, classification, profile = sys.argv[1:]
root = Path(root)
grading = root / "grading.json"
print(json.dumps(endpoints(
    root / "evaluation", classification, profile=profile,
    grading=json.loads(grading.read_text()) if grading.is_file() else None,
)))
"""

CLEANUP_AUDIT = """
import sys
from nanoverse.environment.processes import cleanup_receipt_valid

raise SystemExit(0 if cleanup_receipt_valid(sys.argv[1], sys.argv[2]) else 1)
"""

SUPERVISOR_PATH = """
from pathlib import Path
import nanoverse.environment

print(Path(nanoverse.environment.__file__).with_name("supervise.py"))
"""


class EpisodeError(RuntimeError):
    """An episode without a verified measurement."""


class CleanupUnconfirmed(EpisodeError):
    """The runtime did not establish that it released the simulator and SDK session."""


class HorizonScores(BaseModel):
    within_native: StrictBool
    within_budget: StrictBool
    score_within_native: float = Field(ge=0, le=1)
    score_within_budget: float = Field(ge=0, le=1)
    task_step_limit: int = Field(gt=0)
    control_budget: int = Field(gt=0)


class Episode:
    def __init__(
        self,
        directory: Path,
        bridge_python: Path,
        simulation_python: Path,
        policy: Path | None,
        timeout: float,
        cleanup_timeout: float,
    ) -> None:
        self.directory = directory
        self.bridge_python = bridge_python
        self.simulation_python = simulation_python
        self.policy = policy
        self.timeout = timeout
        self.cleanup_timeout = cleanup_timeout
        self.engine = (str(bridge_python), "-m", "ember_bridge.cli")
        self.environment = {
            name: os.environ[name]
            for name in (
                "PATH",
                "LANG",
                "LC_ALL",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "LD_LIBRARY_PATH",
                "VK_ICD_FILENAMES",
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                "NANOVERSE_TTC_MANIFEST",
                "NANOVERSE_GUEST_PROGRAMS",
                "WARP_CACHE_PATH",
                "CUDA_CACHE_PATH",
            )
            if name in os.environ
        }
        home = directory / "home"
        home.mkdir(mode=0o700)
        no_decisions = directory / "decisions.env"
        no_decisions.touch()
        self.environment.update(
            HOME=str(home),
            NV_API_KEY="gym-model-server",
            OPENAI_DECISION_API_KEY_FILE=str(no_decisions),
            PYTHONDONTWRITEBYTECODE="1",
        )

    def export_evidence(self, root: Path) -> Path:
        """Copy the grading receipts and video from the SDK session directory."""
        exported = root.resolve() / self.directory.name
        exported.mkdir(parents=True, exist_ok=False)
        for relative in (
            "engine.log",
            "commands.log",
            "cleanup.json",
            "engine-result.json",
            "horizon.json",
            "native/result.json",
            "native/outcome.json",
            "native/actor/transcript.jsonl",
            "native/actor/ttc-runtime.json",
            "native/evaluation/metadata.json",
            "native/evaluation/reset.json",
            "native/evaluation/sensors.mp4",
        ):
            source = (self.directory / relative).resolve()
            if source.is_relative_to(self.directory) and source.is_file():
                path = exported / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, path)
        return exported

    async def execute(self, *args: str, timeout: float, log: Path | None = None) -> bytes:
        """Start one owned process group and drain it on cancellation or timeout."""
        with (log or self.directory / "commands.log").open("ab") as stderr:
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *args,
                    cwd=self.directory,
                    env=self.environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stderr if log is not None else asyncio.subprocess.PIPE,
                    stderr=stderr,
                    start_new_session=True,
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                await self.stop(await asyncio.shield(spawn))
                raise
            try:
                async with asyncio.timeout(timeout):
                    stdout, _ = await process.communicate()
                if process.returncode != 0:
                    raise EpisodeError(f"{Path(args[0]).name} exited with status {process.returncode}")
                return stdout or b""
            except BaseException:
                await self.stop(process)
                raise

    async def stop(self, process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            async with asyncio.timeout(self.cleanup_timeout):
                await process.wait()
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def run(self, configuration: dict) -> tuple[str, HorizonScores]:
        config_path = self.directory / "config.json"
        config_path.write_text(json.dumps(configuration, indent=2) + "\n")
        evidence = self.directory / "native"
        cleanup = self.directory / "cleanup.json"
        await self.execute(*self.engine, "validate", "--config", str(config_path), timeout=self.cleanup_timeout)
        supervisor = (
            (await self.execute(str(self.simulation_python), "-c", SUPERVISOR_PATH, timeout=self.cleanup_timeout))
            .decode()
            .strip()
        )
        try:
            await self.execute(
                str(self.bridge_python),
                supervisor,
                "--cleanup-receipt",
                str(cleanup),
                "--cleanup-id",
                self.directory.name,
                "--",
                *self.engine,
                "run",
                "--config",
                str(config_path),
                "--output",
                str(evidence),
                *(("--policy", str(self.policy)) if self.policy else ()),
                timeout=self.timeout,
                log=self.directory / "engine.log",
            )
        finally:
            try:
                await self.execute(
                    str(self.simulation_python),
                    "-c",
                    CLEANUP_AUDIT,
                    str(cleanup),
                    self.directory.name,
                    timeout=self.cleanup_timeout,
                )
            except (EpisodeError, OSError, TimeoutError) as error:
                raise CleanupUnconfirmed("Nanoverse did not confirm process cleanup") from error

        try:
            recorded = json.loads((evidence / "result.json").read_text())
            sdk = recorded.get("sdk_summary") or {}
            ttc = json.loads((evidence / "actor" / "ttc-runtime.json").read_text())
        except (OSError, ValueError) as error:
            raise CleanupUnconfirmed("EmberBridge SDK or TTC cleanup evidence could not be read") from error
        if sdk.get("closed") is not True or sdk.get("faults") or ttc.get("state") != "closed":
            raise CleanupUnconfirmed("EmberBridge did not confirm SDK and TTC cleanup")
        if any(ttc.get(key) for key in ("fault", "cleanup_error", "daemon_cleanup_error", "export_error")):
            raise CleanupUnconfirmed("TTC reported a cleanup error")

        result = json.loads(await self.execute(*self.engine, "result", str(evidence), timeout=self.cleanup_timeout))
        (self.directory / "engine-result.json").write_text(json.dumps(result, indent=2) + "\n")
        if result.get("schema") != "ember_bridge.evaluation_result.v1":
            raise EpisodeError("EmberBridge returned an unsupported evaluation record")
        if result.get("counted") is not True:
            raise EpisodeError(f"EmberBridge did not count the episode ({result.get('class', 'unknown')})")

        classification = result["class"]
        if classification == "scored":
            outcome = json.loads((evidence / "outcome.json").read_text())
            if outcome.get("schema") != "ember_bridge.outcome.v1" or outcome.get("readback_verified") is not True:
                raise EpisodeError("EmberBridge did not verify the native grade")
        elif classification not in {
            "policy_failure.decision_budget",
            "policy_failure.deadline",
            "policy_failure.context",
            "policy_failure.no_progress",
        }:
            raise EpisodeError(f"Unsupported counted outcome ({classification})")
        reset = json.loads((evidence / "evaluation" / "reset.json").read_text())
        metadata = reset.get("metadata") if isinstance(reset, dict) else None
        if not isinstance(metadata, dict):
            raise EpisodeError("Native RoboDojo reset did not record its simulator")
        # The PhysX session identifies itself as RoboDojo and has no Newton backend field.
        physics_backend = metadata.get("physics_backend")
        if physics_backend == "isaaclab3_newton":
            backend = "isaaclab3"
        elif metadata.get("simulator") == "RoboDojo" and physics_backend is None:
            backend = "isaacsim"
        else:
            backend = None
        if backend != configuration["simulator_backend"]:
            raise EpisodeError("Native RoboDojo simulator differs from the configured backend")
        horizon = await self.execute(
            str(self.simulation_python),
            "-c",
            HORIZON_AUDIT,
            str(evidence),
            classification,
            configuration["horizon_profile"],
            timeout=self.cleanup_timeout,
        )
        scores = HorizonScores.model_validate_json(horizon)
        (self.directory / "horizon.json").write_text(scores.model_dump_json(indent=2) + "\n")
        return classification, scores
