# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own one RoboDojo simulator process for a fixed group of cases."""

import asyncio
import json
import os
import signal
import tempfile
from contextlib import suppress
from pathlib import Path


class NativePool:
    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="rd-pool-")
        self.process: asyncio.subprocess.Process | None = None
        self.seats: dict[str, dict] = {}

    async def start(
        self,
        python: Path,
        launcher: Path,
        cases: dict[str, dict],
        environment: dict[str, str],
        log_path: Path,
        timeout: float,
    ) -> None:
        directory = Path(self.directory.name)
        (directory / "cases.json").write_text(json.dumps(cases))
        (directory / "home").mkdir(mode=0o700)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as output:
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    str(python),
                    str(launcher),
                    str(directory),
                    cwd=directory,
                    env={**environment, "HOME": str(directory / "home")},
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                    start_new_session=True,
                )
            )
            try:
                self.process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                self.process = await asyncio.shield(spawn)
                raise
        sockets = [directory / f"{index}.sock" for index in range(len(cases))]
        try:
            async with asyncio.timeout(timeout):
                while not all(path.is_socket() for path in sockets):
                    if self.process.returncode is not None:
                        raise RuntimeError(f"RoboDojo simulator pool exited with status {self.process.returncode}")
                    await asyncio.sleep(0.25)
        except TimeoutError as error:
            raise RuntimeError(
                "RoboDojo simulator pool did not open all case sockets before the startup deadline"
            ) from error
        seats = json.loads((directory / "seats.json").read_text())
        if seats.keys() != cases.keys():
            raise ValueError("Simulator pool did not bind the requested RoboDojo cases")
        for index, case_id in enumerate(cases):
            expected = (str(sockets[index]), str(directory / f"case-{index}"))
            seat = seats[case_id]
            if (seat["native_pool_socket"], seat["native_pool_evidence"]) != expected:
                raise ValueError("Simulator pool returned an unexpected RoboDojo case slot")
        startup_gate = os.environ.get("ROBODOJO_SHARED_STARTUP_GATE")
        if startup_gate:
            gate = Path(startup_gate)
            group = os.environ.get("ROBODOJO_SHARED_STARTUP_GROUP")
            if not (group and group.isdigit()) or not gate.is_absolute() or not gate.is_dir():
                raise ValueError("Shared RoboDojo startup gate is invalid")
            accepted = f"[head_rgb] initial-ready slots={list(range(len(cases)))}"
            if accepted not in log_path.read_text(errors="replace").splitlines():
                raise ValueError("Shared RoboDojo initial head capture is missing valid pixels")
            marker = gate / f"ready-{group}.json"
            pending = gate / f"ready-{group}.partial"
            pending.write_text(json.dumps({"group": int(group), "cases": list(cases)}) + "\n")
            pending.replace(marker)
            async with asyncio.timeout(timeout):
                while not (gate / "release").is_file():
                    if self.process.returncode is not None:
                        raise RuntimeError("RoboDojo pool exited while waiting for the shared startup gate")
                    await asyncio.sleep(0.25)
        self.seats = seats

    async def close(self, timeout: float) -> None:
        try:
            if self.process is not None and self.process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(self.process.wait(), timeout)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    await self.process.wait()
        finally:
            self.directory.cleanup()
