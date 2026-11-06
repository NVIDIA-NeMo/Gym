# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A persistent evaluator subprocess owned by one serial Ray actor."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


LOG = logging.getLogger(__name__)


class ScoringProcess:
    """Keep scoring models loaded, with bounded requests and explicit cleanup."""

    def __init__(self, python: Path, worker: Path, *, config: dict, timeout: float = 1800) -> None:
        self.timeout = timeout
        env = dict(os.environ)
        # Ray's Python 3.13 site-packages must never leak into the 3.12 evaluator.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        self.process = subprocess.Popen(
            [str(python), "-u", str(worker), str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        atexit.register(self.close)
        try:
            ready = self._request(config)
            if ready != {"ready": True}:
                raise RuntimeError(f"LongMT scorer did not initialize: {ready}")
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._lines.put(line)
        finally:
            self._lines.put("")

    def _request(self, payload: dict) -> dict:
        try:
            if self.process.poll() is not None:
                raise RuntimeError(f"LongMT scorer exited with code {self.process.returncode}")
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"LongMT scoring exceeded {self.timeout}s")
                try:
                    line = self._lines.get(timeout=min(30, remaining))
                    break
                except queue.Empty:
                    LOG.info("LongMT scorer pid=%s still processing", self.process.pid)
            if not line:
                raise RuntimeError(f"LongMT scorer closed its response pipe (exit={self.process.poll()})")
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError("LongMT scorer returned a non-object response")
            return result
        except BaseException:
            self.close()
            raise

    def score(self, source_text: str, mt_text: str, target_language: str) -> dict:
        """Score one aligned document pair using the resident models."""
        return self._request({"source_text": source_text, "mt_text": mt_text, "target_language": target_language})

    def close(self) -> None:
        """Terminate and reap the owned process, including failed initializations."""
        atexit.unregister(self.close)
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdin:
            self.process.stdin.close()
        self._reader.join(timeout=5)
        if self.process.stdout:
            self.process.stdout.close()
