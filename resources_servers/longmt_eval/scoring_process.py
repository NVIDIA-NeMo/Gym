# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A persistent evaluator subprocess owned by one serial Ray actor."""

from __future__ import annotations

import atexit
import io
import json
import logging
import os
import queue
import selectors
import subprocess
import threading
import time
from pathlib import Path


LOG = logging.getLogger(__name__)


class ScoringProcess:
    """Keep scoring models loaded, with bounded requests and explicit cleanup."""

    def __init__(self, python: Path, worker: Path, *, config: dict, timeout: float = 1800) -> None:
        self.timeout = timeout
        self.python = python
        self.worker = worker
        self.config = dict(config)
        self._closed = False
        self._start()
        atexit.register(self.close)

    def _start(self) -> None:
        LOG.info("Starting LongMT scoring worker with %s", self.python)
        env = dict(os.environ)
        # Ray's Python 3.13 site-packages must never leak into the 3.12 evaluator.
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        self.process = subprocess.Popen(
            [str(self.python), "-u", str(self.worker), str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=env,
        )
        assert self.process.stdin is not None
        os.set_blocking(self.process.stdin.fileno(), False)
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read, args=(self.process, self._lines), daemon=True)
        self._reader.start()
        try:
            ready = self._request(self.config)
            if ready != {"ready": True}:
                raise RuntimeError(f"LongMT scorer did not initialize: {ready}")
        except BaseException:
            self._stop()
            raise

    @staticmethod
    def _read(process: subprocess.Popen[bytes], lines: queue.Queue[str]) -> None:
        assert process.stdout is not None
        try:
            # Only stdin needs to be unbuffered. Raw stdout.readline() reads one
            # byte per syscall, which makes large JSON responses needlessly slow.
            with io.BufferedReader(process.stdout) as stdout:
                for line in stdout:
                    lines.put(line.decode("utf-8", errors="replace"))
        finally:
            lines.put("")

    def _request(self, payload: dict) -> dict:
        try:
            if self.process.poll() is not None:
                raise RuntimeError(f"LongMT scorer exited with code {self.process.returncode}")
            assert self.process.stdin is not None
            deadline = time.monotonic() + self.timeout
            pending = memoryview((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
            # A worker can stop reading even after readiness. Never block on its pipe.
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"LongMT scoring exceeded {self.timeout}s during request transmission")
                    if not selector.select(timeout=min(30, remaining)):
                        LOG.info("LongMT scorer pid=%s waiting to receive request", self.process.pid)
                        continue
                    try:
                        written = os.write(self.process.stdin.fileno(), pending[:65536])
                    except BlockingIOError:
                        continue
                    pending = pending[written:]
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
            self._stop()
            raise

    def score(self, source_text: str, mt_text: str, target_language: str) -> dict:
        """Score one aligned document pair using the resident models."""
        if self._closed:
            raise RuntimeError("LongMT scoring process is closed")
        if self.process.poll() is not None:
            self._stop()
            self._start()
        return self._request({"source_text": source_text, "mt_text": mt_text, "target_language": target_language})

    def close(self) -> None:
        """Terminate and reap the owned process, including failed initializations."""
        self._closed = True
        atexit.unregister(self.close)
        self._stop()

    def _stop(self) -> None:
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
