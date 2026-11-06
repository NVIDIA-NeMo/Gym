# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON-lines scoring worker; runs only in Python 3.12, without Gym or Ray."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time


LOG = logging.getLogger(__name__)


def _watch_parent(parent: int) -> None:
    # Ray may SIGKILL an actor, bypassing its cleanup hooks. Release its GPU models.
    while os.getppid() == parent:
        time.sleep(2)
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("LongMT scoring requires Python 3.12")
    threading.Thread(target=_watch_parent, args=(int(sys.argv[1]),), daemon=True).start()
    # Keep both Python and native library stdout away from the JSON response pipe.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1) as responses:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        from scorer import SegaleScorer

        config = json.loads(sys.stdin.readline())
        LOG.info("Loading LongMT Python 3.12 scorer pid=%d", os.getpid())
        scorer = SegaleScorer(**config)
        responses.write(json.dumps({"ready": True}) + "\n")
        for line in sys.stdin:
            try:
                LOG.info("Starting document scoring")
                result = scorer.score(**json.loads(line))
                LOG.info("Completed document scoring")
            except Exception as exc:
                LOG.exception("Document scoring failed")
                result = {"error": f"{type(exc).__name__}: {exc}"}
            responses.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
