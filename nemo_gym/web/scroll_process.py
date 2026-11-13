# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A killable PyAutoGUI scroll, without changing parser-produced amounts.

Only the optional relaxed-scroll contract uses this boundary. The child does
not own Chromium or Playwright. On timeout it is killed and reaped before the
driver captures/evaluates the resulting page; the partial action is not retried.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time


LOG = logging.getLogger(__name__)


def run_scroll(direction: str, amount: int, point: tuple[int, int], *, timeout: float) -> None:
    if direction not in {"up", "down", "left", "right"}:
        raise ValueError("unsupported scroll direction")
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise ValueError("scroll amount must be a non-negative integer")
    started = time.monotonic()
    LOG.info("event=scroll_process_start direction=%s amount=%d timeout=%s", direction, amount, timeout)
    try:
        # subprocess.run kills and waits for its direct child on TimeoutExpired.
        # This module never starts grandchildren or invokes a shell.
        subprocess.run(
            [sys.executable, "-m", "nemo_gym.web.scroll_process"],
            input=json.dumps({"direction": direction, "amount": amount, "point": point}),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        LOG.error(
            "event=scroll_process_timeout amount=%d elapsed=%.3f child_reaped=true", amount, time.monotonic() - started
        )
        raise TimeoutError(
            f"scroll exceeded {timeout:g}s; child killed and reaped; partial action not retried"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"scroll subprocess failed with exit code {exc.returncode}: {exc.stderr[-2000:]}") from exc
    LOG.info("event=scroll_process_complete amount=%d elapsed=%.3f", amount, time.monotonic() - started)


def main() -> None:
    data = json.load(sys.stdin)
    os.environ.pop("WAYLAND_DISPLAY", None)
    import pyautogui

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    pyautogui.moveTo(*data["point"])
    amount, direction = data["amount"], data["direction"]
    if direction in {"up", "down"}:
        pyautogui.scroll(amount if direction == "up" else -amount)
    else:
        pyautogui.hscroll(amount if direction == "right" else -amount)


if __name__ == "__main__":
    main()
