# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys
import time

import pytest

from nemo_gym.web.browser_instance_pool import LocalBrowserInstancePoolProvider
from nemo_gym.web.browser_session import BrowserSessionError


@pytest.mark.parametrize(
    "reply,expected", [("431\n", ":431"), ("0\n", ":0"), ("garbage\n", None), ("65536\n", None), ("", None)]
)
def test_displayfd_uses_a_real_child_descriptor_and_rejects_invalid_replies(monkeypatch, tmp_path, reply, expected):
    provider = LocalBrowserInstancePoolProvider(size=1, port_base=42000, log_dir=str(tmp_path))
    slot = provider._slots[0]
    popen = subprocess.Popen
    invocations = []

    def launch(command, **kwargs):
        invocations.append((command, kwargs))
        fd = int(command[command.index("-displayfd") + 1])
        code = f"import os,time; os.write({fd},{reply.encode()!r}); os.close({fd}); time.sleep(30)"
        return popen([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    try:
        with (tmp_path / "server.log").open("ab") as log:
            if expected is None:
                with pytest.raises(BrowserSessionError):
                    provider._launch_xvfb(slot, log, time.monotonic() + 2)
            else:
                provider._launch_xvfb(slot, log, time.monotonic() + 2)
                assert slot.display == expected
        command, kwargs = invocations[0]
        assert command[0:2] == ["Xvfb", "-displayfd"]
        assert kwargs["pass_fds"] == (int(command[2]),)
    finally:
        provider._stop(slot)


def test_explicit_occupied_display_never_starts_or_cleans_another_server(monkeypatch, tmp_path):
    provider = LocalBrowserInstancePoolProvider(size=1, port_base=42000, display_base=350, log_dir=str(tmp_path))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **kw: pytest.fail("must not start on another server's display")
    )
    with (tmp_path / "server.log").open("ab") as log, pytest.raises(BrowserSessionError, match="already in use"):
        provider._launch_xvfb(provider._slots[0], log, time.monotonic() + 2)
    assert provider._slots[0].xvfb is None


@pytest.mark.skipif(not shutil.which("Xvfb") or not shutil.which("xdpyinfo"), reason="native Xvfb tools unavailable")
def test_two_real_xservers_allocate_distinct_displays_and_cleanup_is_owned(tmp_path):
    providers = [
        LocalBrowserInstancePoolProvider(size=1, port_base=42000 + i, log_dir=str(tmp_path)) for i in range(2)
    ]
    try:
        for index, provider in enumerate(providers):
            with (tmp_path / f"xvfb-{index}.log").open("ab") as log:
                provider._launch_xvfb(provider._slots[0], log, time.monotonic() + 10)
        displays = [provider._slots[0].display for provider in providers]
        assert len(set(displays)) == 2
        for display in displays:
            subprocess.run(["xdpyinfo", "-display", display], capture_output=True, check=True, timeout=3)
        providers[1]._stop(providers[1]._slots[0])
        subprocess.run(["xdpyinfo", "-display", displays[0]], capture_output=True, check=True, timeout=3)
    finally:
        for provider in providers:
            provider._stop(provider._slots[0])
