# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from benchmarks.visualwebarena.run import SiteControl


@pytest.fixture
def service():
    state = {
        "triggers": 0,
        "polls": 0,
        "lost_trigger": False,
        "lost_status": False,
        "immediate_ready": False,
        "delayed_start": False,
        "ongoing": False,
        "failure": False,
        "redirect": False,
        "stolen": 0,
        "busy": False,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/reset":
                state["triggers"] += 1
                if state["lost_trigger"]:
                    self.close_connection = True
                    return
                if state["redirect"]:
                    self.send_response(302)
                    self.send_header("Location", "/stolen")
                    self.end_headers()
                    return
                self.send_response(202)
                body = b"Reset ongoing"
            elif self.path == "/status":
                state["polls"] += 1
                if state["lost_status"] and state["polls"] == 1:
                    self.close_connection = True
                    return
                self.send_response(500 if state["failure"] else 200)
                if state["immediate_ready"] or (state["delayed_start"] and state["polls"] == 1):
                    body = b"Ready for duty!"
                elif state["ongoing"] or state["polls"] < (4 if state["lost_status"] else 3):
                    body = b"Reset ongoing"
                else:
                    body = b"Ready for duty!"
            elif self.path == "/healthz":
                self.send_response(200)
                body = json.dumps(
                    {
                        "status": "ok",
                        "sessions": int(state["busy"]),
                        "creating": 0,
                        "cleanup_pending": 0,
                        "late_release_pending": 0,
                        "browser_provider": {"active_leases": int(state["busy"])},
                    }
                ).encode()
            else:
                state["stolen"] += 1
                self.send_response(200)
                body = b"unexpected"
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    control = SiteControl(
        reset_url=base + "/reset",
        status_url=base + "/status",
        health_urls=[base + "/healthz"],
        token_file=None,
        timeout_seconds=0.5,
        poll_seconds=0.001,
    )
    try:
        yield control, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mode", ["normal", "lost_trigger", "lost_status"])
def test_reset_202_and_ambiguous_network_responses_never_retrigger(service, mode):
    control, state = service
    if mode != "normal":
        state[mode] = True
    assert control.require_idle()[0]["sessions"] == 0
    control.reset()
    assert state["triggers"] == 1 and state["polls"] >= 2


@pytest.mark.parametrize(
    "mode,error", [("failure", RuntimeError), ("redirect", RuntimeError), ("ongoing", TimeoutError)]
)
def test_reset_failures_are_bounded_and_do_not_follow_redirects(service, mode, error):
    control, state = service
    state[mode] = True
    control.timeout_seconds = 0.03
    with pytest.raises(error):
        control.reset()
    assert state["triggers"] == 1 and state["stolen"] == 0


def test_active_browser_workers_prevent_global_reset(service):
    control, state = service
    state["busy"] = True
    with pytest.raises(RuntimeError, match="healthy and idle"):
        control.require_idle()
    assert state["triggers"] == 0


@pytest.mark.parametrize("lost_trigger", [False, True])
def test_acknowledged_or_lost_trigger_cannot_accept_a_stale_ready_state(service, lost_trigger):
    control, state = service
    state["lost_trigger"] = lost_trigger
    state["immediate_ready"] = True
    control.timeout_seconds = 0.03
    with pytest.raises(TimeoutError, match="did not prove a reset transition"):
        control.reset()
    assert state["triggers"] == 1 and state["polls"] > 0


def test_202_then_stale_ready_waits_for_the_actual_reset_transition(service):
    control, state = service
    state["delayed_start"] = True
    control.reset()
    assert state["triggers"] == 1 and state["polls"] >= 3
