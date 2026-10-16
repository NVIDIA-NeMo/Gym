# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for a Resources Server's Environment Server session lifecycle.

:func:`check_resources_session_contract` drives the app a server builds through its real HTTP routes and middleware,
in process, the way an Environment Server calls a deployed server: typed seed, repeated seed, close, repeated close,
close of a never-seeded session, a seed after close, and a close without the session cookie after a seed whose reply
was lost. Call it from the server's own tests. Because it goes through the routes, it also catches a route that
another registration shadows.
"""

from collections.abc import Callable

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)


def check_resources_session_contract(
    app: FastAPI,
    seed: ResourcesSeedSessionRequest,
    *,
    keeps_state: bool,
    holds_state: Callable[[str], bool] | None = None,
) -> None:
    """Seed, re-seed, and close a session through ``app``, and raise AssertionError on a contract violation.

    ``seed`` must use a ``resources_session_id`` the app has not seen. With ``keeps_state``, a seed that arrives
    after its session closed, or after a close for an ID that was never seeded, must be rejected; a server that
    keeps no state has nothing such a seed could recreate.

    The check also seeds a second session and drops its cookie, as when the seed's reply is lost, then closes it
    without a cookie, as the Environment Server does. That close must succeed. With ``holds_state``, a callable that
    reports whether the server holds state for a ``resources_session_id``, both closes must release that state.
    """
    from fastapi.testclient import TestClient

    client = TestClient(app, raise_server_exceptions=False)
    seed_body = seed.model_dump(mode="json")
    close_body = ResourcesCloseSessionRequest(
        resources_session_id=seed.resources_session_id, episode_id=seed.episode_id
    ).model_dump(mode="json")

    for attempt in ("seed", "repeated seed"):
        response = client.post("/seed_session", json=seed_body)
        assert response.status_code == 200, f"{attempt} failed: {response.status_code} {response.text}"
        seeded = ResourcesSeedSessionResponse.model_validate(response.json())
        assert seeded.resources_session_id == seed.resources_session_id, f"{attempt} returned another session"

    for attempt in ("close", "repeated close"):
        response = client.post("/close_session", json=close_body)
        assert response.status_code == 200, f"{attempt} failed: {response.status_code} {response.text}"
        closed = ResourcesCloseSessionResponse.model_validate(response.json())
        assert closed.resources_session_id == seed.resources_session_id, f"{attempt} named another session"

    never_seeded = seed.model_copy(update={"resources_session_id": f"{seed.resources_session_id}-never-seeded"})
    response = client.post(
        "/close_session",
        json=ResourcesCloseSessionRequest(
            resources_session_id=never_seeded.resources_session_id, episode_id=seed.episode_id
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, f"close of a never-seeded session failed: {response.status_code}"

    lost_reply = seed.model_copy(update={"resources_session_id": f"{seed.resources_session_id}-lost-seed-reply"})
    client.cookies.clear()
    response = client.post("/seed_session", json=lost_reply.model_dump(mode="json"))
    assert response.status_code == 200, f"seed failed: {response.status_code} {response.text}"
    if holds_state is not None:
        assert holds_state(lost_reply.resources_session_id), "seed created no state for the check to release"
    # The Environment Server never received this seed's cookie, so its close carries none.
    client.cookies.clear()
    response = client.post(
        "/close_session",
        json=ResourcesCloseSessionRequest(
            resources_session_id=lost_reply.resources_session_id, episode_id=lost_reply.episode_id
        ).model_dump(mode="json"),
    )
    assert response.status_code == 200, f"close without a cookie failed: {response.status_code} {response.text}"

    if holds_state is not None:
        for closed_seed in (seed, lost_reply):
            assert not holds_state(closed_seed.resources_session_id), (
                f"close left state for {closed_seed.resources_session_id}"
            )

    if keeps_state:
        for late_seed in (seed, never_seeded, lost_reply):
            client.cookies.clear()
            response = client.post("/seed_session", json=late_seed.model_dump(mode="json"))
            assert response.status_code >= 400, (
                f"seed after close recreated {late_seed.resources_session_id}: {response.status_code}"
            )
