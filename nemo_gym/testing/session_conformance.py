# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for a Resources Server's Environment Server session lifecycle.

:func:`check_resources_session_contract` drives the app a server builds through its real HTTP routes and middleware,
in process, the way an Environment Server calls a deployed server: typed seed, repeated seed, a conflicting seed,
close, repeated close, a close that names another episode, close of a never-seeded session, a seed after close, a
close without the session cookie after a seed whose reply was lost, and optionally a verify after close. Call it from
the server's own tests. Because it goes through the routes, it also catches a route that another registration
shadows.
"""

from collections.abc import Callable, Mapping

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.core.failure_kinds import SESSION_EXPIRED, SESSION_LOST


def check_resources_session_contract(
    app: FastAPI,
    seed: ResourcesSeedSessionRequest,
    *,
    keeps_state: bool,
    holds_state: Callable[[str], bool] | None = None,
    verify_body: Mapping[str, object] | None = None,
) -> None:
    """Seed, re-seed, and close a session through ``app``, and raise AssertionError on a contract violation.

    ``seed`` must use a ``resources_session_id`` the app has not seen. With ``keeps_state``:

    - a seed that names the open session's ID with another episode is refused with 409;
    - a close of the closed session that names another episode is refused with 409;
    - a seed that arrives after its session closed, or after a close for an ID that was never seeded, is refused
      with 409 while the server remembers the close (``session_close_retry_window_seconds``). A server that keeps
      no state has nothing such a seed could recreate.

    The check also seeds a second session and drops its cookie, as when the seed's reply is lost, then closes it
    without a cookie, as the Environment Server does. That close must succeed. With ``holds_state``, a callable that
    reports whether the server holds state for a ``resources_session_id``, both closes must release that state.

    With ``verify_body`` (requires ``keeps_state``), a ``/verify`` sent with the closed session's cookie must
    return a masked sample whose ``failure_kind`` is ``session_lost`` or ``session_expired``, not a reward of 0.
    Pass it for a server whose score depends on per-rollout state.
    """
    from fastapi.testclient import TestClient

    if verify_body is not None and not keeps_state:
        raise ValueError("verify_body checks a stateful server's lost-session verify; it requires keeps_state")

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
    session_cookies = dict(client.cookies)

    if keeps_state:
        other_episode = seed.episode_id.model_copy(update={"rollout_id": f"{seed.episode_id.rollout_id}-other"})
        conflicting = seed.model_copy(update={"episode_id": other_episode})
        client.cookies.clear()
        response = client.post("/seed_session", json=conflicting.model_dump(mode="json"))
        assert response.status_code == 409, (
            f"a seed binding {seed.resources_session_id} to another episode must get 409, "
            f"got {response.status_code}: {response.text}"
        )
        client.cookies.update(session_cookies)

    for attempt in ("close", "repeated close"):
        response = client.post("/close_session", json=close_body)
        assert response.status_code == 200, f"{attempt} failed: {response.status_code} {response.text}"
        closed = ResourcesCloseSessionResponse.model_validate(response.json())
        assert closed.resources_session_id == seed.resources_session_id, f"{attempt} named another session"

    if keeps_state:
        response = client.post(
            "/close_session",
            json=ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id, episode_id=other_episode
            ).model_dump(mode="json"),
        )
        assert response.status_code == 409, (
            f"a close of {seed.resources_session_id} that names another episode must get 409, "
            f"got {response.status_code}: {response.text}"
        )

    if verify_body is not None:
        client.cookies.clear()
        client.cookies.update(session_cookies)
        response = client.post("/verify", json=dict(verify_body))
        assert response.status_code == 200, (
            f"verify of a closed session must answer with a masked sample, got {response.status_code}: {response.text}"
        )
        verified = response.json()
        assert verified.get("mask_sample") is True and verified.get("failure_kind") in {
            SESSION_LOST,
            SESSION_EXPIRED,
        }, f"verify of a closed session must be masked as session_lost or session_expired, got {verified}"
        assert verified.get("reward") == 0.0, f"a masked lost-session sample must carry reward 0, got {verified}"

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
            assert response.status_code == 409, (
                f"seed after close must get 409 for {late_seed.resources_session_id}, "
                f"got {response.status_code}: {response.text}"
            )
