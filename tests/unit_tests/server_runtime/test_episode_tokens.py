# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from itsdangerous import URLSafeSerializer

from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.server_runtime.base_resources_server import mcp_session_token_serializer
from nemo_gym.server_runtime.episode_tokens import (
    _EPISODE_TOKEN_SALT,
    EPISODE_TOKEN_SECRET_ENV,
    EpisodeTokens,
    episode_token_secret,
)


SECRET = "EnvironmentServer___bird_interact"


def test_a_minted_token_reads_back_as_the_claims_the_server_keeps() -> None:
    tokens = EpisodeTokens(SECRET)
    episode_id = EpisodeId(rollout_id="3-0", attempt=2)

    token, kept = tokens.mint(episode_id)
    read = tokens.read(token)

    assert read is not None
    assert read.episode_id == episode_id
    assert kept.matches(read)


def test_each_grant_of_one_episode_gets_its_own_nonce() -> None:
    """Re-granting an episode, such as for a retried attempt, retires the earlier token."""
    tokens = EpisodeTokens(SECRET)
    episode_id = EpisodeId(rollout_id="3-0")

    first_token, _ = tokens.mint(episode_id)
    _, second = tokens.mint(episode_id)

    assert not second.matches(tokens.read(first_token))


def test_a_token_for_another_episode_does_not_match() -> None:
    """A neighbor's rollout ID is easy to guess; its nonce is not."""
    tokens = EpisodeTokens(SECRET)
    _, kept = tokens.mint(EpisodeId(rollout_id="3-0"))
    other_token, _ = tokens.mint(EpisodeId(rollout_id="4-0"))

    assert not kept.matches(tokens.read(other_token))


def test_a_forged_token_with_a_guessed_nonce_does_not_match() -> None:
    """Even a signature made with the server's key fails without the episode's nonce."""
    tokens = EpisodeTokens(SECRET)
    episode_id = EpisodeId(rollout_id="3-0")
    _, kept = tokens.mint(episode_id)
    forged = URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).dumps({"episode": "3-0", "nonce": "guess"})

    read = tokens.read(forged)

    assert read is not None and read.episode_id == episode_id
    assert not kept.matches(read)


@pytest.mark.parametrize(
    "token",
    [
        pytest.param("not-a-token", id="garbage"),
        pytest.param(URLSafeSerializer("other-secret", salt=_EPISODE_TOKEN_SALT).dumps({}), id="other-key"),
        pytest.param(mcp_session_token_serializer(SECRET).dumps({"sid": "3-0"}), id="mcp-token-same-key"),
        pytest.param(
            URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).dumps({"episode": "3-0"}), id="missing-nonce"
        ),
        pytest.param(
            URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).dumps({"episode": "", "nonce": "n"}),
            id="invalid-episode",
        ),
        pytest.param(URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).dumps(["3-0", "n"]), id="not-an-object"),
        pytest.param(
            URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).make_signer().sign(b"not base64 json!").decode(),
            id="signed-undecodable-payload",
        ),
        pytest.param(
            URLSafeSerializer(SECRET, salt=_EPISODE_TOKEN_SALT).dumps({"episode": "3-0", "nonce": "\u00e9"}),
            id="non-ascii-nonce",
        ),
    ],
)
def test_a_string_this_server_did_not_mint_reads_as_none(token: str) -> None:
    assert EpisodeTokens(SECRET).read(token) is None


def test_a_tampered_token_reads_as_none() -> None:
    tokens = EpisodeTokens(SECRET)
    token, _ = tokens.mint(EpisodeId(rollout_id="3-0"))
    payload, signature = token.rsplit(".", 1)

    assert tokens.read(f"{payload}.{signature[::-1]}") is None


def test_the_secret_is_random_for_each_run_and_shared_with_the_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The main process makes the secret; a worker started after it inherits the same one from the environment."""
    monkeypatch.delenv(EPISODE_TOKEN_SECRET_ENV, raising=False)
    main_process = episode_token_secret()
    assert main_process != SECRET and len(main_process) >= 32
    assert episode_token_secret() == main_process

    token, kept = EpisodeTokens(main_process).mint(EpisodeId(rollout_id="3-0"))
    worker = EpisodeTokens(episode_token_secret())
    assert kept.matches(worker.read(token))

    monkeypatch.delenv(EPISODE_TOKEN_SECRET_ENV)
    assert episode_token_secret() != main_process
