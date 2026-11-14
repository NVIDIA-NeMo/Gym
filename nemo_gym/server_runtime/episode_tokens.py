# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tokens that let an agent call one episode's tools on an Environment Server."""

import hmac
import os
import secrets
from dataclasses import dataclass

from itsdangerous import BadData, URLSafeSerializer
from pydantic import ValidationError

from nemo_gym.core.episode_types import EpisodeId


# The header that carries an episode token on a direct HTTP tool call.
EPISODE_TOKEN_HEADER = "X-NeMo-Gym-Episode-Token"
# Namespaces the signature, so an episode token cannot pass for another token signed with the same key.
_EPISODE_TOKEN_SALT = "nemo-gym-episode-token"
# The signing secret of this server run. Workers inherit it from the main process.
EPISODE_TOKEN_SECRET_ENV = "NEMO_GYM_EPISODE_TOKEN_SECRET"


def episode_token_secret() -> str:
    """The random secret that signs this server run's episode tokens.

    The first call in a process without the secret makes one and sets it in ``EPISODE_TOKEN_SECRET_ENV``. A
    multi-worker server builds its server object in the main process before uvicorn starts the workers, so every
    worker inherits the same secret and reads tokens that another worker minted.
    """
    secret = os.environ.get(EPISODE_TOKEN_SECRET_ENV)
    if not secret:
        secret = os.environ[EPISODE_TOKEN_SECRET_ENV] = secrets.token_urlsafe(32)
    return secret


@dataclass(frozen=True)
class EpisodeTokenClaims:
    """The episode a token names, and the nonce that makes the token unguessable."""

    episode_id: EpisodeId
    nonce: str

    def matches(self, other: "EpisodeTokenClaims") -> bool:
        """Whether ``other`` names the same episode with the same nonce, compared in constant time."""
        return self.episode_id == other.episode_id and hmac.compare_digest(self.nonce, other.nonce)


class EpisodeTokens:
    """Mint and read the tokens an Environment Server grants for one episode's tools.

    A rollout ID is predictable (``<task index>-<rollout index>`` unless the caller sets one), so each token
    carries a random nonce. The server accepts a token only while it holds the same claims for that episode
    (``EpisodeTokenClaims.matches``): the nonce is what authenticates a call. The signature, with a random secret
    for each server run (``episode_token_secret``), keeps the claims intact, and lets a worker that does not hold
    the episode read which episode a token names.
    """

    def __init__(self, secret: str) -> None:
        self._serializer = URLSafeSerializer(secret, salt=_EPISODE_TOKEN_SALT)

    def mint(self, episode_id: EpisodeId) -> tuple[str, EpisodeTokenClaims]:
        """A new token for ``episode_id``, and the claims the server keeps to check it."""
        claims = EpisodeTokenClaims(episode_id=episode_id, nonce=secrets.token_urlsafe(32))
        token = self._serializer.dumps({"episode": episode_id.capture_key, "nonce": claims.nonce})
        return token, claims

    def read(self, token: str) -> EpisodeTokenClaims | None:
        """The claims of a token this server signed, or ``None`` for any other string. It never raises."""
        try:
            payload = self._serializer.loads(token)
        except BadData:
            # A bad signature, or a valid signature over a payload that does not decode.
            return None
        if not isinstance(payload, dict):
            return None
        capture_key, nonce = payload.get("episode"), payload.get("nonce")
        # A minted nonce is URL-safe ASCII; a non-ASCII one could not be compared in constant time.
        if not isinstance(capture_key, str) or not isinstance(nonce, str) or not nonce or not nonce.isascii():
            return None
        try:
            episode_id = EpisodeId.from_capture_key(capture_key)
        except (ValueError, ValidationError):
            return None
        return EpisodeTokenClaims(episode_id=episode_id, nonce=nonce)
