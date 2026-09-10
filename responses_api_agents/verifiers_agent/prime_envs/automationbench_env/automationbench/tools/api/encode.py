# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Stateless utility tools for the API toolset (not REST endpoints)."""

import base64


def base64_encode(text: str) -> str:
    """Encode text to base64url — the format Gmail API body fields require.

    The Gmail messages.send / drafts endpoints accept the message body only in
    base64url form: either ``raw`` (the full RFC 2822 message, base64url-encoded)
    or ``payload.body.data`` (the body text, base64url-encoded). Use this tool to
    produce that encoding, then pass the result to api_fetch. This is a local
    helper — it does not call any API endpoint.

    Args:
        text: Plaintext to encode (an email body, or a full RFC 2822 message).

    Returns:
        The base64url-encoded string.
    """
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")
