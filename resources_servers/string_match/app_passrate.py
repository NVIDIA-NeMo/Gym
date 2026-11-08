# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entrypoint; new configurations use app.py with strip_trailing_eos: true."""

from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchVerifyRequest,
    _extract_last_assistant_text,
    strip_trailing_eos,
)


class PassrateStringMatchResourcesServer(StringMatchResourcesServer):
    def _grading_text(self, body: StringMatchVerifyRequest) -> str:
        return strip_trailing_eos(_extract_last_assistant_text(body))


if __name__ == "__main__":
    PassrateStringMatchResourcesServer.run_webserver()
