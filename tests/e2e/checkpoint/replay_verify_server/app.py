# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint e2e suite's slow-verify resources server, declaring its ``/verify`` safe to replay."""

from tests.e2e.checkpoint.slow_verify_server.app import SlowVerifyResourcesServer


class ReplayVerifyResourcesServer(SlowVerifyResourcesServer):
    checkpoint_verify = "replay"


if __name__ == "__main__":
    ReplayVerifyResourcesServer.run_webserver()
