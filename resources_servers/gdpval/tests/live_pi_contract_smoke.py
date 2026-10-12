# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the published Pi smoke command working; new runs can use live_sandbox_smoke."""

from resources_servers.gdpval.tests.live_sandbox_smoke import main


if __name__ == "__main__":
    main()
