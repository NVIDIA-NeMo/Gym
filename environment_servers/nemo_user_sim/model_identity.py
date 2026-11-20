# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the UserSim environment server on the model_identity benchmark's prepared spec."""

import os
from pathlib import Path

from benchmarks.model_identity.prepare import DATA
from environment_servers.nemo_user_sim.app import UserSimEnvironmentServer


SPEC_ENV_VAR = "USERSIM_IDENTITY_DISCLOSURE_SPEC"
# Written by prepare.py next to the prepared rows; it holds the expected identity.
SPEC_PATH = DATA.with_suffix(".spec.yaml")


def use_prepared_spec(spec_path: Path = SPEC_PATH) -> None:
    """Point UserSim at the prepared spec, replacing any exported value."""
    if not spec_path.is_file():
        raise FileNotFoundError(f"Prepared identity spec not found: {spec_path}. Run `gym eval prepare` first.")
    os.environ[SPEC_ENV_VAR] = str(spec_path)


if __name__ == "__main__":
    use_prepared_spec()
    UserSimEnvironmentServer.run_webserver()
