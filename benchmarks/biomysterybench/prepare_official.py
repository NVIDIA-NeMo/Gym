# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Anthropic's original 99-task BioMysteryBench release."""

from pathlib import Path

from benchmarks.biomysterybench.prepare import prepare as _prepare


def prepare() -> Path:
    """Gym preparation entrypoint for the published 99-task release."""

    return _prepare(release_name="official-99")


if __name__ == "__main__":
    prepare()
