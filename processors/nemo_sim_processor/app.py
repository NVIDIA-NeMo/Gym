# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable entrypoint for NeMoSimProcessor."""

from nemo_gym.processors.nemo_sim_processor import NeMoSimProcessor


if __name__ == "__main__":
    NeMoSimProcessor.run_webserver()
