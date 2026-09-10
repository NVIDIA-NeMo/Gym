# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable entrypoint for the core user-assistant processor."""

from nemo_gym.processors.user_assistant import UserAssistantProcessor


if __name__ == "__main__":
    UserAssistantProcessor.run_webserver()
