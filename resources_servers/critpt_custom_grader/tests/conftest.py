# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registers the scripted fake provider under ``critpt_fake`` so config-driven tests can select it by name."""

from nemo_gym.sandbox import register_provider
from nemo_gym.sandbox.providers.registry import list_providers
from resources_servers.critpt_custom_grader.tests.support import ScriptedProvider


if ScriptedProvider.name not in list_providers():
    register_provider(ScriptedProvider.name, ScriptedProvider)
