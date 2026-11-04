# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
python_sources(sources=["**/*.py"])
# Configs are consumed dynamically, so Python import inference cannot see them.
files(name="runtime-config", sources=["**/*.yaml", "**/*.json"])
