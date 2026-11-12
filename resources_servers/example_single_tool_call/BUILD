# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
python_sources(sources=["*.py"])
files(name="runtime-inputs", sources=["configs/*.yaml", "tests/*.jsonl", "data/*.jsonl", "requirements.txt"])
python_tests(
    name="tests",
    sources=["tests/test_*.py"],
    dependencies=[":runtime-inputs", "//:project-config", "nemo_gym:runtime-config"],
)
