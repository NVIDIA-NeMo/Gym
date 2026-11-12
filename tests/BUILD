# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
python_sources(sources=["conftest.py", "__init__.py", "unit_tests/conftest.py", "unit_tests/harbor_registry_fakes.py", "unit_tests/**/__init__.py"])
python_tests(
    name="pilot",
    sources=[
        "unit_tests/utils/test_text.py",
        "unit_tests/utils/test_install_lock.py",
        "unit_tests/config/test_secret_utils.py",
        "unit_tests/trajectory/test_model_call_record.py",
        "unit_tests/test_git_ref.py",
    ],
    dependencies=["//:project-config", "nemo_gym:runtime-config"],
)
