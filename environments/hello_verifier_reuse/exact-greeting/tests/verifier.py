# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from nemo_gym.verifiers.files import text_file_equals


verify = partial(
    text_file_equals,
    path="hello-gym.txt",
    content="Hello from NeMo Gym!",
)
