# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/image2code/vision2code (Apache-2.0).
# Source: vision2code/generation/prompts.py; base revision 000f2e162dd62fac5637c718e018d7123b3aff1e.
# Modified source SHA-256: e90b11d4d738d79ec03aa6a4a63681a30d3e6e0215ab765ecb631a878f6bc95c (see NOTICE).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions Copyright 2026 Vision2Code Authors (Apache-2.0).
# Focused port; see NOTICE and tests/fixtures/provenance.json.
SYSTEM_PROMPT = (
    "You are an expert Python plotting assistant. "
    "Given an input image, write Python code to recreate it using matplotlib/numpy as needed. "
    "Save the image only to the Python variable OUTPUT_PATH. "
    "Do not quote OUTPUT_PATH and do not append a file extension to it. "
    "Return code only."
)
USER_PROMPT = "Generate Python code to recreate this image. Return code only."
