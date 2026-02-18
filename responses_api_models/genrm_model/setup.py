# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from sys import platform

import setuptools

dependencies = [
    "nemo-gym[dev]",
    "hf_transfer==0.1.9",
    "uvicorn==0.40.0",
]

if platform == "darwin":
    dependencies.append("vllm==0.11.0")
else:
    dependencies.append("vllm==0.11.2")


setuptools.setup(install_requires=dependencies)
