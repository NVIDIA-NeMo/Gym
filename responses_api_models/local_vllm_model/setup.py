# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

    # We specifically pin the vllm dependency because we have tested on this version.
    # Updated Thu Dec 11, 2025 with vllm==0.11.2
    # License: Apache 2.0 https://github.com/vllm-project/vllm/blob/89988ec8c2a0c3e18e63767d9df5ca8f6b8ff21c/LICENSE
    # "vllm==0.11.2",
    # VLLM is resolved below since installation on Macs requires special workarounds.

    # hf_transfer for faster model download from HuggingFace
    # Updated Mon Jan 05, 2026 with vllm==0.1.9
    # License: Apache 2.0 https://github.com/huggingface/hf_transfer/blob/51499cc4ff0fe218082e13f27881a06811913751/LICENSE
    "hf_transfer==0.1.9",
]

# We use an older version of vllm on Macs since that is the latest version pip-installable. We may need to version tests later on.
if platform == "darwin":
    dependencies.append("vllm==0.11.0")
else:
    dependencies.append("vllm==0.13.0")


setuptools.setup(install_requires=dependencies)
