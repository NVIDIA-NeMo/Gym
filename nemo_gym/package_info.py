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


MAJOR = 0
MINOR = 2
PATCH = 0
PRE_RELEASE = "rc0"

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "nemo_gym"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/NeMo-Gym"
__repository_url__ = "https://github.com/NVIDIA-NeMo/NeMo-Gym"
__download_url__ = "https://github.com/NVIDIA-NeMo/NeMo-Gym/releases"
__description__ = "NeMo project description"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, pytorch, torch"
