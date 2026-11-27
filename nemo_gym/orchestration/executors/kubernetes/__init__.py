# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The Kubernetes executor: ``KubernetesExecutor``.

Its parts, each in its own module:

- ``executor``: render a run's Job manifest, submit it with kubectl and record the submission;
- ``script``: build the manifest and the driver script it runs.
"""

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.orchestration.executors.kubernetes.executor import KubernetesExecutor


__all__ = ["KubernetesExecutor"]

__getattr__ = moved_names_getattr(__name__)
