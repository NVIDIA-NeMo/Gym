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
"""The names of the files and keys a run writes next to its rollouts."""

from __future__ import annotations


QUALITY_SUMMARY_FILENAME = "quality_summary.json"
ROLLOUT_VERDICTS_FILENAME = "rollout_verdicts.jsonl"
# The key under the quality summary's ``run`` that names the rollouts file(s) it was computed from, relative to
# the folder it is written in: the reports keep one name per folder whatever the rollouts file is called.
QUALITY_SUMMARY_SOURCES_KEY = "rollouts"


AGGREGATION_ERROR_KEY = "aggregation_error"
