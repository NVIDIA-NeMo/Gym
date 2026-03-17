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
import json

from app import VlmEvalKitResourcesServer


# From W&B table
fpath = ""
with open(fpath) as f:
    table = json.load(f)

rows = [json.loads(row[0]) | {"benchmark_name": "MMBench_DEV_EN_V11"} for row in table["data"]]

aggregate_metrics = VlmEvalKitResourcesServer._aggregate_MMBench_DEV_EN_V11(None, [rows])
print(json.dumps(aggregate_metrics, indent=4))
