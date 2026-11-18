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
"""Pure PyTorch knn_graph, for machines with no torch_cluster wheel (gRNAde only uses this one function).

Same result as torch_cluster for one point cloud: for each point, edges from its k nearest other points."""

import torch


def knn_graph(x, k, batch=None, loop=False, flow="source_to_target", cosine=False, num_workers=1):
    if batch is not None or cosine:
        raise NotImplementedError("the shim handles a single euclidean point cloud")
    n = x.size(0)
    dist = torch.cdist(x, x)
    if not loop:
        dist.fill_diagonal_(float("inf"))
    neighbours = dist.topk(min(k, n - (0 if loop else 1)), dim=1, largest=False).indices
    query = torch.arange(n, device=x.device).unsqueeze(1).expand_as(neighbours)
    source, target = neighbours.reshape(-1), query.reshape(-1)
    return torch.stack([source, target] if flow == "source_to_target" else [target, source])
