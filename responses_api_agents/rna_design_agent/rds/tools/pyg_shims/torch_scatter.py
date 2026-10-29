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
"""Pure PyTorch scatter_add, for machines with no torch_scatter wheel (gRNAde only uses this one function)."""

import torch


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    dim = dim % src.dim()
    if index.dim() == 1 and src.dim() > 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(src)
    if out is None:
        size = list(src.shape)
        size[dim] = dim_size if dim_size is not None else (int(index.max()) + 1 if index.numel() else 0)
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)
