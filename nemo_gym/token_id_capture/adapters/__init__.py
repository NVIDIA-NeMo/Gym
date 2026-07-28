# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Engine-specific ``CaptureAdapter`` implementations (one module per engine).

Everything engine-blind -- record and digest build, the fail-closed
stage-then-respond ordering, coords assembly -- lives in
``token_id_capture/staging/capture.py``; an adapter contributes only how
prefix ids enter a request, how the exact prompt/generated ids and logprobs
come off a response, and the serving-layer hookup. These modules sit outside
the ``staging`` purity scope, though ``vllm.py`` happens to carry no engine
imports (it drives duck-typed request/response payloads).
"""
