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
from generators.direct import generate_direct
from generators.error_correction import generate_error_correction
from generators.multistep import generate_multistep_related, generate_multistep_unrelated
from generators.schema_only import generate_schema_only
from generators.translation import generate_translation


ALL_GENERATORS = {
    "direct": generate_direct,
    "translation": generate_translation,
    "multistep_related": generate_multistep_related,
    "multistep_unrelated": generate_multistep_unrelated,
    "schema_only": generate_schema_only,
    "error_correction": generate_error_correction,
}
