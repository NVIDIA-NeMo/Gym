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

"""The installable conformance kit: golden call sequences -> byte-exact records.

Every framework's ``TokenSink``/``TokenSource`` implementation and every
engine adapter runs these fixtures in its own CI (see ``kit.py``); the same
fixtures are replayed through the gate in S3 and must produce byte-identical
manifests. Conformance is tested, not trusted.
"""

from nemo_gym.token_id_capture.conformance.kit import (
    ConformanceFailure,
    build_fixture_artifacts,
    fixture_names,
    load_fixture,
    run_lineage_conformance,
    run_sink_source_conformance,
)


__all__ = [
    "ConformanceFailure",
    "build_fixture_artifacts",
    "fixture_names",
    "load_fixture",
    "run_lineage_conformance",
    "run_sink_source_conformance",
]
