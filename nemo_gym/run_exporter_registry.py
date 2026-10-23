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
"""Names of the exporters `gym eval export` can run.

Kept apart from `nemo_gym.exporters` so the CLI can list them in `--help` without importing pydantic
or any backend.
"""

# Exporter name -> "module:class". Classes load on demand.
RUN_EXPORTERS: dict[str, str] = {
    "atif": "nemo_gym.exporters.atif:AtifExporter",
    "evalrdl": "nemo_gym.exporters.evalrdl:EvalRDLExporter",
    "mlflow": "nemo_gym.exporters.mlflow:MLflowExporter",
    "wandb": "nemo_gym.exporters.wandb:WandbExporter",
}
# Exporter name -> "module:class" of the model holding its settings, importable without the exporter's SDK.
RUN_EXPORTER_CONFIGS: dict[str, str] = {
    "atif": "nemo_gym.atif_export:ExportAtifConfig",
    "evalrdl": "nemo_gym.exporters.evalrdl:EvalRDLExportConfig",
    "mlflow": "nemo_gym.config_types:MLflowExportConfig",
    "wandb": "nemo_gym.config_types:WandbExportConfig",
}
DEFAULT_RUN_EXPORTER = "atif"
