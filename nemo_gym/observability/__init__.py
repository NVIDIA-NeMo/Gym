# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""NeMo Gym observability: JSONL event recording, OTel trace export, and W&B artifact upload.

The module-level context-var API is the primary integration surface:

    from nemo_gym.observability import observability_span, event_context

    with event_context(trajectory_id="task-0-rollout-0"):
        async with observability_span("sandbox.exec", phase="execution",
                                      attributes={"command": cmd}):
            result = await sandbox.exec(cmd)

Spans are recorded to JSONL on disk and emitted as OTel traces to any configured
TracerProvider (e.g. NeMo Lens → Honeycomb / Grafana Tempo).

Configure via environment variables (prefix NEMO_GYM_OBSERVABILITY_*) or Hydra
config keys. See ``build_recorder_from_env()`` and ``build_recorder_from_config()``
for the full option list.
"""

from nemo_gym.observability.events import (
    SCHEMA_VERSION,
    classify_command,
    command_attributes,
    stable_hash,
)
from nemo_gym.observability.recorder import (
    SandboxEventRecorder,
    build_recorder_from_config,
    build_recorder_from_env,
    current_recorder,
    ensure_env_recorder,
    event_context,
    log_wandb_artifact,
    observability_span,
    observability_sync_span,
    push_event_context,
    record_event,
    reset_current_recorder,
    reset_event_context,
    set_current_recorder,
    suppress_observability_events,
    use_recorder,
)
from nemo_gym.observability.resource import SandboxResourceSampler
from nemo_gym.observability.summary import (
    flatten_summary_metrics,
    summarize_observability,
    write_summary,
)
from nemo_gym.observability.traces import export_trace_artifacts
from nemo_gym.observability.trajectory_trace import (
    extract_agent_tool_spans,
    ingest_agent_trajectory_events,
)


__all__ = [
    "SCHEMA_VERSION",
    "SandboxEventRecorder",
    "SandboxResourceSampler",
    "build_recorder_from_config",
    "build_recorder_from_env",
    "classify_command",
    "command_attributes",
    "current_recorder",
    "ensure_env_recorder",
    "event_context",
    "export_trace_artifacts",
    "extract_agent_tool_spans",
    "flatten_summary_metrics",
    "ingest_agent_trajectory_events",
    "log_wandb_artifact",
    "observability_span",
    "observability_sync_span",
    "push_event_context",
    "record_event",
    "reset_current_recorder",
    "reset_event_context",
    "set_current_recorder",
    "stable_hash",
    "summarize_observability",
    "suppress_observability_events",
    "use_recorder",
    "write_summary",
]
